"""Tests for filtering, referencing, normalisation and epoch extraction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import welch

from src.config import DatasetConfig, PreprocessingConfig
from src.data.loader import load_raw
from src.data.physionet_spec import (
    CLASS_TO_INDEX,
    LEFT_HAND_IMAGERY,
    REST,
    RIGHT_HAND_IMAGERY,
    SAMPLING_RATE_HZ,
    RunSpec,
)
from src.preprocessing.epoching import (
    EpochingError,
    build_events,
    extract_epochs,
    subsample_rest,
)
from src.preprocessing.filters import (
    PreprocessingError,
    apply_bandpass,
    normalize_epochs,
    preprocess_raw,
)


class TestFiltering:
    def test_preprocessing_preserves_sample_count(self, synthetic_config):
        """Filtering and re-referencing must not resample or truncate the signal.

        This is the 'preprocessing does not change sample count unexpectedly'
        check: a filter that silently decimated would corrupt every downstream
        onset_sample without any other visible symptom.
        """
        raw = load_raw(RunSpec(1, 4), synthetic_config)
        before_samples = raw.n_times
        before_sfreq = raw.info["sfreq"]
        before_channels = len(raw.ch_names)

        processed = preprocess_raw(raw, synthetic_config.preprocessing)

        assert processed.n_times == before_samples
        assert processed.info["sfreq"] == before_sfreq
        assert len(processed.ch_names) == before_channels

    def test_preprocessing_preserves_annotations(self, synthetic_config):
        raw = load_raw(RunSpec(1, 4), synthetic_config)
        before = len(raw.annotations)
        processed = preprocess_raw(raw, synthetic_config.preprocessing)
        assert len(processed.annotations) == before

    def test_bandpass_attenuates_out_of_band_power(self, synthetic_config):
        raw = load_raw(RunSpec(1, 4), synthetic_config)
        pre = synthetic_config.preprocessing
        before = raw.get_data()
        after = preprocess_raw(raw, pre).get_data()

        freqs, psd_before = welch(before, fs=SAMPLING_RATE_HZ, nperseg=512, axis=-1)
        _, psd_after = welch(after, fs=SAMPLING_RATE_HZ, nperseg=512, axis=-1)

        in_band = (freqs >= pre.l_freq + 1) & (freqs <= pre.h_freq - 1)
        out_band = (freqs > pre.h_freq + 10) & (freqs < SAMPLING_RATE_HZ / 2 - 5)

        # In-band power largely survives; out-of-band power is crushed.
        in_ratio = psd_after[:, in_band].mean() / psd_before[:, in_band].mean()
        out_ratio = psd_after[:, out_band].mean() / psd_before[:, out_band].mean()
        assert in_ratio > 0.3
        assert out_ratio < 1e-3
        assert out_ratio < in_ratio

    def test_common_average_reference_zeroes_channel_mean(self, synthetic_config):
        raw = load_raw(RunSpec(1, 4), synthetic_config)
        processed = preprocess_raw(raw, synthetic_config.preprocessing)
        across_channels = processed.get_data().mean(axis=0)
        assert np.allclose(across_channels, 0.0, atol=1e-15)

    def test_car_can_be_disabled(self, synthetic_config):
        pre = replace(synthetic_config.preprocessing, common_average_reference=False)
        raw = load_raw(RunSpec(1, 4), synthetic_config)
        processed = preprocess_raw(raw, pre)
        assert not np.allclose(processed.get_data().mean(axis=0), 0.0, atol=1e-15)

    def test_bandpass_above_nyquist_is_rejected(self, synthetic_config):
        raw = load_raw(RunSpec(1, 4), synthetic_config)
        with pytest.raises(PreprocessingError, match="Nyquist"):
            apply_bandpass(raw, replace(synthetic_config.preprocessing, h_freq=100.0))

    def test_inverted_passband_is_rejected(self, synthetic_config):
        raw = load_raw(RunSpec(1, 4), synthetic_config)
        with pytest.raises(PreprocessingError, match="passband"):
            apply_bandpass(
                raw, replace(synthetic_config.preprocessing, l_freq=30.0, h_freq=8.0)
            )


class TestNormalization:
    def test_zscore_per_channel_standardises_each_channel(self):
        rng = np.random.default_rng(0)
        data = rng.normal(loc=5.0, scale=3.0, size=(4, 8, 100))
        out = normalize_epochs(data, "zscore_per_channel")
        np.testing.assert_allclose(out.mean(axis=2), 0.0, atol=1e-10)
        np.testing.assert_allclose(out.std(axis=2), 1.0, atol=1e-10)

    def test_zscore_per_epoch_uses_pooled_statistics(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(4, 8, 100))
        out = normalize_epochs(data, "zscore_per_epoch")
        np.testing.assert_allclose(out.mean(axis=(1, 2)), 0.0, atol=1e-10)
        np.testing.assert_allclose(out.std(axis=(1, 2)), 1.0, atol=1e-10)

    def test_normalisation_is_computed_within_each_epoch(self):
        """No statistic may be shared across epochs -- that would leak across a
        future train/test split."""
        rng = np.random.default_rng(2)
        data = rng.normal(size=(6, 4, 50))
        full = normalize_epochs(data, "zscore_per_channel")
        subset = normalize_epochs(data[:3], "zscore_per_channel")
        np.testing.assert_allclose(full[:3], subset, atol=1e-12)

    def test_flat_channel_does_not_produce_nan(self):
        data = np.zeros((2, 3, 20))
        out = normalize_epochs(data, "zscore_per_channel")
        assert np.isfinite(out).all()

    def test_none_is_a_passthrough(self):
        data = np.arange(24, dtype=float).reshape(2, 3, 4)
        np.testing.assert_array_equal(normalize_epochs(data, "none"), data)

    def test_unknown_method_rejected(self):
        with pytest.raises(PreprocessingError):
            normalize_epochs(np.zeros((1, 1, 4)), "not-a-method")


class TestEventBuilding:
    def test_events_use_run_specific_labels(self, synthetic_config):
        spec = RunSpec(1, 4)
        raw = load_raw(spec, synthetic_config)
        events, event_id = build_events(raw, spec, [REST, LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY])
        assert event_id[LEFT_HAND_IMAGERY] == CLASS_TO_INDEX[LEFT_HAND_IMAGERY]
        assert event_id[RIGHT_HAND_IMAGERY] == CLASS_TO_INDEX[RIGHT_HAND_IMAGERY]
        assert set(events[:, 2]) <= set(CLASS_TO_INDEX.values())

    def test_class_filter_excludes_rest(self, synthetic_config):
        spec = RunSpec(1, 4)
        raw = load_raw(spec, synthetic_config)
        events, event_id = build_events(raw, spec, [LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY])
        assert REST not in event_id
        assert CLASS_TO_INDEX[REST] not in set(events[:, 2])

    def test_bilateral_run_rejected_as_not_a_phase1_class(self, synthetic_config):
        """Run 6 produces BOTH_FISTS/BOTH_FEET, which are not Phase 1 classes; the
        pipeline must refuse rather than silently mislabel them as left/right."""
        spec = RunSpec(1, 6)
        config = replace(synthetic_config, runs=(6,))
        raw = load_raw(spec, config)
        with pytest.raises(EpochingError, match="not a Phase 1 class"):
            build_events(raw, spec, [REST, LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY])

    def test_execution_run_rejected(self, synthetic_config):
        """Run 3 is left/right but *executed*, not imagined -- accepting it would
        silently mix motor execution into a motor-imagery dataset."""
        spec = RunSpec(1, 3)
        config = replace(synthetic_config, runs=(3,))
        raw = load_raw(spec, config)
        with pytest.raises(EpochingError, match="not a Phase 1 class"):
            build_events(raw, spec, [REST, LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY])


class TestRestSubsampling:
    def test_caps_rest_count(self):
        rest = CLASS_TO_INDEX[REST]
        left = CLASS_TO_INDEX[LEFT_HAND_IMAGERY]
        events = np.array([[i * 100, 0, rest if i % 2 == 0 else left] for i in range(20)])
        out = subsample_rest(events, max_per_run=3, seed=0)
        assert (out[:, 2] == rest).sum() == 3
        assert (out[:, 2] == left).sum() == (events[:, 2] == left).sum()

    def test_no_cap_keeps_everything(self):
        events = np.array([[i * 100, 0, CLASS_TO_INDEX[REST]] for i in range(10)])
        np.testing.assert_array_equal(subsample_rest(events, None, 0), events)

    def test_retained_rest_spans_the_run(self):
        """Evenly spaced, not clustered -- retained rest must cover the whole run."""
        rest = CLASS_TO_INDEX[REST]
        events = np.array([[i * 100, 0, rest] for i in range(30)])
        out = subsample_rest(events, max_per_run=5, seed=0)
        assert out[0, 0] == 0
        assert out[-1, 0] == 2900


class TestEpochExtraction:
    def test_epoch_shape_is_exactly_640_samples(self, synthetic_config):
        """4 s at 160 Hz with a half-open window. A closed window would give 641
        and break any fixed-input model."""
        spec = RunSpec(1, 4)
        raw = preprocess_raw(load_raw(spec, synthetic_config), synthetic_config.preprocessing)
        result = extract_epochs(raw, spec, synthetic_config)
        data = result.data
        assert data.ndim == 3
        assert data.shape[1] == 64
        assert data.shape[2] == 640
        assert data.shape[2] == synthetic_config.preprocessing.expected_epoch_samples

    def test_metadata_aligned_with_data(self, synthetic_config):
        spec = RunSpec(1, 4)
        raw = preprocess_raw(load_raw(spec, synthetic_config), synthetic_config.preprocessing)
        result = extract_epochs(raw, spec, synthetic_config)
        assert len(result.metadata) == result.data.shape[0]
        assert len(result.labels) == result.data.shape[0]

    def test_epoch_window_bounds_are_consistent(self, synthetic_config):
        spec = RunSpec(1, 4)
        raw = preprocess_raw(load_raw(spec, synthetic_config), synthetic_config.preprocessing)
        meta = extract_epochs(raw, spec, synthetic_config).metadata
        pre = synthetic_config.preprocessing
        np.testing.assert_allclose(meta["epoch_start"], meta["onset_time"] + pre.epoch_tmin)
        np.testing.assert_allclose(meta["epoch_end"], meta["onset_time"] + pre.epoch_tmax)
        assert (meta["epoch_end"] > meta["epoch_start"]).all()

    def test_all_three_classes_extracted(self, synthetic_config):
        spec = RunSpec(1, 4)
        raw = preprocess_raw(load_raw(spec, synthetic_config), synthetic_config.preprocessing)
        labels = set(extract_epochs(raw, spec, synthetic_config).metadata["class_label"])
        assert labels == {REST, LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY}

    def test_epochs_lie_inside_the_recording(self, synthetic_config):
        spec = RunSpec(1, 4)
        raw = preprocess_raw(load_raw(spec, synthetic_config), synthetic_config.preprocessing)
        result = extract_epochs(raw, spec, synthetic_config)
        duration = raw.n_times / raw.info["sfreq"]
        assert (result.metadata["epoch_start"] >= 0).all()
        assert (result.metadata["epoch_end"] <= duration).all()

    def test_longer_window_changes_sample_count_predictably(self, synthetic_config):
        config = replace(
            synthetic_config,
            preprocessing=replace(synthetic_config.preprocessing, epoch_tmax=2.0),
        )
        spec = RunSpec(1, 4)
        raw = preprocess_raw(load_raw(spec, config), config.preprocessing)
        assert extract_epochs(raw, spec, config).data.shape[2] == 320
