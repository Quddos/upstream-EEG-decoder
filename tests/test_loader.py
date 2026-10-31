"""Tests for EDF loading, channel standardisation and recording validation."""

from __future__ import annotations

from dataclasses import replace

import mne
import numpy as np
import pytest

from src.config import DatasetConfig
from src.data.loader import (
    RecordingValidationError,
    annotations_table,
    load_raw,
    standardize_channel_names,
    summarize_recording,
    validate_raw,
)
from src.data.physionet_spec import (
    EDF_CHANNEL_NAMES,
    N_EEG_CHANNELS,
    SAMPLING_RATE_HZ,
    RunSpec,
)


class TestChannelStandardisation:
    def test_strips_edf_padding_dots(self):
        assert standardize_channel_names(["Fc5.", "C3..", "Iz.."]) == ["FC5", "C3", "Iz"]

    def test_all_64_edf_names_resolve_to_montage_positions(self):
        """If any name failed to resolve, its electrode position would be missing
        and every topographic plot would silently be wrong."""
        standardized = standardize_channel_names(EDF_CHANNEL_NAMES)
        montage = mne.channels.make_standard_montage("standard_1005")
        assert len(standardized) == N_EEG_CHANNELS
        missing = [ch for ch in standardized if ch not in montage.ch_names]
        assert missing == [], f"channels not in montage: {missing}"

    def test_output_names_are_unique(self):
        assert len(set(standardize_channel_names(EDF_CHANNEL_NAMES))) == N_EEG_CHANNELS

    def test_hand_knob_channels_present(self):
        standardized = standardize_channel_names(EDF_CHANNEL_NAMES)
        for channel in ("C3", "C4", "Cz"):
            assert channel in standardized

    def test_unknown_names_pass_through_unchanged(self):
        assert standardize_channel_names(["NOTACHANNEL."]) == ["NOTACHANNEL"]


class TestLoading:
    def test_loads_with_expected_shape(self, synthetic_config: DatasetConfig):
        raw = load_raw(RunSpec(1, 4), synthetic_config)
        assert raw.info["sfreq"] == SAMPLING_RATE_HZ
        assert len(mne.pick_types(raw.info, eeg=True)) == N_EEG_CHANNELS
        assert raw.n_times > 0

    def test_channel_names_are_standardised_after_load(self, synthetic_config):
        raw = load_raw(RunSpec(1, 4), synthetic_config)
        assert "C3" in raw.ch_names
        assert not any("." in ch for ch in raw.ch_names)

    def test_montage_is_attached(self, synthetic_config):
        raw = load_raw(RunSpec(1, 4), synthetic_config)
        montage = raw.get_montage()
        assert montage is not None
        positions = montage.get_positions()["ch_pos"]
        assert all(np.all(np.isfinite(positions[ch])) for ch in raw.ch_names)

    def test_loading_is_idempotent(self, synthetic_config):
        """A second load must reuse the file on disk, not regenerate it."""
        spec = RunSpec(1, 8)
        first = load_raw(spec, synthetic_config)
        second = load_raw(spec, synthetic_config)
        np.testing.assert_allclose(first.get_data(), second.get_data())


class TestValidation:
    def test_wrong_sampling_rate_is_rejected(self, synthetic_config):
        """This is what protects the pipeline from the subjects reported to be
        recorded at a non-standard rate -- detected by measurement, not blacklist."""
        raw = load_raw(RunSpec(1, 4), synthetic_config)
        raw.resample(128.0, verbose="ERROR")
        with pytest.raises(RecordingValidationError, match="sampling rate"):
            validate_raw(raw, RunSpec(1, 4))

    def test_wrong_channel_count_is_rejected(self, synthetic_config):
        raw = load_raw(RunSpec(1, 4), synthetic_config)
        raw.drop_channels(["C3", "C4"])
        with pytest.raises(RecordingValidationError, match="EEG channels"):
            validate_raw(raw, RunSpec(1, 4))

    def test_unexpected_annotation_vocabulary_is_rejected(self, synthetic_config):
        raw = load_raw(RunSpec(1, 4), synthetic_config)
        raw.set_annotations(mne.Annotations(onset=[1.0], duration=[1.0], description=["T9"]))
        with pytest.raises(RecordingValidationError, match="annotation"):
            validate_raw(raw, RunSpec(1, 4))

    def test_conforming_recording_passes(self, synthetic_config):
        raw = load_raw(RunSpec(1, 4), synthetic_config)
        validate_raw(raw, RunSpec(1, 4))  # must not raise


class TestInspection:
    def test_summary_reports_measured_properties(self, synthetic_config):
        summary = summarize_recording(RunSpec(1, 4), synthetic_config)
        assert summary.subject_id == "S001"
        assert summary.run_id == "R04"
        assert summary.task_type == "unilateral_imagery"
        assert summary.sampling_rate == SAMPLING_RATE_HZ
        assert summary.n_eeg_channels == N_EEG_CHANNELS
        assert summary.duration_s > 100
        assert len(summary.sha256) == 64

    def test_summary_resolves_classes_per_run(self, synthetic_config):
        summary = summarize_recording(RunSpec(1, 4), synthetic_config)
        assert "LEFT_HAND_IMAGERY" in summary.class_counts
        assert "RIGHT_HAND_IMAGERY" in summary.class_counts
        assert summary.event_counts["T1"] == summary.class_counts["LEFT_HAND_IMAGERY"]
        assert summary.event_counts["T2"] == summary.class_counts["RIGHT_HAND_IMAGERY"]

    def test_annotation_table_has_labels_and_monotonic_onsets(self, synthetic_config):
        spec = RunSpec(1, 4)
        raw = load_raw(spec, synthetic_config, preload=False)
        table = annotations_table(spec, raw)
        assert len(table) > 0
        assert table["class_label"].notna().all()
        assert table["onset_time"].is_monotonic_increasing
        assert set(table["event_code"]) <= {"T0", "T1", "T2"}

    def test_annotation_onsets_match_samples(self, synthetic_config):
        spec = RunSpec(1, 4)
        raw = load_raw(spec, synthetic_config, preload=False)
        table = annotations_table(spec, raw)
        expected = (table["onset_time"] * SAMPLING_RATE_HZ).round().astype(int)
        assert (table["onset_sample"] == expected).all()


class TestSourceProvenanceGuard:
    """Simulated data must never be reportable as real, or vice versa.

    Source-scoped paths make mixing unlikely; this guard makes it impossible.
    Reporting a synthetic fixture as ``source: physionet`` would be a scientific
    integrity failure, not merely a bug.
    """

    def test_synthetic_fixtures_are_marked(self, synthetic_config):
        from src.data.loader import is_synthetic_recording

        raw = load_raw(RunSpec(1, 4), synthetic_config)
        assert is_synthetic_recording(raw) is True

    def test_fixture_read_as_physionet_is_rejected(self, synthetic_config):
        """The dangerous direction: simulated data reported as real."""
        from src.data.loader import DataSourceMismatchError, verify_source

        raw = load_raw(RunSpec(1, 4), synthetic_config)
        physionet_config = replace(synthetic_config, source="physionet")
        with pytest.raises(DataSourceMismatchError, match="synthetic-fixture marker"):
            verify_source(raw, RunSpec(1, 4), physionet_config)

    def test_unmarked_recording_read_as_synthetic_is_rejected(self, synthetic_config):
        """The other direction: an offline 'unit test' silently using real data."""
        from src.data.loader import DataSourceMismatchError, verify_source

        raw = load_raw(RunSpec(1, 4), synthetic_config)
        # Strip the marker to imitate a real recording.
        with raw.info._unlock():
            raw.info["subject_info"] = {"his_id": "X"}
        with pytest.raises(DataSourceMismatchError, match="does not carry"):
            verify_source(raw, RunSpec(1, 4), synthetic_config)

    def test_matching_source_passes(self, synthetic_config):
        from src.data.loader import verify_source

        raw = load_raw(RunSpec(1, 4), synthetic_config)
        verify_source(raw, RunSpec(1, 4), synthetic_config)  # must not raise

    def test_summary_records_provenance_fields(self, synthetic_config):
        summary = summarize_recording(RunSpec(1, 4), synthetic_config)
        assert summary.source == "synthetic"
        assert summary.file_bytes > 0
        assert len(summary.sha256) == 64
        row = summary.to_row()
        for field in ("source", "dataset_version", "download_source", "acquisition_date"):
            assert field in row
