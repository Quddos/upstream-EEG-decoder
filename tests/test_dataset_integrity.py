"""End-to-end integrity checks on the built dataset.

This module implements the validation checklist required by the Phase 1 brief:

    * expected sampling frequency
    * expected EEG channel count
    * no missing labels
    * no empty epochs
    * valid epoch dimensions
    * subject IDs are preserved
    * left/right classes exist
    * preprocessing does not change sample count unexpectedly
      (the continuous-signal half is in test_preprocessing.py; the stored-dataset
      half is here)
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.config import DatasetConfig
from src.data.physionet_spec import (
    CLASS_TO_INDEX,
    LEFT_HAND_IMAGERY,
    N_EEG_CHANNELS,
    PHASE1_CLASSES,
    REST,
    RIGHT_HAND_IMAGERY,
    SAMPLING_RATE_HZ,
)
from src.io.store import EEGDataset, load_dataset, load_summary
from src.preprocessing.epoching import METADATA_COLUMNS
from src.viz.figures import lateralization_index
from tests.conftest import TEST_SUBJECTS


class TestRequiredValidations:
    """The eight checks the brief names explicitly."""

    def test_expected_sampling_frequency(self, dataset: EEGDataset):
        assert (dataset.metadata["sampling_rate"] == SAMPLING_RATE_HZ).all()

    def test_expected_eeg_channel_count(self, dataset: EEGDataset):
        assert dataset.X.shape[1] == N_EEG_CHANNELS
        assert len(dataset.channels) == N_EEG_CHANNELS
        assert (dataset.metadata["channel_count"] == N_EEG_CHANNELS).all()

    def test_no_missing_labels(self, dataset: EEGDataset):
        assert dataset.metadata["class_label"].notna().all()
        assert dataset.metadata["class_index"].notna().all()
        assert not (dataset.metadata["class_label"].astype(str) == "").any()
        assert set(dataset.metadata["class_label"]) <= set(PHASE1_CLASSES)
        # labels in the array and in the index must agree
        expected = dataset.metadata["class_label"].map(CLASS_TO_INDEX).to_numpy()
        np.testing.assert_array_equal(dataset.y, expected)

    def test_no_empty_epochs(self, dataset: EEGDataset):
        assert len(dataset) > 0
        assert dataset.X.size > 0
        # No epoch may be all-zero or constant -- that would mean a dead recording.
        per_epoch_std = dataset.X.std(axis=(1, 2))
        assert (per_epoch_std > 0).all()

    def test_no_nan_or_inf(self, dataset: EEGDataset):
        assert np.isfinite(dataset.X).all()

    def test_valid_epoch_dimensions(self, dataset: EEGDataset, synthetic_config):
        expected_samples = synthetic_config.preprocessing.expected_epoch_samples
        assert dataset.X.ndim == 3
        assert dataset.X.shape == (len(dataset), N_EEG_CHANNELS, expected_samples)
        assert dataset.X.shape[2] == 640
        assert (dataset.metadata["n_times"] == expected_samples).all()

    def test_subject_ids_are_preserved(self, dataset: EEGDataset):
        """Without this, a subject-wise split is impossible and leakage is certain."""
        assert "subject_id" in dataset.metadata
        assert "subject" in dataset.metadata
        assert set(dataset.metadata["subject"]) == set(TEST_SUBJECTS)
        assert set(dataset.metadata["subject_id"]) == {f"S{s:03d}" for s in TEST_SUBJECTS}
        np.testing.assert_array_equal(
            dataset.groups, dataset.metadata["subject"].to_numpy()
        )
        # every epoch is attributable to exactly one subject
        assert len(dataset.groups) == len(dataset)

    def test_left_and_right_classes_exist(self, dataset: EEGDataset):
        counts = dataset.class_counts()
        assert counts.get(LEFT_HAND_IMAGERY, 0) > 0
        assert counts.get(RIGHT_HAND_IMAGERY, 0) > 0
        assert counts.get(REST, 0) > 0

    def test_both_classes_present_for_every_subject(self, dataset: EEGDataset):
        """A subject-wise split needs both classes on each side of the split."""
        per_subject = dataset.metadata.groupby("subject")["class_label"].nunique()
        assert (per_subject >= 2).all()
        for subject, group in dataset.metadata.groupby("subject"):
            labels = set(group["class_label"])
            assert LEFT_HAND_IMAGERY in labels, f"subject {subject} has no left epochs"
            assert RIGHT_HAND_IMAGERY in labels, f"subject {subject} has no right epochs"

    def test_stored_epoch_length_matches_declared_window(self, dataset, synthetic_config):
        """The stored-data half of 'preprocessing does not change sample count'."""
        pre = synthetic_config.preprocessing
        expected = int(round((pre.epoch_tmax - pre.epoch_tmin) * SAMPLING_RATE_HZ))
        assert dataset.X.shape[2] == expected
        assert len(dataset.times) == expected
        np.testing.assert_allclose(dataset.times[0], pre.epoch_tmin, atol=1e-9)
        np.testing.assert_allclose(
            dataset.times[-1], pre.epoch_tmax - 1.0 / SAMPLING_RATE_HZ, atol=1e-9
        )


class TestMetadataCompleteness:
    REQUIRED = (
        "subject_id",
        "run_id",
        "event_code",
        "class_label",
        "onset_sample",
        "onset_time",
        "sampling_rate",
        "channel_count",
        "epoch_start",
        "epoch_end",
    )

    def test_all_required_fields_present(self, dataset: EEGDataset):
        for column in self.REQUIRED:
            assert column in dataset.metadata.columns, f"missing metadata field {column}"

    def test_metadata_column_contract_is_stable(self):
        for column in self.REQUIRED:
            assert column in METADATA_COLUMNS

    def test_one_metadata_row_per_epoch(self, dataset: EEGDataset):
        assert len(dataset.metadata) == dataset.X.shape[0]

    def test_epoch_uids_are_unique(self, dataset: EEGDataset):
        assert dataset.metadata["epoch_uid"].is_unique

    def test_event_codes_are_valid(self, dataset: EEGDataset):
        assert set(dataset.metadata["event_code"]) <= {"T0", "T1", "T2"}

    def test_event_code_matches_class_label(self, dataset: EEGDataset):
        """In runs 4/8/12: T0->REST, T1->LEFT, T2->RIGHT. A mismatch here is the
        single most damaging bug this pipeline could have."""
        expected = {"T0": REST, "T1": LEFT_HAND_IMAGERY, "T2": RIGHT_HAND_IMAGERY}
        for code, label in expected.items():
            rows = dataset.metadata[dataset.metadata["event_code"] == code]
            assert not rows.empty, f"no {code} events found"
            assert set(rows["class_label"]) == {label}

    def test_only_imagery_runs_present(self, dataset: EEGDataset):
        assert set(dataset.metadata["run"]) <= {4, 8, 12}
        assert set(dataset.metadata["task_type"]) == {"unilateral_imagery"}

    def test_onsets_are_non_negative_and_ordered_within_run(self, dataset: EEGDataset):
        assert (dataset.metadata["onset_sample"] >= 0).all()
        assert (dataset.metadata["onset_time"] >= 0).all()
        for _, group in dataset.metadata.groupby(["subject_id", "run_id"]):
            assert group["onset_sample"].is_monotonic_increasing

    def test_onset_sample_and_time_agree(self, dataset: EEGDataset):
        expected = (dataset.metadata["onset_time"] * SAMPLING_RATE_HZ).round().astype(int)
        assert (dataset.metadata["onset_sample"] == expected).all()

    def test_no_duplicate_events(self, dataset: EEGDataset):
        keys = dataset.metadata[["subject_id", "run_id", "onset_sample"]]
        assert not keys.duplicated().any()


class TestSummaryArtifact:
    def test_summary_written_and_consistent(self, synthetic_config, dataset):
        summary = load_summary(synthetic_config)
        assert summary["n_epochs"] == len(dataset)
        assert summary["n_channels"] == N_EEG_CHANNELS
        assert summary["sampling_rate"] == SAMPLING_RATE_HZ
        assert summary["classes"] == list(PHASE1_CLASSES)
        assert summary["n_subjects"] == len(TEST_SUBJECTS)
        assert summary["class_counts"] == dataset.class_counts()

    def test_summary_records_provenance(self, synthetic_config):
        summary = load_summary(synthetic_config)
        assert summary["config_fingerprint"] == synthetic_config.fingerprint
        assert "mne" in summary["library_versions"]
        assert summary["dataset_url"].startswith("https://physionet.org")

    def test_synthetic_source_is_flagged(self, synthetic_config):
        """A dataset built from fixtures must announce itself as such."""
        summary = load_summary(synthetic_config)
        assert summary["source"] == "synthetic"
        assert summary["synthetic"] is True

    def test_summary_is_json_serialisable(self, synthetic_config):
        path = synthetic_config.resolved_paths.metadata / "dataset_summary.json"
        json.loads(path.read_text(encoding="utf-8"))


class TestReproducibility:
    def test_config_fingerprint_is_stable(self, synthetic_config):
        assert synthetic_config.fingerprint == synthetic_config.fingerprint

    def test_fingerprint_changes_with_preprocessing(self, synthetic_config):
        from dataclasses import replace

        other = replace(
            synthetic_config,
            preprocessing=replace(synthetic_config.preprocessing, l_freq=4.0),
        )
        assert other.fingerprint != synthetic_config.fingerprint

    def test_fingerprint_ignores_paths(self, synthetic_config, tmp_path):
        from dataclasses import replace

        from src.config import PathsConfig

        moved = replace(synthetic_config, paths=PathsConfig(root=tmp_path))
        assert moved.fingerprint == synthetic_config.fingerprint

    def test_rebuild_is_deterministic(self, synthetic_config, dataset):
        """Re-running the pipeline over the same raw files reproduces the data."""
        from src.preprocessing.pipeline import build_dataset

        build_dataset(synthetic_config)
        rebuilt = load_dataset(synthetic_config, normalize=False)
        np.testing.assert_allclose(rebuilt.X, dataset.X, rtol=1e-6, atol=1e-12)
        np.testing.assert_array_equal(rebuilt.y, dataset.y)


class TestSubjectWiseSplitting:
    """Leakage protection: the data must support grouping by subject."""

    def test_groups_enable_leave_one_subject_out(self, dataset: EEGDataset):
        unique = np.unique(dataset.groups)
        assert len(unique) == len(TEST_SUBJECTS)
        for subject in unique:
            train = dataset.groups != subject
            test = dataset.groups == subject
            assert train.sum() > 0 and test.sum() > 0
            # no subject may appear on both sides
            assert not set(dataset.groups[train]) & set(dataset.groups[test])

    def test_per_subject_archives_exist(self, synthetic_config, dataset):
        processed = synthetic_config.resolved_paths.processed
        for subject_id in dataset.metadata["subject_id"].unique():
            assert (processed / f"{subject_id}.npz").exists()

    def test_loading_one_subject_returns_only_that_subject(self, synthetic_config):
        subset = load_dataset(synthetic_config, subjects=[1], normalize=False)
        assert set(subset.metadata["subject"]) == {1}
        assert len(subset) > 0


class TestNormalizedLoad:
    def test_normalisation_applied_on_load(self, normalized_dataset: EEGDataset):
        per_channel = normalized_dataset.X.mean(axis=2)
        np.testing.assert_allclose(per_channel, 0.0, atol=1e-4)
        np.testing.assert_allclose(normalized_dataset.X.std(axis=2), 1.0, atol=1e-3)

    def test_normalisation_preserves_shape_and_labels(self, dataset, normalized_dataset):
        assert normalized_dataset.X.shape == dataset.X.shape
        np.testing.assert_array_equal(normalized_dataset.y, dataset.y)

    def test_stored_data_is_unnormalised(self, dataset: EEGDataset):
        """On disk the epochs must stay in volts so PSDs remain meaningful."""
        assert np.abs(dataset.X).max() < 1e-3  # volts, not z-scores
        assert dataset.X.std() < 1e-3


class TestScientificSanity:
    """Does the extracted data show the physiology it is supposed to?"""

    def test_contralateral_lateralisation(self, dataset: EEGDataset):
        """Left-hand imagery suppresses mu over C4 (right hemisphere); right-hand
        imagery suppresses it over C3.  The lateralisation index log(C4/C3) must
        therefore be negative for left and positive for right.

        This is an end-to-end test of the *labelling*: if T1/T2 were swapped, or
        epochs were cut at the wrong samples, or channel names were misresolved,
        these signs would not come out.
        """
        left = lateralization_index(dataset, LEFT_HAND_IMAGERY)
        right = lateralization_index(dataset, RIGHT_HAND_IMAGERY)
        assert left < 0, f"left-hand imagery should suppress C4, got index {left:+.3f}"
        assert right > 0, f"right-hand imagery should suppress C3, got index {right:+.3f}"
        assert right > left

    def test_rest_is_not_strongly_lateralised(self, dataset: EEGDataset):
        rest = lateralization_index(dataset, REST)
        left = lateralization_index(dataset, LEFT_HAND_IMAGERY)
        right = lateralization_index(dataset, RIGHT_HAND_IMAGERY)
        assert abs(rest) < min(abs(left), abs(right))

    def test_spectral_power_confined_to_passband(self, dataset, synthetic_config):
        from scipy.signal import welch

        pre = synthetic_config.preprocessing
        freqs, psd = welch(dataset.X, fs=SAMPLING_RATE_HZ, nperseg=256, axis=-1)
        in_band = psd[..., (freqs >= pre.l_freq) & (freqs <= pre.h_freq)].sum()
        total = psd.sum()
        assert in_band / total > 0.9
