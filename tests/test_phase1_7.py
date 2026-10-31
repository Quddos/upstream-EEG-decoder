"""Tests for Phase 1.7: integrity verification, the sensorimotor ROI, and the
dataset-level structural checks.

Entirely offline.  The one thing that must never regress here is the honesty of
:class:`VerificationReport`: an unverifiable set of files must never report as a
verified one, because that distinction is what separates "this is the file we
processed" from "this is the file PhysioNet published".
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.cli.dataset_report import EXCLUDED_RUNS, IDENTITY_COLUMNS, check_dataset
from src.config import DatasetConfig
from src.data.download import EVENT_SUFFIX, event_filename
from src.data.integrity import (
    ManifestError,
    count_sidecar_annotations,
    files_for_specs,
    parse_manifest,
    sha256_of,
    verify_files,
)
from src.data.physionet_spec import (
    SENSORIMOTOR_ROI,
    SENSORIMOTOR_ROI_NAME,
    RunSpec,
    UNILATERAL_IMAGERY_RUNS,
    event_map_for_run,
    roi_indices,
)
from src.io.store import EEGDataset

#: The ROI exactly as the Phase 1.7 brief specifies it, in the brief's order.
BRIEF_ROI = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)

MONTAGE_64 = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
)


def write(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class TestChecksums:
    def test_sha256_matches_hashlib(self, tmp_path):
        payload = b"eegmmidb" * 500
        path = write(tmp_path / "a.edf", payload)
        assert sha256_of(path) == hashlib.sha256(payload).hexdigest()

    def test_hashing_is_chunked_but_exact_for_large_files(self, tmp_path):
        payload = bytes(range(256)) * 8192  # > 1 MiB, forces multiple chunks
        path = write(tmp_path / "big.edf", payload)
        assert sha256_of(path) == hashlib.sha256(payload).hexdigest()


class TestManifestParsing:
    def test_parses_digest_and_path_pairs(self, tmp_path):
        digest = "a" * 64
        manifest = write(
            tmp_path / "SHA256SUMS.txt",
            f"{digest} S001/S001R04.edf\n{'b' * 64} S001/S001R04.edf.event\n".encode(),
        )
        entries = parse_manifest(manifest)
        assert entries["S001/S001R04.edf"] == digest
        assert len(entries) == 2

    def test_skips_malformed_lines_without_failing(self, tmp_path):
        """The manifest covers 109 subjects; one odd line must not block us."""
        manifest = write(
            tmp_path / "m.txt",
            b"garbage\n\n# comment\n" + (b"c" * 64) + b" S002/S002R08.edf\n",
        )
        assert parse_manifest(manifest) == {"S002/S002R08.edf": "c" * 64}

    def test_a_manifest_with_no_usable_entries_raises(self, tmp_path):
        manifest = write(tmp_path / "m.txt", b"nothing here\nnor here\n")
        with pytest.raises(ManifestError):
            parse_manifest(manifest)


class TestVerification:
    def _setup(self, tmp_path, payload=b"real edf bytes"):
        edf = write(tmp_path / "raw" / "S001" / "S001R04.edf", payload)
        manifest = write(
            tmp_path / "SHA256SUMS.txt",
            f"{hashlib.sha256(payload).hexdigest()} S001/S001R04.edf\n".encode(),
        )
        return edf, manifest

    def test_matching_file_verifies(self, tmp_path):
        edf, manifest = self._setup(tmp_path)
        report = verify_files([("S001/S001R04.edf", edf)], manifest)
        assert report.all_verified
        assert report.results[0].status == "OK"
        assert "VERIFIED" in report.claim()

    def test_altered_file_is_caught(self, tmp_path):
        edf, manifest = self._setup(tmp_path)
        edf.write_bytes(b"tampered")
        report = verify_files([("S001/S001R04.edf", edf)], manifest)
        assert not report.all_verified
        assert report.results[0].status == "MISMATCH"
        assert "FAILED" in report.claim()

    def test_missing_file_is_caught(self, tmp_path):
        edf, manifest = self._setup(tmp_path)
        edf.unlink()
        report = verify_files([("S001/S001R04.edf", edf)], manifest)
        assert report.results[0].status == "MISSING"
        assert not report.all_verified

    def test_file_absent_from_manifest_is_not_a_pass(self, tmp_path):
        edf, manifest = self._setup(tmp_path)
        report = verify_files([("S999/S999R04.edf", edf)], manifest)
        assert report.results[0].status == "NOT_IN_MANIFEST"
        assert not report.all_verified

    def test_without_a_manifest_nothing_is_verified(self, tmp_path):
        """The critical honesty property: unverifiable must not read as verified."""
        edf, _ = self._setup(tmp_path)
        report = verify_files([("S001/S001R04.edf", edf)], None)
        assert not report.manifest_available
        assert not report.all_verified
        assert "NOT VERIFIED" in report.claim()

    def test_a_missing_manifest_path_behaves_as_no_manifest(self, tmp_path):
        edf, _ = self._setup(tmp_path)
        report = verify_files([("S001/S001R04.edf", edf)], tmp_path / "absent.txt")
        assert not report.manifest_available
        assert not report.all_verified

    def test_an_empty_file_list_is_never_verified(self, tmp_path):
        _, manifest = self._setup(tmp_path)
        assert not verify_files([], manifest).all_verified

    def test_files_for_specs_covers_edf_and_sidecar(self, tmp_path):
        specs = [RunSpec(subject=1, run=4)]
        pairs = files_for_specs(specs, tmp_path, include_events=True)
        names = [rel for rel, _ in pairs]
        assert names == ["S001/S001R04.edf", "S001/S001R04.edf.event"]

    def test_files_for_specs_can_skip_sidecars(self, tmp_path):
        pairs = files_for_specs([RunSpec(subject=1, run=4)], tmp_path, include_events=False)
        assert [rel for rel, _ in pairs] == ["S001/S001R04.edf"]


class TestSidecar:
    def test_event_filename_appends_the_suffix(self):
        assert event_filename(RunSpec(subject=3, run=12)) == f"S003R12.edf{EVENT_SUFFIX}"

    def test_counts_annotation_codes(self, tmp_path):
        # Shaped like the real WFDB sidecar: binary noise around aux strings.
        payload = (
            b"\x00\x01T0 duration: 4.2\xff\xfeT1 duration: 4.1"
            b"\x00T0 duration: 4.2\x00T2 duration: 4.1\x00T0 duration: 4.2"
        )
        counts = count_sidecar_annotations(write(tmp_path / "s.event", payload))
        assert counts == {"T0": 3, "T1": 1, "T2": 1}


class TestSensorimotorROI:
    def test_matches_the_brief_exactly_including_order(self):
        assert SENSORIMOTOR_ROI == BRIEF_ROI
        assert len(SENSORIMOTOR_ROI) == 21

    def test_has_a_stable_name_for_provenance(self):
        assert SENSORIMOTOR_ROI_NAME == "sensorimotor_21"

    def test_resolves_against_the_full_montage(self):
        picks = roi_indices(MONTAGE_64, SENSORIMOTOR_ROI)
        assert len(picks) == 21
        assert [MONTAGE_64[i] for i in picks] == list(SENSORIMOTOR_ROI)

    def test_indices_follow_roi_order_not_montage_order(self):
        picks = roi_indices(("C4", "C3", "Cz"), ("C3", "C4"))
        assert picks == (1, 0)

    def test_a_missing_electrode_raises_rather_than_shortening(self):
        """A silently shortened ROI would change the model input invisibly."""
        with pytest.raises(ValueError, match="absent"):
            roi_indices(("C3", "C4"), SENSORIMOTOR_ROI)

    def test_is_not_applied_to_the_stored_dataset_by_default(self):
        """64 channels must stay on disk so the ablation remains possible."""
        assert DatasetConfig().preprocessing.channel_subset is None


class TestDatasetROIView:
    def _dataset(self, n_epochs=4):
        X = np.random.default_rng(0).normal(size=(n_epochs, len(MONTAGE_64), 32))
        return EEGDataset(
            X=X,
            y=np.zeros(n_epochs, dtype=np.int64),
            groups=np.ones(n_epochs, dtype=np.int64),
            metadata=pd.DataFrame({"epoch_uid": [f"E{i}" for i in range(n_epochs)]}),
            channels=MONTAGE_64,
            times=np.arange(32) / 160.0,
        )

    def test_roi_view_has_21_channels(self):
        roi = self._dataset().sensorimotor_roi()
        assert roi.X.shape[1] == 21
        assert roi.channels == SENSORIMOTOR_ROI

    def test_roi_view_keeps_data_and_names_aligned(self):
        dataset = self._dataset()
        roi = dataset.sensorimotor_roi()
        for name in ("C3", "C4", "CPz"):
            expected = dataset.X[:, list(MONTAGE_64).index(name), :]
            actual = roi.X[:, list(roi.channels).index(name), :]
            assert np.array_equal(expected, actual)

    def test_roi_view_does_not_mutate_the_full_dataset(self):
        dataset = self._dataset()
        before = dataset.X.copy()
        dataset.sensorimotor_roi()
        assert dataset.X.shape[1] == 64
        assert np.array_equal(dataset.X, before)

    def test_epoch_metadata_survives_the_view(self):
        dataset = self._dataset()
        roi = dataset.sensorimotor_roi()
        assert len(roi.metadata) == len(dataset.metadata)
        assert np.array_equal(roi.y, dataset.y)
        assert np.array_equal(roi.groups, dataset.groups)


def make_index(subjects=(1, 2), runs=UNILATERAL_IMAGERY_RUNS):
    """A minimal but structurally valid epoch index."""
    rows = []
    uid = 0
    for subject in subjects:
        for run in runs:
            for code, label in event_map_for_run(run).items():
                rows.append(
                    {
                        "epoch_uid": f"E{uid:07d}",
                        "subject_id": f"S{subject:03d}",
                        "run_id": f"R{run:02d}",
                        "subject": subject,
                        "run": run,
                        "event_code": code,
                        "class_label": label,
                        "onset_time": float(uid),
                        "source": "physionet",
                        "artifact_flag": False,
                    }
                )
                uid += 1
    return pd.DataFrame(rows)


class TestStructuralChecks:
    def _config(self, subjects=(1, 2)):
        return DatasetConfig(subjects=tuple(subjects), runs=UNILATERAL_IMAGERY_RUNS)

    def test_a_well_formed_dataset_has_no_problems(self):
        assert check_dataset(make_index(), self._config(), MONTAGE_64) == []

    def test_a_missing_subject_run_is_reported(self):
        index = make_index()
        index = index[~((index["subject"] == 1) & (index["run"] == 8))]
        problems = check_dataset(index, self._config(), MONTAGE_64)
        assert any("S001 R08" in p for p in problems)

    def test_a_forbidden_run_is_reported(self):
        index = make_index()
        index.loc[index["run"] == 4, "run"] = 6  # bilateral imagery, excluded
        problems = check_dataset(index, self._config(), MONTAGE_64)
        assert any("excluded by the brief" in p or "unexpected runs" in p for p in problems)

    def test_every_excluded_run_is_actually_excluded_from_the_phase(self):
        assert not set(EXCLUDED_RUNS) & set(UNILATERAL_IMAGERY_RUNS)
        assert set(EXCLUDED_RUNS) == {3, 5, 6, 7, 9, 10, 11, 13, 14}

    def test_a_mislabelled_event_code_is_reported(self):
        """The check must catch a label that contradicts event_map_for_run."""
        index = make_index()
        index.loc[index["event_code"] == "T1", "class_label"] = "RIGHT_HAND_IMAGERY"
        problems = check_dataset(index, self._config(), MONTAGE_64)
        assert any("T1" in p and "event_map_for_run" in p for p in problems)

    def test_a_missing_identity_column_is_reported(self):
        for column in IDENTITY_COLUMNS.values():
            index = make_index().drop(columns=[column])
            problems = check_dataset(index, self._config(), MONTAGE_64)
            assert any(column in p for p in problems), f"{column} not reported"

    def test_a_null_identity_value_is_reported(self):
        index = make_index()
        index.loc[0, "subject_id"] = None
        problems = check_dataset(index, self._config(), MONTAGE_64)
        assert any("subject_id" in p for p in problems)

    def test_duplicate_epoch_ids_are_reported(self):
        index = make_index()
        index.loc[1, "epoch_uid"] = index.loc[0, "epoch_uid"]
        problems = check_dataset(index, self._config(), MONTAGE_64)
        assert any("not unique" in p for p in problems)

    def test_a_missing_class_in_one_recording_is_reported(self):
        index = make_index()
        drop = (index["subject"] == 1) & (index["run"] == 4) & (
            index["class_label"] == "REST"
        )
        problems = check_dataset(index[~drop], self._config(), MONTAGE_64)
        assert any("REST" in p and "missing" in p for p in problems)

    def test_a_short_montage_is_reported(self):
        problems = check_dataset(make_index(), self._config(), MONTAGE_64[:32])
        assert any("64 channels" in p for p in problems)


class TestEventMappingIsResolvedNotAssumed:
    def test_the_three_imagery_runs_share_one_mapping(self):
        maps = {run: event_map_for_run(run) for run in UNILATERAL_IMAGERY_RUNS}
        assert len({tuple(sorted(m.items())) for m in maps.values()}) == 1

    def test_that_mapping_is_the_one_the_brief_specifies(self):
        for run in UNILATERAL_IMAGERY_RUNS:
            mapping = event_map_for_run(run)
            assert mapping["T0"] == "REST"
            assert mapping["T1"] == "LEFT_HAND_IMAGERY"
            assert mapping["T2"] == "RIGHT_HAND_IMAGERY"

    def test_bilateral_runs_still_map_differently(self):
        """The guard that makes a global T1/T2 rule wrong must remain in force."""
        for run in (6, 10, 14):
            mapping = event_map_for_run(run)
            assert mapping["T1"] != "LEFT_HAND_IMAGERY"
            assert mapping["T2"] != "RIGHT_HAND_IMAGERY"
