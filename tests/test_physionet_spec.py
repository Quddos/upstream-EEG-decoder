"""Tests for the dataset specification.

These are the most important tests in the suite.  Every downstream label depends
on the run-dependent T1/T2 mapping being right, and a silent error here would
produce a dataset that trains a decoder to predict the wrong hand -- with no
symptom anywhere else in the pipeline.
"""

from __future__ import annotations

import pytest

from src.data import physionet_spec as spec


class TestRecordingProperties:
    def test_sampling_rate_is_160hz(self):
        assert spec.SAMPLING_RATE_HZ == 160.0

    def test_channel_count_is_64(self):
        assert spec.N_EEG_CHANNELS == 64
        assert len(spec.EDF_CHANNEL_NAMES) == 64

    def test_channel_names_are_unique(self):
        assert len(set(spec.EDF_CHANNEL_NAMES)) == 64

    def test_subject_and_run_counts(self):
        assert spec.N_SUBJECTS_TOTAL == 109
        assert spec.N_RUNS_PER_SUBJECT == 14


class TestRunTaxonomy:
    def test_every_run_has_a_task_type(self):
        for run in range(1, 15):
            assert spec.task_type_for_run(run) is not None

    def test_invalid_runs_rejected(self):
        for run in (0, 15, -1, 100):
            with pytest.raises(ValueError):
                spec.task_type_for_run(run)

    def test_unilateral_imagery_runs_are_4_8_12(self):
        """The runs this project decodes. Getting this wrong means decoding
        motor *execution*, or a fists-vs-feet task, instead of left/right imagery."""
        assert spec.UNILATERAL_IMAGERY_RUNS == (4, 8, 12)
        for run in (4, 8, 12):
            assert spec.is_unilateral_imagery_run(run)

    def test_execution_runs_are_not_imagery(self):
        for run in (3, 7, 11):
            assert not spec.is_unilateral_imagery_run(run)
            assert spec.task_type_for_run(run) is spec.TaskType.UNILATERAL_EXECUTION

    def test_bilateral_runs_are_not_unilateral_imagery(self):
        for run in (5, 6, 9, 10, 13, 14):
            assert not spec.is_unilateral_imagery_run(run)

    def test_baseline_runs_have_no_task(self):
        for run in (1, 2):
            assert spec.event_map_for_run(run) == {"T0": spec.REST}

    def test_run_groups_partition_all_14_runs(self):
        covered = set(
            spec.BASELINE_RUNS
            + spec.UNILATERAL_EXECUTION_RUNS
            + spec.UNILATERAL_IMAGERY_RUNS
            + spec.BILATERAL_EXECUTION_RUNS
            + spec.BILATERAL_IMAGERY_RUNS
        )
        assert covered == set(range(1, 15))


class TestEventMapping:
    """T1/T2 semantics are run-dependent -- this is the trap the brief warns about."""

    @pytest.mark.parametrize("run", [4, 8, 12])
    def test_imagery_runs_map_t1_left_t2_right(self, run):
        mapping = spec.event_map_for_run(run)
        assert mapping["T0"] == spec.REST
        assert mapping["T1"] == spec.LEFT_HAND_IMAGERY
        assert mapping["T2"] == spec.RIGHT_HAND_IMAGERY

    @pytest.mark.parametrize("run", [3, 7, 11])
    def test_execution_runs_map_to_execution_labels_not_imagery(self, run):
        mapping = spec.event_map_for_run(run)
        assert mapping["T1"] == spec.LEFT_HAND_EXECUTION
        assert mapping["T2"] == spec.RIGHT_HAND_EXECUTION
        assert spec.LEFT_HAND_IMAGERY not in mapping.values()

    @pytest.mark.parametrize("run", [6, 10, 14])
    def test_bilateral_imagery_t1_is_both_fists_not_left_hand(self, run):
        """The specific fallacy the brief calls out: T1 is NOT globally 'left'."""
        mapping = spec.event_map_for_run(run)
        assert mapping["T1"] == spec.BOTH_FISTS_IMAGERY
        assert mapping["T2"] == spec.BOTH_FEET_IMAGERY
        assert mapping["T1"] != spec.LEFT_HAND_IMAGERY

    @pytest.mark.parametrize("run", [5, 9, 13])
    def test_bilateral_execution_mapping(self, run):
        mapping = spec.event_map_for_run(run)
        assert mapping["T1"] == spec.BOTH_FISTS_EXECUTION
        assert mapping["T2"] == spec.BOTH_FEET_EXECUTION

    def test_t1_means_different_things_in_different_runs(self):
        """Direct assertion that the mapping is not global."""
        assert spec.event_map_for_run(4)["T1"] != spec.event_map_for_run(6)["T1"]
        assert spec.event_map_for_run(4)["T1"] != spec.event_map_for_run(3)["T1"]

    def test_t0_is_rest_in_every_run(self):
        for run in range(1, 15):
            assert spec.event_map_for_run(run)["T0"] == spec.REST

    def test_event_map_is_a_copy(self):
        """Callers must not be able to mutate the shared specification."""
        mapping = spec.event_map_for_run(4)
        mapping["T1"] = "CORRUPTED"
        assert spec.event_map_for_run(4)["T1"] == spec.LEFT_HAND_IMAGERY


class TestClassCodes:
    def test_phase1_classes_are_rest_left_right(self):
        assert spec.PHASE1_CLASSES == (
            spec.REST,
            spec.LEFT_HAND_IMAGERY,
            spec.RIGHT_HAND_IMAGERY,
        )

    def test_class_index_mapping_is_bijective(self):
        assert spec.CLASS_TO_INDEX == {"REST": 0, "LEFT_HAND_IMAGERY": 1, "RIGHT_HAND_IMAGERY": 2}
        assert set(spec.INDEX_TO_CLASS) == set(spec.CLASS_TO_INDEX.values())
        for name, index in spec.CLASS_TO_INDEX.items():
            assert spec.INDEX_TO_CLASS[index] == name

    def test_contralateral_channels(self):
        """Left hand -> right hemisphere (C4); right hand -> left hemisphere (C3)."""
        assert spec.CONTRALATERAL_HAND_CHANNELS[spec.LEFT_HAND_IMAGERY] == "C4"
        assert spec.CONTRALATERAL_HAND_CHANNELS[spec.RIGHT_HAND_IMAGERY] == "C3"


class TestIdentifiers:
    def test_subject_id_formatting(self):
        assert spec.subject_id(1) == "S001"
        assert spec.subject_id(109) == "S109"

    def test_subject_id_range_enforced(self):
        for bad in (0, 110, -3):
            with pytest.raises(ValueError):
                spec.subject_id(bad)

    def test_run_id_formatting(self):
        assert spec.run_id(4) == "R04"
        assert spec.run_id(12) == "R12"

    def test_edf_filename(self):
        assert spec.edf_filename(1, 4) == "S001R04.edf"
        assert spec.edf_filename(109, 14) == "S109R14.edf"

    def test_run_spec_resolves_everything(self):
        rs = spec.RunSpec(subject=5, run=8)
        assert rs.subject_id == "S005"
        assert rs.run_id == "R08"
        assert rs.filename == "S005R08.edf"
        assert rs.task_type is spec.TaskType.UNILATERAL_IMAGERY
        assert rs.event_map["T2"] == spec.RIGHT_HAND_IMAGERY

    def test_iter_run_specs_is_a_full_product(self):
        specs = spec.iter_run_specs([1, 2], [4, 8])
        assert len(specs) == 4
        assert {(s.subject, s.run) for s in specs} == {(1, 4), (1, 8), (2, 4), (2, 8)}


class TestResolveRunTask:
    """`resolve_run_task` is the explicit run-type check required by Phase 1.5.

    The run type must be *resolved*, never inferred from the word "imagery":
    runs 6, 10 and 14 are imagery runs too, but bilateral ones whose T1/T2 mean
    something entirely different.
    """

    def test_unilateral_imagery_run_by_number(self):
        info = spec.resolve_run_task(4)
        assert info["run"] == "R04"
        assert info["run_number"] == 4
        assert info["task"] == "unilateral_fist_imagery"
        assert info["is_unilateral_imagery"] is True
        assert info["t0"] == spec.REST
        assert info["t1"] == spec.LEFT_HAND_IMAGERY
        assert info["t2"] == spec.RIGHT_HAND_IMAGERY

    def test_accepts_run_identifier_string(self):
        assert spec.resolve_run_task("R04") == spec.resolve_run_task(4)
        assert spec.resolve_run_task("r04") == spec.resolve_run_task(4)

    @pytest.mark.parametrize("run", [4, 8, 12])
    def test_all_three_imagery_runs_resolve_identically(self, run):
        info = spec.resolve_run_task(run)
        assert info["is_unilateral_imagery"] is True
        assert info["t1"] == spec.LEFT_HAND_IMAGERY
        assert info["t2"] == spec.RIGHT_HAND_IMAGERY

    def test_bilateral_imagery_is_not_unilateral(self):
        """Run 6 is imagery but NOT the left/right paradigm."""
        info = spec.resolve_run_task(6)
        assert info["task"] == "bilateral_fist_foot_imagery"
        assert info["is_unilateral_imagery"] is False
        assert info["t1"] == spec.BOTH_FISTS_IMAGERY
        assert info["t2"] == spec.BOTH_FEET_IMAGERY

    def test_execution_run_is_not_imagery(self):
        info = spec.resolve_run_task(3)
        assert info["task"] == "unilateral_fist_execution"
        assert info["is_unilateral_imagery"] is False
        assert info["t1"] == spec.LEFT_HAND_EXECUTION

    def test_baseline_runs_have_no_task_codes(self):
        info = spec.resolve_run_task(1)
        assert info["t0"] == spec.REST
        assert info["t1"] is None and info["t2"] is None

    def test_exactly_three_runs_are_unilateral_imagery(self):
        flagged = [r for r in range(1, 15) if spec.resolve_run_task(r)["is_unilateral_imagery"]]
        assert flagged == [4, 8, 12]

    def test_invalid_run_rejected(self):
        for bad in (0, 15, "R99", "banana"):
            with pytest.raises(ValueError):
                spec.resolve_run_task(bad)


class TestDatasetIdentity:
    def test_version_and_doi_recorded(self):
        assert spec.DATASET_VERSION == "1.0.0"
        assert spec.DATASET_DOI == "10.13026/C28G6P"
        assert spec.PHYSIONET_RECORD_URL.startswith("https://physionet.org/")
        assert spec.PHYSIONET_CHECKSUMS_URL.endswith("SHA256SUMS.txt")
