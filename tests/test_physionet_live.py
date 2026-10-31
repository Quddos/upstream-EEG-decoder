"""Validation against the *real* PhysioNet recordings.

These tests are marked ``physionet`` and are **deselected by default**.  Run them
explicitly::

    pytest -m physionet -v

They are the gate between Phase 1 and Phase 2.  The rest of the suite proves the
pipeline's logic is correct against synthetic fixtures; this module proves the real
EEGMMIDB files match the specification the pipeline was built against.

Discovery, not download
-----------------------
These tests validate whatever real recordings are **already present** under
``data/raw/`` -- they never download.  That keeps the suite honest in two ways:

* it cannot silently pull 3.4 GB during what looks like a unit-test run, and
* a missing recording produces a **skip with an explicit reason**, never a pass.

If no real recordings are present at all, the whole module skips with
``PHYSIONET DATA UNAVAILABLE``.  A skip is not a pass: ``pytest`` reports it
separately, and the Phase 1.5 report must quote the skip, not omit it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from src.config import DatasetConfig, PathsConfig
from src.data.loader import (
    is_synthetic_recording,
    load_raw,
    local_edf_path,
    summarize_recording,
)
from src.data.physionet_spec import (
    EDF_CHANNEL_NAMES,
    LEFT_HAND_IMAGERY,
    N_EEG_CHANNELS,
    REST,
    RIGHT_HAND_IMAGERY,
    SAMPLING_RATE_HZ,
    UNILATERAL_IMAGERY_RUNS,
    RunSpec,
    resolve_run_task,
)
from src.io.store import load_dataset
from src.preprocessing.pipeline import build_dataset
from src.viz.figures import lateralization_index

pytestmark = pytest.mark.physionet

#: Where real recordings must live.  Deliberately the physionet-scoped tree.
REAL_PATHS = PathsConfig().for_source("physionet").resolve()

SKIP_REASON = (
    "PHYSIONET DATA UNAVAILABLE - no real EDF recordings found under "
    f"{REAL_PATHS.raw}. Place real EEGMMIDB files there (or run "
    "`python -m src.cli.build_dataset --source physionet` in a networked "
    "environment). This is a SKIP, not a pass."
)


def _discover_real_runs() -> List[RunSpec]:
    """Every unilateral-imagery recording physically present under ``data/raw``."""
    found: List[RunSpec] = []
    for subject in range(1, 110):
        for run in UNILATERAL_IMAGERY_RUNS:
            spec = RunSpec(subject=subject, run=run)
            if local_edf_path(spec, REAL_PATHS.raw).exists():
                found.append(spec)
    return found


REAL_RUNS = _discover_real_runs()

if not REAL_RUNS:
    pytest.skip(SKIP_REASON, allow_module_level=True)


def _config_for(specs: List[RunSpec]) -> DatasetConfig:
    return replace(
        DatasetConfig(),
        subjects=tuple(sorted({s.subject for s in specs})),
        runs=tuple(sorted({s.run for s in specs})),
        source="physionet",
        strict=True,
    )


@pytest.fixture(scope="module")
def live_config() -> DatasetConfig:
    return _config_for(REAL_RUNS)


def test_discovery_reports_what_was_found():
    """Make the scope of this run visible in the test output."""
    listing = ", ".join(f"{s.subject_id}{s.run_id}" for s in REAL_RUNS)
    print(f"\nreal recordings under test ({len(REAL_RUNS)}): {listing}")
    assert REAL_RUNS


class TestRealRecordingFormat:
    """Confirm the documented format against the actual files."""

    @pytest.mark.parametrize("spec", REAL_RUNS, ids=lambda s: f"{s.subject_id}{s.run_id}")
    def test_is_not_a_synthetic_fixture(self, spec, live_config):
        """Guard against validating a fixture and calling it real data."""
        raw = load_raw(spec, live_config, preload=False)
        assert not is_synthetic_recording(raw)

    @pytest.mark.parametrize("spec", REAL_RUNS, ids=lambda s: f"{s.subject_id}{s.run_id}")
    def test_sampling_rate_is_160hz(self, spec, live_config):
        raw = load_raw(spec, live_config, preload=False)
        assert raw.info["sfreq"] == SAMPLING_RATE_HZ

    @pytest.mark.parametrize("spec", REAL_RUNS, ids=lambda s: f"{s.subject_id}{s.run_id}")
    def test_has_64_eeg_channels(self, spec, live_config):
        raw = load_raw(spec, live_config, preload=False)
        assert len(raw.ch_names) == N_EEG_CHANNELS

    @pytest.mark.parametrize("spec", REAL_RUNS, ids=lambda s: f"{s.subject_id}{s.run_id}")
    def test_edf_channel_order_matches_the_spec(self, spec, live_config):
        """If this fails, ``EDF_CHANNEL_NAMES`` is wrong -- which would not break
        loading (names are read from the file) but would make every synthetic
        fixture structurally unfaithful."""
        from src.data.loader import standardize_channel_names

        raw = load_raw(spec, live_config, preload=False)
        assert raw.ch_names == standardize_channel_names(EDF_CHANNEL_NAMES)

    @pytest.mark.parametrize("spec", REAL_RUNS, ids=lambda s: f"{s.subject_id}{s.run_id}")
    def test_annotations_use_only_t0_t1_t2(self, spec, live_config):
        raw = load_raw(spec, live_config, preload=False)
        assert set(raw.annotations.description) <= {"T0", "T1", "T2"}

    @pytest.mark.parametrize("spec", REAL_RUNS, ids=lambda s: f"{s.subject_id}{s.run_id}")
    def test_run_is_about_two_minutes(self, spec, live_config):
        summary = summarize_recording(spec, live_config)
        assert 100 < summary.duration_s < 150

    @pytest.mark.parametrize("spec", REAL_RUNS, ids=lambda s: f"{s.subject_id}{s.run_id}")
    def test_both_task_codes_present(self, spec, live_config):
        """A run 4 with only T1 events would mean the protocol assumption is wrong."""
        summary = summarize_recording(spec, live_config)
        assert summary.event_counts["T1"] > 0
        assert summary.event_counts["T2"] > 0

    @pytest.mark.parametrize("spec", REAL_RUNS, ids=lambda s: f"{s.subject_id}{s.run_id}")
    def test_trial_count_matches_expected_scale(self, spec, live_config):
        """~15 task trials per 2-minute run (4.2 s rest + 4.1 s task cycles)."""
        summary = summarize_recording(spec, live_config)
        n_task = summary.event_counts["T1"] + summary.event_counts["T2"]
        assert 10 <= n_task <= 20, f"unexpected trial count: {n_task}"

    @pytest.mark.parametrize("spec", REAL_RUNS, ids=lambda s: f"{s.subject_id}{s.run_id}")
    def test_cue_durations_match_the_protocol(self, spec, live_config):
        """Documented timing: ~4.2 s rest, ~4.1 s task."""
        raw = load_raw(spec, live_config, preload=False)
        durations = {round(float(d), 1) for d in raw.annotations.duration}
        assert durations <= {4.1, 4.2}, f"unexpected cue durations: {durations}"

    @pytest.mark.parametrize("spec", REAL_RUNS, ids=lambda s: f"{s.subject_id}{s.run_id}")
    def test_provenance_is_recorded(self, spec, live_config):
        summary = summarize_recording(spec, live_config)
        assert summary.source == "physionet"
        assert summary.dataset_version == "1.0.0"
        assert len(summary.sha256) == 64
        assert summary.file_bytes > 1_000_000
        assert summary.acquisition_date is not None


class TestRealRunSemantics:
    """The run type must be resolved, not assumed."""

    @pytest.mark.parametrize("spec", REAL_RUNS, ids=lambda s: f"{s.subject_id}{s.run_id}")
    def test_run_resolves_to_unilateral_imagery(self, spec):
        info = resolve_run_task(spec.run)
        assert info["is_unilateral_imagery"] is True
        assert info["task"] == "unilateral_fist_imagery"
        assert info["t0"] == REST
        assert info["t1"] == LEFT_HAND_IMAGERY
        assert info["t2"] == RIGHT_HAND_IMAGERY

    @pytest.mark.parametrize("spec", REAL_RUNS, ids=lambda s: f"{s.subject_id}{s.run_id}")
    def test_annotation_labels_resolve_through_the_run_map(self, spec, live_config):
        from src.data.loader import annotations_table

        raw = load_raw(spec, live_config, preload=False)
        table = annotations_table(spec, raw)
        assert table["class_label"].notna().all()
        for code, label in (
            ("T0", REST),
            ("T1", LEFT_HAND_IMAGERY),
            ("T2", RIGHT_HAND_IMAGERY),
        ):
            rows = table[table["event_code"] == code]
            if not rows.empty:
                assert set(rows["class_label"]) == {label}


class TestRealDatasetBuild:
    """Build from the real files and validate the resulting dataset."""

    @pytest.fixture(scope="class")
    def live_dataset(self, live_config):
        build_dataset(live_config)
        return load_dataset(live_config, normalize=False)

    def test_dataset_shape(self, live_dataset, live_config):
        assert live_dataset.X.ndim == 3
        assert live_dataset.X.shape[1] == N_EEG_CHANNELS
        assert live_dataset.X.shape[2] == live_config.preprocessing.expected_epoch_samples

    def test_no_nan_or_empty_epochs(self, live_dataset):
        assert np.isfinite(live_dataset.X).all()
        assert (live_dataset.X.std(axis=(1, 2)) > 0).all()

    def test_both_hands_present(self, live_dataset):
        labels = set(live_dataset.metadata["class_label"])
        assert LEFT_HAND_IMAGERY in labels
        assert RIGHT_HAND_IMAGERY in labels

    def test_amplitudes_are_physiological(self, live_dataset):
        """After an 8-30 Hz band-pass, EEG should sit in the tens of microvolts.
        Orders of magnitude away would mean a unit-scaling error."""
        p99 = float(np.percentile(np.abs(live_dataset.X), 99))
        assert 1e-6 < p99 < 200e-6, f"99th percentile amplitude {p99 * 1e6:.1f} uV"

    def test_metadata_marks_the_data_as_real(self, live_dataset):
        assert set(live_dataset.metadata["source"]) == {"physionet"}

    def test_event_code_matches_class_label(self, live_dataset):
        expected = {"T0": REST, "T1": LEFT_HAND_IMAGERY, "T2": RIGHT_HAND_IMAGERY}
        for code, label in expected.items():
            rows = live_dataset.metadata[live_dataset.metadata["event_code"] == code]
            if not rows.empty:
                assert set(rows["class_label"]) == {label}

    def test_subject_ids_preserved(self, live_dataset, live_config):
        assert set(live_dataset.metadata["subject"]) == set(live_config.subjects)


class TestRealPhysiology:
    """The claim that matters -- reported, never enforced.

    Contralateral ERD is a group-level effect.  A well-known fraction of EEGMMIDB
    subjects are BCI-illiterate and show weak or absent modulation, and a single
    run contributes only ~7 trials per hand.  These tests therefore **report** the
    measured lateralisation and only assert that it is finite and computable.

    Nothing here alters the data to make lateralisation appear, and no expected
    C3/C4 suppression value is hard-coded.
    """

    @pytest.fixture(scope="class")
    def live_dataset(self, live_config):
        build_dataset(live_config)
        return load_dataset(live_config, normalize=False)

    def test_lateralisation_is_computable(self, live_dataset, record_property):
        left = lateralization_index(live_dataset, LEFT_HAND_IMAGERY)
        right = lateralization_index(live_dataset, RIGHT_HAND_IMAGERY)
        record_property("lateralization_left", left)
        record_property("lateralization_right", right)
        record_property("separation_right_minus_left", right - left)
        print(
            f"\nmeasured lateralisation log(C4mu/C3mu):"
            f"\n  LEFT  hand : {left:+.3f}"
            f"\n  RIGHT hand : {right:+.3f}"
            f"\n  separation : {right - left:+.3f}  (positive = expected direction)"
        )
        assert np.isfinite(left) and np.isfinite(right)

    @pytest.mark.xfail(
        reason=(
            "Contralateral ERD is a group-level effect measured here on very few "
            "trials. A failure is a finding about this subject/run, not a pipeline "
            "bug -- investigate per subject rather than 'fixing' the data."
        ),
        strict=False,
    )
    def test_lateralisation_points_the_expected_way(self, live_dataset):
        left = lateralization_index(live_dataset, LEFT_HAND_IMAGERY)
        right = lateralization_index(live_dataset, RIGHT_HAND_IMAGERY)
        assert left < right, (
            f"expected left-hand imagery to shift mu power toward C3 relative to "
            f"right-hand imagery; got left={left:+.3f}, right={right:+.3f}"
        )
