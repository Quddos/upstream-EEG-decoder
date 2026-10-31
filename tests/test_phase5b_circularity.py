"""Phase 5B -- circularity tests for the run-level stability analysis.

Phase 5B adds a grouping level (run) to the Phase 5 diagnostics, which creates two
new ways to cheat that the Phase 5 audit did not have to cover:

* a run-level diagnostic could borrow from the subject's **other runs**, which
  would destroy the repeated-measures logic the stability claim depends on;
* splitting by run is one step away from splitting by epoch, and an epoch-level
  split would make every run look artificially separable.

Everything Phase 5 guaranteed still has to hold: the frozen Phase 4B outcome must
not reach the diagnostic, and no subject may borrow from another. Each check has a
failing control.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.physionet_spec import LEFT_HAND_IMAGERY, REST, RIGHT_HAND_IMAGERY
from src.features import run_diagnostics as rd

N_SUBJECTS = 5
N_CHANNELS = 4
N_TIMES = 320
SFREQ = 160.0
CHANNELS = ("C3", "C4", "Cz", "C1")
RUNS = (4, 8, 12)
PER_RUN = 16   # 8 per class -- close to the real 7v8, and far less noisy than 4v4


@pytest.fixture(scope="module")
def cohort():
    """Subjects whose lateralisation differs *between runs*, by construction.

    Subject 1 is lateralised in every run, subject 2 in only one, subjects 3-5 in
    none. That is exactly the trait/state distinction Phase 5B claims to measure,
    so the fixture can tell whether the analysis can see it.
    """
    rng = np.random.default_rng(7)
    t = np.arange(N_TIMES) / SFREQ
    X, rows = [], []
    for subject in range(1, N_SUBJECTS + 1):
        for run in RUNS:
            lateralised = subject == 1 or (subject == 2 and run == 8)
            for i in range(PER_RUN):
                cls = LEFT_HAND_IMAGERY if i % 2 == 0 else RIGHT_HAND_IMAGERY
                epoch = rng.normal(0, 1e-6, size=(N_CHANNELS, N_TIMES))
                gain = {c: 1.0 for c in CHANNELS}
                if lateralised:
                    gain["C4" if cls == LEFT_HAND_IMAGERY else "C3"] = 0.35
                for c, name in enumerate(CHANNELS):
                    epoch[c] += gain[name] * 3e-6 * np.sin(2 * np.pi * 10 * t + rng.normal())
                X.append(epoch)
                rows.append({
                    "epoch_uid": f"E{subject:03d}{run:02d}{i:03d}",
                    "subject_id": f"S{subject:03d}", "subject": subject,
                    "run": run, "run_id": f"R{run:02d}", "class_label": cls,
                })
    return np.asarray(X), pd.DataFrame(rows)


class TestRunDiagnosticsAreOutcomeBlind:
    """The frozen Phase 4B outcome must never reach the diagnostic."""

    def test_module_imports_nothing_that_carries_an_outcome(self):
        tree = ast.parse(inspect.getsource(rd))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "sklearn" not in imported
        assert "json" not in imported

    def test_module_reads_no_files(self):
        tree = ast.parse(inspect.getsource(rd))
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = node.func
                if isinstance(target, ast.Name):
                    called.add(target.id)
                elif isinstance(target, ast.Attribute):
                    called.add(target.attr)
        for forbidden in ("open", "read_csv", "read_json", "loads"):
            assert forbidden not in called

    def test_signature_cannot_accept_an_outcome(self):
        parameters = set(inspect.signature(rd.run_level_diagnostics).parameters)
        assert parameters == {"X", "metadata", "channels", "sfreq"}

    def test_injecting_outcome_metadata_changes_nothing(self, cohort):
        """Control: an accuracy column in the metadata must be ignored."""
        X, meta = cohort
        clean = rd.run_level_diagnostics(X, meta, CHANNELS, SFREQ)

        contaminated = meta.copy()
        rng = np.random.default_rng(0)
        mapping = {s: rng.random() for s in contaminated["subject_id"].unique()}
        contaminated["losocv_balanced_accuracy"] = contaminated["subject_id"].map(mapping)
        contaminated["group"] = np.where(
            contaminated["losocv_balanced_accuracy"] < 0.5, "below_chance", "at_or_above_chance"
        )
        after = rd.run_level_diagnostics(X, contaminated, CHANNELS, SFREQ)

        numeric = clean.select_dtypes(include=[np.number]).columns
        assert np.allclose(clean[numeric].to_numpy(), after[numeric].to_numpy(),
                           equal_nan=True)

    def test_outcome_grouping_is_the_frozen_phase4b_one(self):
        path = Path("data/metadata/phase5/phase5_outcome_definition.json")
        if not path.exists():
            pytest.skip("outcome definition not present in this checkout")
        definition = json.loads(path.read_text())
        assert definition["n_below"] == 8
        assert set(definition["below_subjects"]) == {
            "S003", "S028", "S038", "S024", "S021", "S036", "S018", "S009"
        }


class TestRunIndependence:
    """A run's diagnostic must use only that subject's, that run's epochs."""

    def test_a_run_is_unchanged_when_other_runs_are_removed(self, cohort):
        """The repeated-measures guarantee: runs must not borrow from each other."""
        X, meta = cohort
        full = rd.run_level_diagnostics(X, meta, CHANNELS, SFREQ)
        full = full.set_index(["subject_id", "run"])

        keep = (meta["run"] == 8).to_numpy()
        partial = rd.run_level_diagnostics(
            X[keep], meta[keep].reset_index(drop=True), CHANNELS, SFREQ
        ).set_index(["subject_id", "run"])

        numeric = partial.select_dtypes(include=[np.number]).columns
        for key in partial.index:
            assert np.allclose(
                full.loc[key, numeric].to_numpy(float),
                partial.loc[key, numeric].to_numpy(float), equal_nan=True,
            ), f"{key} changed when the subject's other runs were removed"

    def test_a_run_is_unchanged_when_other_subjects_are_removed(self, cohort):
        X, meta = cohort
        full = rd.run_level_diagnostics(X, meta, CHANNELS, SFREQ).set_index(
            ["subject_id", "run"])
        keep = meta["subject_id"].isin(["S001", "S003"]).to_numpy()
        partial = rd.run_level_diagnostics(
            X[keep], meta[keep].reset_index(drop=True), CHANNELS, SFREQ
        ).set_index(["subject_id", "run"])
        numeric = partial.select_dtypes(include=[np.number]).columns
        for key in partial.index:
            assert np.allclose(full.loc[key, numeric].to_numpy(float),
                               partial.loc[key, numeric].to_numpy(float), equal_nan=True)

    def test_a_run_pooled_variant_would_be_detected(self, cohort):
        """Control: a diagnostic that pools a subject's runs must fail the check."""
        X, meta = cohort

        def leaky(X_, meta_, channels, sfreq):
            frame = rd.run_level_diagnostics(X_, meta_, channels, sfreq)
            # Replace each run's d with the subject's mean across runs.
            frame["lat_d"] = frame.groupby("subject_id")["lat_d"].transform("mean")
            return frame

        full = leaky(X, meta, CHANNELS, SFREQ).set_index(["subject_id", "run"])
        keep = (meta["run"] == 8).to_numpy()
        partial = leaky(X[keep], meta[keep].reset_index(drop=True),
                        CHANNELS, SFREQ).set_index(["subject_id", "run"])
        assert not np.allclose(full.loc[("S001", 8), "lat_d"],
                               partial.loc[("S001", 8), "lat_d"])

    def test_run_order_does_not_matter(self, cohort):
        X, meta = cohort
        forward = rd.run_level_diagnostics(X, meta, CHANNELS, SFREQ).set_index(
            ["subject_id", "run"])
        order = np.arange(len(meta))[::-1]
        backward = rd.run_level_diagnostics(
            X[order], meta.iloc[order].reset_index(drop=True), CHANNELS, SFREQ
        ).set_index(["subject_id", "run"])
        numeric = forward.select_dtypes(include=[np.number]).columns
        assert np.allclose(forward[numeric].to_numpy(),
                           backward.loc[forward.index, numeric].to_numpy(), equal_nan=True)


class TestNoEpochLevelSplitting:
    """Run is the smallest unit; an epoch-level split must remain impossible."""

    def test_every_row_is_one_subject_and_one_run(self, cohort):
        X, meta = cohort
        frame = rd.run_level_diagnostics(X, meta, CHANNELS, SFREQ)
        assert len(frame) == N_SUBJECTS * len(RUNS)
        assert not frame.duplicated(subset=["subject_id", "run"]).any()

    def test_epoch_counts_match_the_source_runs(self, cohort):
        X, meta = cohort
        frame = rd.run_level_diagnostics(X, meta, CHANNELS, SFREQ)
        for _, row in frame.iterrows():
            mask = ((meta["subject_id"] == row["subject_id"]) & (meta["run"] == row["run"]))
            expected_left = int((mask & (meta["class_label"] == LEFT_HAND_IMAGERY)).sum())
            expected_right = int((mask & (meta["class_label"] == RIGHT_HAND_IMAGERY)).sum())
            assert row["n_left"] == expected_left
            assert row["n_right"] == expected_right

    def test_shuffling_epochs_across_runs_changes_the_result(self, cohort):
        """Control: if run membership were ignored, this would produce no change."""
        X, meta = cohort
        honest = rd.run_level_diagnostics(X, meta, CHANNELS, SFREQ)
        scrambled = meta.copy()
        rng = np.random.default_rng(3)
        for subject in scrambled["subject_id"].unique():
            mask = scrambled["subject_id"] == subject
            values = scrambled.loc[mask, "run"].to_numpy().copy()
            rng.shuffle(values)
            scrambled.loc[mask, "run"] = values
        broken = rd.run_level_diagnostics(X, scrambled, CHANNELS, SFREQ)
        assert not np.allclose(honest["lat_d"].to_numpy(), broken["lat_d"].to_numpy(),
                               equal_nan=True)


class TestTheAnalysisCanSeeWhatItClaims:
    """A stability claim is worthless if the method cannot detect instability."""

    def test_a_consistently_lateralised_subject_is_usable_in_every_run(self, cohort):
        X, meta = cohort
        frame = rd.run_level_diagnostics(X, meta, CHANNELS, SFREQ)
        subject = frame[frame["subject_id"] == "S001"]
        assert subject["usable_lateralisation"].all()

    def test_a_single_run_responder_is_detected_as_mixed(self, cohort):
        """The state-dependent case must come out mixed, not uniformly usable."""
        X, meta = cohort
        frame = rd.run_level_diagnostics(X, meta, CHANNELS, SFREQ)
        summary = rd.summarise_subject(frame).set_index("subject_id")
        assert bool(summary.loc["S002", "mixed_across_runs"])
        assert not bool(summary.loc["S002", "all_runs_usable"])

    def test_a_non_lateralised_subject_has_fewer_usable_runs(self, cohort):
        """Comparative, not absolute -- and that is the point.

        Asserting that a non-lateralised subject has *zero* usable runs would be
        wrong: at this sample size the criterion fires by chance a substantial
        fraction of the time, which is precisely why Phase 5B reports the observed
        rate against a permutation null rather than against zero. What must hold
        is the ordering.
        """
        X, meta = cohort
        frame = rd.run_level_diagnostics(X, meta, CHANNELS, SFREQ)
        summary = rd.summarise_subject(frame).set_index("subject_id")
        assert summary.loc["S004", "n_usable_runs"] < summary.loc["S001", "n_usable_runs"]
        assert not bool(summary.loc["S004", "all_runs_usable"])

    def test_the_usable_criterion_is_the_frozen_phase5_one(self):
        assert rd.USABLE_D_THRESHOLD == 0.5

    def test_null_rate_is_well_above_zero_at_this_sample_size(self, cohort):
        """The criterion is noisy at ~8 epochs per class, and the null proves it.

        Reported so that an observed pass rate is never read as if chance were 0.
        """
        X, meta = cohort
        null = rd.usable_rate_null(X, meta, CHANNELS, SFREQ, n_permutations=200)
        assert 0.0 < null["null_mean_usable_rate"] < 0.5
