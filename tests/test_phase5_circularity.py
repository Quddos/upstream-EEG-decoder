"""Phase 5 -- circularity and leakage tests for the diagnostic audit.

Phase 5 makes a claim of a kind the earlier phases did not: that a property
measured in the data explains a model outcome. That claim is worthless if the
property was derived from the outcome, computed with knowledge of it, or defined
so as to correlate with it. These tests attack those routes specifically.

The three ways this analysis could be circular, and the class that attacks each:

1. **the diagnostic sees the outcome** -- a diagnostic computed from, selected by,
   or tuned against LOSOCV accuracy. :class:`TestDiagnosticsAreOutcomeBlind`
2. **the diagnostic is a model** -- a fitted discriminant reported as if it were a
   descriptive property of the signal. :class:`TestDiagnosticsAreNotModels`
3. **the diagnostic borrows from other subjects** -- a per-subject quantity that
   actually depends on the rest of the cohort, which would smuggle population
   structure into a subject-level claim. :class:`TestSubjectIndependence`

As everywhere in this project, checks are exercised twice: clean input must pass
and a deliberately broken variant must be caught.
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
from src.features import diagnostics as diag

N_SUBJECTS = 6
PER_SUBJECT = 18
N_CHANNELS = 4
N_TIMES = 320
SFREQ = 160.0
CHANNELS = ("C3", "C4", "Cz", "C1")


@pytest.fixture(scope="module")
def cohort():
    """Subjects that differ in whether they have a lateralised pattern at all.

    Subjects 1-3 get a real class-dependent C3/C4 imbalance; 4-6 get none. That
    makes the fixture able to distinguish "has signal" from "lacks signal", which
    is precisely the discrimination Phase 5 claims to make.
    """
    rng = np.random.default_rng(5)
    X, rows = [], []
    t = np.arange(N_TIMES) / SFREQ
    for subject in range(1, N_SUBJECTS + 1):
        lateralised = subject <= 3
        for i in range(PER_SUBJECT):
            cls = (LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY, REST)[i % 3]
            epoch = rng.normal(0, 1e-6, size=(N_CHANNELS, N_TIMES))
            mu_amp = 4e-6 if cls == REST else 2e-6
            gain = {"C3": 1.0, "C4": 1.0, "Cz": 1.0, "C1": 1.0}
            if lateralised and cls == LEFT_HAND_IMAGERY:
                gain["C4"] = 0.4
            if lateralised and cls == RIGHT_HAND_IMAGERY:
                gain["C3"] = 0.4
            for c, name in enumerate(CHANNELS):
                epoch[c] += gain[name] * mu_amp * np.sin(2 * np.pi * 10 * t + rng.normal())
            X.append(epoch)
            rows.append(
                {
                    "epoch_uid": f"E{subject:03d}{i:03d}",
                    "subject_id": f"S{subject:03d}",
                    "subject": subject,
                    "run": 4 + (i % 3) * 4,
                    "class_label": cls,
                }
            )
    return np.asarray(X), pd.DataFrame(rows)


class TestDiagnosticsAreOutcomeBlind:
    """Route 1: no diagnostic may see, read, or depend on the LOSOCV outcome."""

    def test_diagnostics_module_imports_nothing_that_carries_an_outcome(self):
        """Structural, via the AST -- prose in docstrings is not a violation.

        String-matching the source would flag the module's own explanation of why
        it is outcome-blind, so the check parses the code instead: it inspects the
        actual import statements and the actual call targets.
        """
        tree = ast.parse(inspect.getsource(diag))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "sklearn" not in imported
        assert not {"json", "pickle"} & imported

    def test_diagnostics_module_reads_no_files(self):
        """It cannot load a results file, so it cannot see an outcome."""
        tree = ast.parse(inspect.getsource(diag))
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = node.func
                if isinstance(target, ast.Name):
                    called.add(target.id)
                elif isinstance(target, ast.Attribute):
                    called.add(target.attr)
        for forbidden in ("open", "read_csv", "read_json", "load", "loads"):
            assert forbidden not in called, (
                f"diagnostics module calls {forbidden!r}; it must not read any file"
            )

    def test_signature_takes_only_signal_and_metadata(self, cohort):
        """The function cannot receive an outcome even if a caller wanted to pass one."""
        parameters = set(inspect.signature(diag.subject_diagnostics).parameters)
        assert parameters == {"X", "metadata", "channels", "sfreq"}

    def test_diagnostics_unchanged_when_an_outcome_column_is_added(self, cohort):
        """Control: injecting an outcome into the metadata must change nothing."""
        X, meta = cohort
        clean = diag.subject_diagnostics(X, meta, CHANNELS, SFREQ)

        contaminated = meta.copy()
        rng = np.random.default_rng(0)
        mapping = {s: rng.random() for s in contaminated["subject_id"].unique()}
        contaminated["losocv_balanced_accuracy"] = contaminated["subject_id"].map(mapping)
        after = diag.subject_diagnostics(X, contaminated, CHANNELS, SFREQ)

        numeric = clean.select_dtypes(include=[np.number]).columns
        assert np.allclose(clean[numeric].to_numpy(), after[numeric].to_numpy(),
                           equal_nan=True)

    def test_outcome_labels_come_only_from_the_frozen_phase4b_result(self):
        """The group definition must cite its pre-existing source, not Phase 5."""
        path = Path("data/metadata/phase5/phase5_outcome_definition.json")
        if not path.exists():
            pytest.skip("outcome definition not present in this checkout")
        definition = json.loads(path.read_text())
        assert "phase4b" in definition["source"]
        assert definition["threshold"] == 0.50
        assert definition["n_below"] + definition["n_above"] == 40


class TestDiagnosticsAreNotModels:
    """Route 2: a descriptive statistic must not be a fitted discriminant."""

    def test_fisher_ratio_inverts_no_covariance(self):
        """Checked on the AST, so the docstring explaining this does not trip it."""
        tree = ast.parse(inspect.getsource(diag.fisher_ratio).strip())
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = node.func
                if isinstance(target, ast.Name):
                    called.add(target.id)
                elif isinstance(target, ast.Attribute):
                    called.add(target.attr)
        for forbidden in ("inv", "pinv", "solve", "lstsq", "fit", "fit_transform"):
            assert forbidden not in called

    def test_fisher_ratio_is_a_closed_form_of_its_inputs(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([2.0, 3.0, 4.0, 5.0])
        expected = (a.mean() - b.mean()) ** 2 / (a.var(ddof=1) + b.var(ddof=1))
        assert diag.fisher_ratio(a, b) == pytest.approx(expected)

    def test_fisher_ratio_is_symmetric_and_sign_free(self):
        a = np.array([1.0, 2.0, 3.0, 9.0])
        b = np.array([4.0, 5.0, 6.0, 2.0])
        assert diag.fisher_ratio(a, b) == pytest.approx(diag.fisher_ratio(b, a))

    def test_diagnostics_are_deterministic(self, cohort):
        X, meta = cohort
        first = diag.subject_diagnostics(X, meta, CHANNELS, SFREQ)
        second = diag.subject_diagnostics(X, meta, CHANNELS, SFREQ)
        numeric = first.select_dtypes(include=[np.number]).columns
        assert np.allclose(first[numeric].to_numpy(), second[numeric].to_numpy(),
                           equal_nan=True)


class TestSubjectIndependence:
    """Route 3: a per-subject diagnostic must not depend on the rest of the cohort."""

    def test_subset_gives_identical_values_for_the_retained_subjects(self, cohort):
        """The structural guarantee: no pooled statistic anywhere in the pipeline."""
        X, meta = cohort
        full = diag.subject_diagnostics(X, meta, CHANNELS, SFREQ).set_index("subject_id")

        keep = meta["subject_id"].isin(["S001", "S004"]).to_numpy()
        partial = diag.subject_diagnostics(
            X[keep], meta[keep].reset_index(drop=True), CHANNELS, SFREQ
        ).set_index("subject_id")

        numeric = partial.select_dtypes(include=[np.number]).columns
        for subject in partial.index:
            assert np.allclose(
                full.loc[subject, numeric].to_numpy(float),
                partial.loc[subject, numeric].to_numpy(float),
                equal_nan=True,
            ), f"{subject} diagnostics changed when other subjects were removed"

    def test_a_cohort_normalised_diagnostic_would_be_detected(self, cohort):
        """Control: a deliberately pooled statistic must fail the subset check."""
        X, meta = cohort

        def leaky(X_, meta_, channels, sfreq):
            frame = diag.subject_diagnostics(X_, meta_, channels, sfreq)
            # z-score across subjects -- a cohort-level statistic
            frame["mu_snr_mean"] = (
                (frame["mu_snr_mean"] - frame["mu_snr_mean"].mean())
                / frame["mu_snr_mean"].std(ddof=1)
            )
            return frame

        full = leaky(X, meta, CHANNELS, SFREQ).set_index("subject_id")
        keep = meta["subject_id"].isin(["S001", "S004"]).to_numpy()
        partial = leaky(X[keep], meta[keep].reset_index(drop=True),
                        CHANNELS, SFREQ).set_index("subject_id")
        assert not np.allclose(
            full.loc["S001", "mu_snr_mean"], partial.loc["S001", "mu_snr_mean"]
        )

    def test_subject_order_does_not_matter(self, cohort):
        X, meta = cohort
        forward = diag.subject_diagnostics(X, meta, CHANNELS, SFREQ).set_index("subject_id")
        order = np.arange(len(meta))[::-1]
        reversed_frame = diag.subject_diagnostics(
            X[order], meta.iloc[order].reset_index(drop=True), CHANNELS, SFREQ
        ).set_index("subject_id")
        numeric = forward.select_dtypes(include=[np.number]).columns
        assert np.allclose(
            forward[numeric].to_numpy(), reversed_frame.loc[forward.index, numeric].to_numpy(),
            equal_nan=True,
        )


class TestDiagnosticsMeasureWhatTheyClaim:
    """The diagnostics must respond to signal, or a null result means nothing."""

    def test_lateralised_subjects_score_higher_than_flat_ones(self, cohort):
        X, meta = cohort
        frame = diag.subject_diagnostics(X, meta, CHANNELS, SFREQ).set_index("subject_id")
        lateralised = frame.loc[["S001", "S002", "S003"], "fisher_lateralisation"]
        flat = frame.loc[["S004", "S005", "S006"], "fisher_lateralisation"]
        assert lateralised.min() > flat.max(), (
            "the diagnostic cannot distinguish subjects with a lateralised pattern "
            "from subjects without one, so a null result would be uninformative"
        )

    def test_shuffling_labels_destroys_the_lateralisation_diagnostic(self, cohort):
        """Control: the diagnostic must depend on the labels, not on the epochs alone."""
        X, meta = cohort
        real = diag.subject_diagnostics(X, meta, CHANNELS, SFREQ).set_index("subject_id")

        rng = np.random.default_rng(1)
        shuffled = meta.copy()
        for subject in shuffled["subject_id"].unique():
            mask = shuffled["subject_id"] == subject
            values = shuffled.loc[mask, "class_label"].to_numpy()
            rng.shuffle(values)
            shuffled.loc[mask, "class_label"] = values
        broken = diag.subject_diagnostics(X, shuffled, CHANNELS, SFREQ).set_index("subject_id")

        assert (real.loc[["S001", "S002", "S003"], "fisher_lateralisation"].mean()
                > broken.loc[["S001", "S002", "S003"], "fisher_lateralisation"].mean())

    def test_mu_snr_does_not_depend_on_the_labels(self, cohort):
        """Signal *presence* is a label-free property; it must behave like one.

        This is what allows mu SNR and lateralisation to dissociate in the real
        data -- they are measuring different things by construction.
        """
        X, meta = cohort
        real = diag.subject_diagnostics(X, meta, CHANNELS, SFREQ).set_index("subject_id")
        rng = np.random.default_rng(2)
        shuffled = meta.copy()
        for subject in shuffled["subject_id"].unique():
            mask = shuffled["subject_id"] == subject
            values = shuffled.loc[mask, "class_label"].to_numpy()
            rng.shuffle(values)
            shuffled.loc[mask, "class_label"] = values
        broken = diag.subject_diagnostics(X, shuffled, CHANNELS, SFREQ).set_index("subject_id")
        assert np.allclose(real["mu_snr_mean"].to_numpy(),
                           broken["mu_snr_mean"].to_numpy())


class TestTerminology:
    """The brief forbids calling the task-vs-rest contrast ERD; enforce it."""

    def test_no_column_is_named_erd(self, cohort):
        X, meta = cohort
        frame = diag.subject_diagnostics(X, meta, CHANNELS, SFREQ)
        for column in frame.columns:
            assert "erd" not in column.lower(), (
                f"{column} implies event-related desynchronisation, which this "
                f"epoching cannot measure (no pre-cue baseline)"
            )

    def test_the_module_documents_why(self):
        source = inspect.getsource(diag)
        assert "no pre-cue baseline" in source.lower()
        assert "task_vs_rest_mu_change" in source
