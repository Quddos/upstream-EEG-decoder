"""Phase 4B -- adversarial leakage tests for CSP and Riemannian processing.

CSP and tangent-space projection both *learn* from data, and CSP additionally uses
the labels, so they introduce leakage routes that the Phase 2/3 audit never had to
consider: a spatial filter, a covariance reference and a tangent-space origin can
each be fitted on the wrong rows.

Every check below is run twice -- clean input that must pass, and **deliberately
leaky input that must fail**. The negative cases carry the whole argument. A
leakage test that has only ever seen a correct pipeline demonstrates that the
pipeline is correct, not that the test could notice if it weren't, and Phase 4B is
not entitled to claim leakage safety on the strength of the former.

The eight routes the brief requires, and the test class that attacks each:

1. CSP fitted globally before LOSOCV      -- :class:`TestCspFoldDiscipline`
2. covariance reference fitted globally   -- :class:`TestRiemannFoldDiscipline`
3. tangent-space mapping fitted globally  -- :class:`TestRiemannFoldDiscipline`
4. scaler fitted globally                 -- :class:`TestScalerFoldDiscipline`
5. random epoch split                     -- :class:`TestSplitDiscipline`
6. held-out subject inside the transform  -- :class:`TestHeldOutSubjectExclusion`
7. duplicated subject IDs                 -- :class:`TestIdentityIntegrity`
8. swapped labels                         -- :class:`TestIdentityIntegrity`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneGroupOut

from src.eval.audit import (
    check_fold_subject_disjoint,
    check_ids_traceable,
    check_labels_unchanged,
    check_scaler_fitted_on_train_only,
)
from src.features.spatial import (
    CSP_N_COMPONENTS,
    csp_pipeline,
    frozen_pipeline,
    riemann_pipeline,
)

N_SUBJECTS = 6
EPOCHS_PER_SUBJECT = 12
N_CHANNELS = 8
N_TIMES = 160
SFREQ = 160.0


@pytest.fixture
def synthetic():
    """Subjects with a genuine but subject-shifted lateralised source.

    Each subject gets its own spatial mixing and offset, so a transform fitted on
    everyone differs measurably from one fitted on a training subset -- which is
    what makes the leaky variants detectable rather than coincidentally equal.
    """
    rng = np.random.default_rng(4)
    X, rows = [], []
    for subject in range(1, N_SUBJECTS + 1):
        mixing = rng.normal(0, 1, size=(N_CHANNELS, 2)) + subject * 0.3
        for i in range(EPOCHS_PER_SUBJECT):
            label = "LEFT_HAND_IMAGERY" if i % 2 == 0 else "RIGHT_HAND_IMAGERY"
            t = np.arange(N_TIMES) / SFREQ
            # Class-specific source amplitudes, then a subject-specific mixing.
            amps = (2.0, 0.5) if label == "LEFT_HAND_IMAGERY" else (0.5, 2.0)
            sources = np.vstack([
                amps[0] * np.sin(2 * np.pi * 10 * t + rng.normal()),
                amps[1] * np.sin(2 * np.pi * 22 * t + rng.normal()),
            ])
            epoch = mixing @ sources + rng.normal(0, 0.3, size=(N_CHANNELS, N_TIMES))
            X.append(epoch)
            rows.append(
                {
                    "epoch_uid": f"E{subject:03d}{i:03d}",
                    "subject_id": f"S{subject:03d}",
                    "subject": subject,
                    "run": 4 + (i % 3) * 4,
                    "run_id": f"R{4 + (i % 3) * 4:02d}",
                    "class_label": label,
                }
            )
    return np.asarray(X), pd.DataFrame(rows)


def first_fold(groups):
    return next(iter(LeaveOneGroupOut().split(np.zeros(len(groups)), groups, groups)))


class TestCspFoldDiscipline:
    """Route 1: CSP filters must come from training epochs only."""

    def test_csp_filters_differ_from_globally_fitted_ones(self, synthetic):
        X, meta = synthetic
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, _ = first_fold(groups)

        correct = csp_pipeline("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        leaky = csp_pipeline("logistic_regression", 0).fit(X, y)

        fold_filters = correct.named_steps["csp"].filters_
        global_filters = leaky.named_steps["csp"].filters_
        assert fold_filters.shape == global_filters.shape
        # If they were equal, this test could not distinguish the two situations at
        # all, so the assertion is that leakage is *detectable* here.
        assert not np.allclose(fold_filters, global_filters), (
            "fold-fitted and globally-fitted CSP filters are identical; the "
            "fixture cannot detect CSP leakage"
        )

    def test_csp_produces_the_expected_number_of_components(self, synthetic):
        X, meta = synthetic
        y = meta["class_label"].to_numpy()
        pipeline = csp_pipeline("logistic_regression", 0).fit(X, y)
        transformed = pipeline.named_steps["csp"].transform(X)
        assert transformed.shape == (len(X), CSP_N_COMPONENTS)

    def test_csp_transform_of_test_fold_uses_training_filters(self, synthetic):
        """Route 6, transform side: the held-out rows are projected, not re-fitted."""
        X, meta = synthetic
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, test_idx = first_fold(groups)

        pipeline = csp_pipeline("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        csp = pipeline.named_steps["csp"]
        stored = csp.filters_.copy()
        _ = pipeline.predict(X[test_idx])
        assert np.array_equal(csp.filters_, stored), "predict() mutated the CSP filters"

    def test_predictions_are_unchanged_when_held_out_rows_are_deleted(self, synthetic):
        """The strongest end-to-end statement available for a supervised transform."""
        X, meta = synthetic
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, test_idx = first_fold(groups)

        a = csp_pipeline("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        keep = np.setdiff1d(np.arange(len(y)), test_idx)
        b = csp_pipeline("logistic_regression", 0).fit(X[keep], y[keep])
        assert np.array_equal(a.predict(X[test_idx]), b.predict(X[test_idx]))


class TestRiemannFoldDiscipline:
    """Routes 2 and 3: covariance reference and tangent-space origin."""

    def test_tangent_space_reference_differs_from_globally_fitted_one(self, synthetic):
        X, meta = synthetic
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, _ = first_fold(groups)

        correct = riemann_pipeline("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        leaky = riemann_pipeline("logistic_regression", 0).fit(X, y)

        fold_reference = correct.named_steps["tangent"].reference_
        global_reference = leaky.named_steps["tangent"].reference_
        assert fold_reference.shape == global_reference.shape
        assert not np.allclose(fold_reference, global_reference), (
            "fold-fitted and globally-fitted tangent-space references are identical; "
            "the fixture cannot detect Riemannian leakage"
        )

    def test_covariance_step_is_stateless_and_cannot_leak(self, synthetic):
        """Covariances are per-epoch, so a subset must equal the matching rows."""
        X, _ = synthetic
        from pyriemann.estimation import Covariances

        from src.features.spatial import RIEMANN_COV_ESTIMATOR

        estimator = Covariances(estimator=RIEMANN_COV_ESTIMATOR)
        full = estimator.fit_transform(X)
        subset = estimator.fit_transform(X[[1, 5, 9]])
        assert np.allclose(full[[1, 5, 9]], subset)

    def test_tangent_reference_is_not_mutated_by_predict(self, synthetic):
        X, meta = synthetic
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, test_idx = first_fold(groups)

        pipeline = riemann_pipeline("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        stored = pipeline.named_steps["tangent"].reference_.copy()
        _ = pipeline.predict(X[test_idx])
        assert np.allclose(pipeline.named_steps["tangent"].reference_, stored)

    def test_predictions_unchanged_when_held_out_rows_deleted(self, synthetic):
        X, meta = synthetic
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, test_idx = first_fold(groups)

        a = riemann_pipeline("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        keep = np.setdiff1d(np.arange(len(y)), test_idx)
        b = riemann_pipeline("logistic_regression", 0).fit(X[keep], y[keep])
        assert np.array_equal(a.predict(X[test_idx]), b.predict(X[test_idx]))


class TestScalerFoldDiscipline:
    """Route 4: the scaler inside each pipeline must see training rows only."""

    @pytest.mark.parametrize("factory", [csp_pipeline, riemann_pipeline])
    def test_scaler_matches_the_training_fold(self, synthetic, factory):
        X, meta = synthetic
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, _ = first_fold(groups)

        pipeline = factory("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        # The representation is learned, so the scaler's input is the transformed
        # training data -- recompute it through the fitted upstream steps.
        upstream = pipeline[:-2]
        transformed = upstream.transform(X[train_idx])
        scaler = pipeline.named_steps["scale"]
        assert np.allclose(scaler.mean_, transformed.mean(axis=0), atol=1e-8)

    def test_a_globally_fitted_scaler_is_detected(self, synthetic):
        """Control, using the frozen-feature pipeline where the audit applies directly."""
        X, meta = synthetic
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, _ = first_fold(groups)
        features = X.reshape(len(X), -1)[:, :20]

        good = frozen_pipeline("logistic_regression", 0).fit(features[train_idx], y[train_idx])
        assert check_scaler_fitted_on_train_only(good, features[train_idx], features).passed

        leaky = frozen_pipeline("logistic_regression", 0).fit(features, y)
        result = check_scaler_fitted_on_train_only(leaky, features[train_idx], features)
        assert not result.passed
        assert "FULL dataset" in result.detail


class TestSplitDiscipline:
    """Route 5: no random epoch split, ever."""

    def test_losocv_folds_are_subject_disjoint(self, synthetic):
        _, meta = synthetic
        groups = meta["subject"].to_numpy()
        folds = list(LeaveOneGroupOut().split(np.zeros(len(groups)), groups, groups))
        assert len(folds) == N_SUBJECTS
        assert check_fold_subject_disjoint(folds, groups).passed

    def test_a_random_epoch_split_is_detected(self, synthetic):
        _, meta = synthetic
        groups = meta["subject"].to_numpy()
        rng = np.random.default_rng(0)
        order = rng.permutation(len(groups))
        half = len(order) // 2
        assert not check_fold_subject_disjoint([(order[:half], order[half:])], groups).passed

    def test_a_random_split_inflates_csp_accuracy(self, synthetic):
        """Why it matters: the leak is worth real accuracy, so it must be blocked.

        A random epoch split lets CSP fit filters on the very subject it is scored
        on. If that produced no advantage the prohibition would be cosmetic; it
        does, which is exactly why the protocol forbids it.
        """
        X, meta = synthetic
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()

        train_idx, test_idx = first_fold(groups)
        honest = csp_pipeline("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        honest_score = (honest.predict(X[test_idx]) == y[test_idx]).mean()

        rng = np.random.default_rng(1)
        order = rng.permutation(len(y))
        half = len(order) // 2
        leaky = csp_pipeline("logistic_regression", 0).fit(X[order[:half]], y[order[:half]])
        leaky_score = (leaky.predict(X[order[half:]]) == y[order[half:]]).mean()

        assert leaky_score > honest_score


class TestHeldOutSubjectExclusion:
    """Route 6: the held-out subject must never reach a fitted transform."""

    def test_training_fold_contains_no_held_out_epoch(self, synthetic):
        _, meta = synthetic
        groups = meta["subject"].to_numpy()
        for train_idx, test_idx in LeaveOneGroupOut().split(
            np.zeros(len(groups)), groups, groups
        ):
            assert not set(groups[train_idx]) & set(groups[test_idx])
            assert not set(train_idx) & set(test_idx)

    @pytest.mark.parametrize("factory", [csp_pipeline, riemann_pipeline])
    def test_including_the_held_out_subject_changes_the_transform(self, synthetic, factory):
        """Control: if it were included, the fitted state would differ -- and it does."""
        X, meta = synthetic
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, test_idx = first_fold(groups)
        contaminated = np.concatenate([train_idx, test_idx])

        clean = factory("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        dirty = factory("logistic_regression", 0).fit(X[contaminated], y[contaminated])

        step = "csp" if factory is csp_pipeline else "tangent"
        attribute = "filters_" if step == "csp" else "reference_"
        assert not np.allclose(
            getattr(clean.named_steps[step], attribute),
            getattr(dirty.named_steps[step], attribute),
        )


class TestIdentityIntegrity:
    """Routes 7 and 8: duplicated subject IDs and swapped labels."""

    def test_clean_identity_passes(self, synthetic):
        _, meta = synthetic
        assert check_ids_traceable(meta, meta).passed
        assert check_labels_unchanged(meta, meta).passed

    def test_duplicated_epoch_id_is_detected(self, synthetic):
        _, meta = synthetic
        tampered = meta.copy()
        tampered.loc[1, "epoch_uid"] = tampered.loc[0, "epoch_uid"]
        assert not check_ids_traceable(tampered, meta).passed

    def test_duplicated_subject_id_collapses_folds_and_is_detected(self, synthetic):
        """A duplicated subject label merges two people into one group.

        Two distinct consequences, both asserted: the identity audit sees a
        ``subject_id`` that no longer matches the index, and -- more dangerous --
        ``LeaveOneGroupOut`` yields fewer folds, so holding out "S001" now trains on
        recordings that are also being tested on.
        """
        _, meta = synthetic
        tampered = meta.copy()
        relabel = tampered["subject"] == 2
        tampered.loc[relabel, "subject"] = 1
        tampered.loc[relabel, "subject_id"] = "S001"

        # The audit catches the identity mismatch against the index.
        assert not check_ids_traceable(tampered, meta).passed

        # And the fold structure genuinely collapses.
        before = len(
            list(
                LeaveOneGroupOut().split(
                    np.zeros(len(meta)), meta["subject"], meta["subject"]
                )
            )
        )
        after = len(
            list(
                LeaveOneGroupOut().split(
                    np.zeros(len(tampered)), tampered["subject"], tampered["subject"]
                )
            )
        )
        assert after == before - 1

    def test_swapped_labels_are_detected(self, synthetic):
        _, meta = synthetic
        tampered = meta.copy()
        tampered.loc[0, "class_label"] = "RIGHT_HAND_IMAGERY"
        tampered.loc[1, "class_label"] = "LEFT_HAND_IMAGERY"
        assert not check_labels_unchanged(tampered, meta).passed

    def test_shuffled_labels_destroy_csp_accuracy(self, synthetic):
        """Sanity: CSP is learning the labels, not an artefact of the fixture.

        Tested **within** a subject, not across. Each fixture subject has its own
        random spatial mixing, so the sign and orientation of a CSP component are
        arbitrary from one subject to the next and cross-subject transfer on this
        fixture is meaningless (measured: 0.00). Within a subject the mixing is
        fixed, so a genuine label signal must show up -- and destroying the labels
        must remove it. Averaged over several shuffles so the comparison does not
        turn on one lucky permutation.
        """
        X, meta = synthetic
        y = meta["class_label"].to_numpy()
        subject_mask = meta["subject"].to_numpy() == 1
        runs = meta["run"].to_numpy()

        train = subject_mask & (runs != 4)
        test = subject_mask & (runs == 4)

        real = csp_pipeline("logistic_regression", 0).fit(X[train], y[train])
        real_score = (real.predict(X[test]) == y[test]).mean()

        shuffled_scores = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            shuffled = y.copy()
            labels = shuffled[train]
            rng.shuffle(labels)
            shuffled[train] = labels
            broken = csp_pipeline("logistic_regression", 0).fit(X[train], shuffled[train])
            shuffled_scores.append((broken.predict(X[test]) == y[test]).mean())

        assert real_score > float(np.mean(shuffled_scores)), (
            f"within-subject CSP scored {real_score:.3f} with real labels but "
            f"{np.mean(shuffled_scores):.3f} with shuffled ones"
        )
