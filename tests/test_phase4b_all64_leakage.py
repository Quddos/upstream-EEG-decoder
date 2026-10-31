"""Phase 4B-final -- leakage tests for the all-64-channel condition.

The Phase 4B audit ran on a 21-channel montage. Widening to 64 changes the risk
profile rather than leaving it alone, so the checks are repeated at the real
dimensionality instead of being assumed to carry over:

* CSP now solves a 64x64 generalised eigenproblem, so a globally-fitted filter set
  has far more freedom to encode the held-out subject than a 21-channel one did.
* The tangent space is 2,080-dimensional against ~1,750 training epochs, i.e. more
  features than samples. A leaked reference point in that regime could be worth a
  great deal of spurious accuracy, and a scaler fitted globally likewise.

Same discipline as every audit in this project: each route is exercised twice, once
on clean input that must pass and once on deliberately leaky input that must be
caught. The negative cases are the evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneGroupOut

from src.eval.audit import check_fold_subject_disjoint, check_scaler_fitted_on_train_only
from src.features.spatial import csp_pipeline, frozen_pipeline, riemann_pipeline

N_SUBJECTS = 5
EPOCHS_PER_SUBJECT = 14
N_CHANNELS = 64
N_TIMES = 160


@pytest.fixture(scope="module")
def wide():
    """Full-width montage with a genuine class signal and per-subject mixing.

    64 channels is the point of these tests, so the fixture uses the real width
    rather than a stand-in. Module-scoped because a 64-channel CSP fit is not
    cheap and every test here reuses the same array.
    """
    rng = np.random.default_rng(11)
    X, rows = [], []
    for subject in range(1, N_SUBJECTS + 1):
        mixing = rng.normal(0, 1, size=(N_CHANNELS, 3)) + subject * 0.25
        for i in range(EPOCHS_PER_SUBJECT):
            label = "LEFT_HAND_IMAGERY" if i % 2 == 0 else "RIGHT_HAND_IMAGERY"
            t = np.arange(N_TIMES) / 160.0
            amps = (2.0, 0.5, 1.0) if label == "LEFT_HAND_IMAGERY" else (0.5, 2.0, 1.0)
            sources = np.vstack([
                amps[0] * np.sin(2 * np.pi * 10 * t + rng.normal()),
                amps[1] * np.sin(2 * np.pi * 22 * t + rng.normal()),
                amps[2] * np.sin(2 * np.pi * 15 * t + rng.normal()),
            ])
            X.append(mixing @ sources + rng.normal(0, 0.4, size=(N_CHANNELS, N_TIMES)))
            rows.append(
                {
                    "epoch_uid": f"E{subject:03d}{i:03d}",
                    "subject_id": f"S{subject:03d}",
                    "subject": subject,
                    "run": 4 + (i % 3) * 4,
                    "class_label": label,
                }
            )
    return np.asarray(X), pd.DataFrame(rows)


def first_fold(groups):
    return next(iter(LeaveOneGroupOut().split(np.zeros(len(groups)), groups, groups)))


class TestGlobalCspFitting:
    """Route 1 at 64 channels: CSP filters must come from the training fold."""

    def test_fold_and_global_filters_differ(self, wide):
        X, meta = wide
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, _ = first_fold(groups)

        fold = csp_pipeline("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        glob = csp_pipeline("logistic_regression", 0).fit(X, y)

        assert fold.named_steps["csp"].filters_.shape[1] == N_CHANNELS
        assert not np.allclose(
            fold.named_steps["csp"].filters_, glob.named_steps["csp"].filters_
        ), "64-channel CSP leakage would be undetectable with this fixture"

    def test_deleting_held_out_rows_changes_nothing(self, wide):
        X, meta = wide
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, test_idx = first_fold(groups)

        a = csp_pipeline("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        keep = np.setdiff1d(np.arange(len(y)), test_idx)
        b = csp_pipeline("logistic_regression", 0).fit(X[keep], y[keep])
        assert np.array_equal(a.predict(X[test_idx]), b.predict(X[test_idx]))


class TestGlobalCovarianceAndTangent:
    """Routes 2 and 3 at 64 channels: 2,080-dimensional tangent space."""

    def test_fold_and_global_reference_differ(self, wide):
        X, meta = wide
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, _ = first_fold(groups)

        fold = riemann_pipeline("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        glob = riemann_pipeline("logistic_regression", 0).fit(X, y)

        reference = fold.named_steps["tangent"].reference_
        assert reference.shape == (N_CHANNELS, N_CHANNELS)
        assert not np.allclose(reference, glob.named_steps["tangent"].reference_)

    def test_tangent_dimension_is_the_full_upper_triangle(self, wide):
        X, meta = wide
        y = meta["class_label"].to_numpy()
        pipeline = riemann_pipeline("logistic_regression", 0).fit(X, y)
        transformed = pipeline[:-2].transform(X)
        assert transformed.shape[1] == N_CHANNELS * (N_CHANNELS + 1) // 2 == 2080

    def test_covariance_step_remains_stateless_at_64_channels(self, wide):
        X, _ = wide
        from pyriemann.estimation import Covariances

        from src.features.spatial import RIEMANN_COV_ESTIMATOR

        estimator = Covariances(estimator=RIEMANN_COV_ESTIMATOR)
        full = estimator.fit_transform(X)
        subset = estimator.fit_transform(X[[0, 7, 21]])
        assert np.allclose(full[[0, 7, 21]], subset)

    def test_deleting_held_out_rows_changes_nothing(self, wide):
        X, meta = wide
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, test_idx = first_fold(groups)

        a = riemann_pipeline("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        keep = np.setdiff1d(np.arange(len(y)), test_idx)
        b = riemann_pipeline("logistic_regression", 0).fit(X[keep], y[keep])
        assert np.array_equal(a.predict(X[test_idx]), b.predict(X[test_idx]))


class TestHeldOutContamination:
    """Route 6 at 64 channels: including the held-out subject must change the fit."""

    @pytest.mark.parametrize("factory,step,attribute", [
        (csp_pipeline, "csp", "filters_"),
        (riemann_pipeline, "tangent", "reference_"),
    ])
    def test_contaminated_training_set_changes_the_transform(
        self, wide, factory, step, attribute
    ):
        X, meta = wide
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, test_idx = first_fold(groups)

        clean = factory("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        dirty_idx = np.concatenate([train_idx, test_idx])
        dirty = factory("logistic_regression", 0).fit(X[dirty_idx], y[dirty_idx])

        assert not np.allclose(
            getattr(clean.named_steps[step], attribute),
            getattr(dirty.named_steps[step], attribute),
        )


class TestScalerAtFullWidth:
    """Route 4: with 2,080 features a globally fitted scaler is a real risk."""

    def test_scaler_matches_the_training_fold(self, wide):
        X, meta = wide
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, _ = first_fold(groups)

        pipeline = riemann_pipeline("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        transformed = pipeline[:-2].transform(X[train_idx])
        assert np.allclose(
            pipeline.named_steps["scale"].mean_, transformed.mean(axis=0), atol=1e-8
        )

    def test_a_globally_fitted_scaler_is_detected(self, wide):
        X, meta = wide
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()
        train_idx, _ = first_fold(groups)
        features = X.reshape(len(X), -1)[:, :64]

        good = frozen_pipeline("logistic_regression", 0).fit(features[train_idx], y[train_idx])
        assert check_scaler_fitted_on_train_only(good, features[train_idx], features).passed

        leaky = frozen_pipeline("logistic_regression", 0).fit(features, y)
        result = check_scaler_fitted_on_train_only(leaky, features[train_idx], features)
        assert not result.passed
        assert "FULL dataset" in result.detail


class TestRandomSplitAtFullWidth:
    """Route 5: and a demonstration that the leak is worth blocking."""

    def test_losocv_folds_are_disjoint(self, wide):
        _, meta = wide
        groups = meta["subject"].to_numpy()
        folds = list(LeaveOneGroupOut().split(np.zeros(len(groups)), groups, groups))
        assert check_fold_subject_disjoint(folds, groups).passed

    def test_random_epoch_split_is_detected(self, wide):
        _, meta = wide
        groups = meta["subject"].to_numpy()
        rng = np.random.default_rng(3)
        order = rng.permutation(len(groups))
        half = len(order) // 2
        assert not check_fold_subject_disjoint([(order[:half], order[half:])], groups).passed

    def test_random_split_inflates_riemannian_accuracy(self, wide):
        """At 2,080 features the inflation should be large -- and it is.

        This is the quantitative case for the prohibition: with more features than
        samples, letting the same subject appear on both sides of the split is not
        a small bias.
        """
        X, meta = wide
        y = meta["class_label"].to_numpy()
        groups = meta["subject"].to_numpy()

        train_idx, test_idx = first_fold(groups)
        honest = riemann_pipeline("logistic_regression", 0).fit(X[train_idx], y[train_idx])
        honest_score = (honest.predict(X[test_idx]) == y[test_idx]).mean()

        rng = np.random.default_rng(5)
        order = rng.permutation(len(y))
        half = len(order) // 2
        leaky = riemann_pipeline("logistic_regression", 0).fit(X[order[:half]], y[order[:half]])
        leaky_score = (leaky.predict(X[order[half:]]) == y[order[half:]]).mean()

        assert leaky_score > honest_score
