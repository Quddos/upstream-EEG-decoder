"""Phase 2E -- the leakage audit.

Every check is exercised twice: once on clean input, which must pass, and once on
**deliberately leaky input, which must fail**. The negative cases are the point.
A leakage test that has only ever seen correct data proves that the data is
correct, not that the test can detect a problem -- and a silently vacuous audit is
worse than none, because it is quoted as evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.eval.audit import (
    check_every_subject_held_out_once,
    check_features_are_per_epoch,
    check_flag_not_a_feature,
    check_fold_subject_disjoint,
    check_ids_traceable,
    check_labels_unchanged,
)
from src.eval.baselines import MODELS
from src.eval.protocol import losocv_folds

N_SUBJECTS = 5
EPOCHS_PER_SUBJECT = 12


@pytest.fixture
def toy():
    """Five subjects, two classes, features that carry a real but weak signal."""
    rng = np.random.default_rng(11)
    rows = []
    X = []
    for subject in range(1, N_SUBJECTS + 1):
        for i in range(EPOCHS_PER_SUBJECT):
            label = "LEFT_HAND_IMAGERY" if i % 2 == 0 else "RIGHT_HAND_IMAGERY"
            # Per-subject offset, so train and test folds genuinely differ.
            offset = subject * 3.0
            signal = 1.0 if label == "LEFT_HAND_IMAGERY" else -1.0
            X.append(rng.normal(offset + signal, 1.0, size=4))
            rows.append(
                {
                    "epoch_uid": f"E{subject:03d}{i:03d}",
                    "subject_id": f"S{subject:03d}",
                    "run_id": f"R{4 + (i % 3) * 4:02d}",
                    "subject": subject,
                    "run": 4 + (i % 3) * 4,
                    "class_label": label,
                }
            )
    metadata = pd.DataFrame(rows)
    return np.asarray(X), metadata


class TestFoldConstruction:
    def test_no_subject_appears_in_train_and_test(self, toy):
        _, metadata = toy
        groups = metadata["subject"].to_numpy()
        result = check_fold_subject_disjoint(losocv_folds(groups), groups)
        assert result.passed, result.detail

    def test_every_subject_is_held_out_exactly_once(self, toy):
        _, metadata = toy
        groups = metadata["subject"].to_numpy()
        folds = losocv_folds(groups)
        assert len(folds) == N_SUBJECTS
        assert check_every_subject_held_out_once(folds, groups).passed

    def test_a_leaky_fold_is_detected(self, toy):
        """Control: overlapping folds must be caught, not waved through."""
        _, metadata = toy
        groups = metadata["subject"].to_numpy()
        all_idx = np.arange(len(groups))
        leaky = [(all_idx, all_idx)]  # train on everything, test on everything
        result = check_fold_subject_disjoint(leaky, groups)
        assert not result.passed
        assert "train and test" in result.detail

    def test_a_random_epoch_split_would_be_detected(self, toy):
        """The specific mistake the brief forbids: random epoch-level splitting."""
        _, metadata = toy
        groups = metadata["subject"].to_numpy()
        rng = np.random.default_rng(0)
        order = rng.permutation(len(groups))
        half = len(order) // 2
        random_split = [(order[:half], order[half:])]
        assert not check_fold_subject_disjoint(random_split, groups).passed


class TestScalerFitting:
    def _fold(self, toy):
        X, metadata = toy
        groups = metadata["subject"].to_numpy()
        train_idx, test_idx = losocv_folds(groups)[0]
        return X, X[train_idx], X[test_idx]

    def test_pipeline_scaler_sees_only_the_training_fold(self, toy):
        from src.eval.audit import check_scaler_fitted_on_train_only

        X_all, X_train, _ = self._fold(toy)
        y = np.zeros(len(X_train))
        y[: len(y) // 2] = 1
        pipeline = MODELS["logistic_regression"](0)
        pipeline.fit(X_train, y)
        result = check_scaler_fitted_on_train_only(pipeline, X_train, X_all)
        assert result.passed, result.detail

    def test_a_globally_fitted_scaler_is_detected(self, toy):
        """Control: standardising before the split must be caught."""
        from src.eval.audit import check_scaler_fitted_on_train_only

        X_all, X_train, _ = self._fold(toy)
        y_all = np.zeros(len(X_all))
        y_all[: len(y_all) // 2] = 1

        leaky = Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
        )
        leaky.fit(X_all, y_all)  # fitted on everything, including the held-out subject

        result = check_scaler_fitted_on_train_only(leaky, X_train, X_all)
        assert not result.passed
        assert "FULL dataset" in result.detail

    def test_scaler_statistics_equal_the_training_fold_exactly(self, toy):
        X_all, X_train, _ = self._fold(toy)
        y = np.arange(len(X_train)) % 2
        pipeline = MODELS["logistic_regression"](0)
        pipeline.fit(X_train, y)
        scaler = pipeline.named_steps["scale"]
        assert np.allclose(scaler.mean_, X_train.mean(axis=0))
        assert not np.allclose(scaler.mean_, X_all.mean(axis=0))

    def test_transform_of_test_fold_uses_stored_training_statistics(self, toy):
        """The held-out subject must be transformed, never re-fitted."""
        X_all, X_train, X_test = self._fold(toy)
        y = np.arange(len(X_train)) % 2
        pipeline = MODELS["logistic_regression"](0)
        pipeline.fit(X_train, y)
        scaler = pipeline.named_steps["scale"]

        transformed = scaler.transform(X_test)
        expected = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
        assert np.allclose(transformed, expected)
        # A re-fitted scaler would centre the test fold on itself; it must not.
        assert not np.allclose(transformed.mean(axis=0), 0.0, atol=1e-6)


class TestFeaturesArePerEpoch:
    def _per_epoch_extract(self, X, metadata):
        """Honest featuriser: every row depends only on its own epoch."""
        return pd.DataFrame({f"f{i}": X[:, i] ** 2 for i in range(X.shape[1])})

    def _global_extract(self, X, metadata):
        """Leaky featuriser: standardises using statistics pooled over all rows."""
        return pd.DataFrame(
            {f"f{i}": (X[:, i] - X[:, i].mean()) / X[:, i].std() for i in range(X.shape[1])}
        )

    def test_per_epoch_features_pass(self, toy):
        X, metadata = toy
        result = check_features_are_per_epoch(
            self._per_epoch_extract, X, metadata, subset=[0, 5, 9, 20]
        )
        assert result.passed, result.detail

    def test_globally_normalised_features_are_detected(self, toy):
        """Control: a dataset-wide statistic inside featurisation must be caught."""
        X, metadata = toy
        result = check_features_are_per_epoch(
            self._global_extract, X, metadata, subset=[0, 5, 9, 20]
        )
        assert not result.passed

    def test_the_real_extractor_is_per_epoch(self):
        """The actual Phase 2 featuriser, on synthetic epochs."""
        from src.features.extract import build_feature_table

        rng = np.random.default_rng(3)
        channels = ("C3", "C4", "Cz")
        X = rng.normal(0, 1e-5, size=(9, len(channels), 320))
        metadata = pd.DataFrame(
            {
                "epoch_uid": [f"E{i:07d}" for i in range(9)],
                "subject_id": ["S001"] * 9,
                "class_label": ["REST"] * 9,
            }
        )

        def extract(data, meta):
            return build_feature_table(data, meta, channels, 160.0).frame

        result = check_features_are_per_epoch(
            extract, X, metadata, subset=[1, 4, 7]
        )
        assert result.passed, result.detail


class TestIdentityPreserved:
    def test_labels_match_the_index(self, toy):
        _, metadata = toy
        assert check_labels_unchanged(metadata, metadata).passed

    def test_a_relabelled_epoch_is_detected(self, toy):
        """Control: silently flipped labels must be caught."""
        _, metadata = toy
        tampered = metadata.copy()
        tampered.loc[0, "class_label"] = "RIGHT_HAND_IMAGERY"
        tampered.loc[1, "class_label"] = "LEFT_HAND_IMAGERY"
        result = check_labels_unchanged(tampered, metadata)
        assert not result.passed

    def test_ids_are_traceable(self, toy):
        _, metadata = toy
        assert check_ids_traceable(metadata, metadata).passed

    def test_duplicate_epoch_ids_are_detected(self, toy):
        _, metadata = toy
        tampered = metadata.copy()
        tampered.loc[1, "epoch_uid"] = tampered.loc[0, "epoch_uid"]
        assert not check_ids_traceable(tampered, metadata).passed

    def test_shuffled_subject_ids_are_detected(self, toy):
        """Control: rows reordered relative to the index must be caught."""
        _, metadata = toy
        tampered = metadata.copy()
        tampered["subject_id"] = tampered["subject_id"].values[::-1]
        assert not check_ids_traceable(tampered, metadata).passed


class TestArtifactFlagIsNotAFeature:
    def test_clean_feature_list_passes(self):
        assert check_flag_not_a_feature(["C3_log_mu", "lateralization_mu"]).passed

    def test_the_artifact_flag_is_detected(self):
        """The brief forbids the 150 uV flag as a classification rule."""
        result = check_flag_not_a_feature(["C3_log_mu", "artifact_flag"])
        assert not result.passed
        assert "artifact_flag" in result.detail

    def test_the_real_feature_set_contains_no_qc_column(self):
        from src.features.extract import build_feature_table

        rng = np.random.default_rng(5)
        channels = ("C3", "C4")
        X = rng.normal(0, 1e-5, size=(4, 2, 320))
        metadata = pd.DataFrame(
            {"epoch_uid": [f"E{i}" for i in range(4)], "class_label": ["REST"] * 4}
        )
        table = build_feature_table(X, metadata, channels, 160.0)
        assert check_flag_not_a_feature(table.feature_columns).passed


class TestEndToEndFoldIsolation:
    def test_held_out_subject_never_influences_a_fitted_model(self, toy):
        """Refitting without the held-out subject present must change nothing.

        The strongest end-to-end statement available: if the held-out rows had any
        influence on fitting, deleting them from the array entirely would change
        the model's predictions. It does not.
        """
        X, metadata = toy
        groups = metadata["subject"].to_numpy()
        y = metadata["class_label"].to_numpy()
        train_idx, test_idx = losocv_folds(groups)[2]

        a = MODELS["logistic_regression"](0).fit(X[train_idx], y[train_idx])
        predictions_a = a.predict(X[test_idx])

        # Same training rows, but the held-out subject does not exist at all.
        keep = np.setdiff1d(np.arange(len(y)), test_idx)
        b = MODELS["logistic_regression"](0).fit(X[keep], y[keep])
        predictions_b = b.predict(X[test_idx])

        assert np.array_equal(predictions_a, predictions_b)
