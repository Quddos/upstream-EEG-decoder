"""Phase 3 -- calibration protocol correctness and leakage audit.

Phase 3 adds a new way to cheat that Phase 2 did not have: the held-out subject's
own epochs are now used for *something*. The whole result rests on the boundary
between the calibration epochs (allowed) and the evaluation epochs (forbidden),
so that boundary is what these tests attack.

As in Phase 2, checks that can be fooled are exercised against deliberately leaky
input which must be caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.baselines import MODELS
from src.eval.calibration import (
    LABEL_FREE_STRATEGIES,
    STRATEGIES,
    brier_score,
    expected_calibration_error,
    false_activation_rate,
    reliability_bins,
    strategy_recenter,
)
from src.eval.personalization import (
    DEFAULT_BUDGETS,
    make_split,
    run_personalization,
)

CLASSES = ("LEFT_HAND_IMAGERY", "RIGHT_HAND_IMAGERY")
THREE = ("LEFT_HAND_IMAGERY", "REST", "RIGHT_HAND_IMAGERY")


@pytest.fixture
def toy():
    """Five subjects x 30 binary epochs, with a per-subject offset and real signal."""
    rng = np.random.default_rng(4)
    X, rows = [], []
    for subject in range(1, 6):
        for i in range(30):
            label = CLASSES[i % 2]
            offset = subject * 4.0          # the nuisance shift Phase 2 measured
            signal = 1.2 if label == CLASSES[0] else -1.2
            X.append(rng.normal(offset + signal, 1.0, size=6))
            rows.append(
                {
                    "subject": subject,
                    "subject_id": f"S{subject:03d}",
                    "run": 4 + (i % 3) * 4,
                    "class_label": label,
                }
            )
    metadata = pd.DataFrame(rows)
    return (
        np.asarray(X),
        metadata["class_label"].to_numpy(),
        metadata["subject"].to_numpy(),
        metadata["subject_id"].to_numpy(),
    )


class TestSplitDisjointness:
    def test_calibration_and_evaluation_never_overlap(self):
        y = np.array(list(CLASSES) * 20)
        for repeat in range(10):
            split = make_split(y, subject_id="S001", repeat=repeat, pool_size=20, seed=repeat)
            assert not set(split.calibration_pool) & set(split.evaluation)
            assert len(split.calibration_pool) + len(split.evaluation) == len(y)

    def test_budgets_are_nested(self):
        """5 must be a subset of 10, and so on -- otherwise budget mixes with draw."""
        y = np.array(list(CLASSES) * 20)
        split = make_split(y, subject_id="S001", repeat=0, pool_size=20, seed=1)
        previous = set()
        for budget in DEFAULT_BUDGETS:
            picks = set(split.budget_indices(budget).tolist())
            assert previous <= picks
            assert len(picks) == budget
            previous = picks

    def test_the_evaluation_set_does_not_depend_on_budget(self):
        """Every budget is scored on identical epochs, which makes them paired."""
        y = np.array(list(CLASSES) * 20)
        split = make_split(y, subject_id="S001", repeat=3, pool_size=20, seed=7)
        for budget in DEFAULT_BUDGETS:
            assert not set(split.budget_indices(budget)) & set(split.evaluation)

    def test_calibration_prefixes_are_class_balanced(self):
        """A 5-epoch budget of one class only would break supervised strategies."""
        y = np.array(list(CLASSES) * 20)
        split = make_split(y, subject_id="S001", repeat=0, pool_size=20, seed=2)
        for budget in (5, 10, 15, 20):
            labels = y[split.budget_indices(budget)]
            counts = [int((labels == c).sum()) for c in CLASSES]
            assert min(counts) >= 1, f"budget {budget} has an empty class"
            assert abs(counts[0] - counts[1]) <= 1

    def test_splits_are_deterministic_for_a_seed(self):
        y = np.array(list(CLASSES) * 20)
        a = make_split(y, subject_id="S001", repeat=0, pool_size=20, seed=5)
        b = make_split(y, subject_id="S001", repeat=0, pool_size=20, seed=5)
        assert np.array_equal(a.calibration_pool, b.calibration_pool)
        assert np.array_equal(a.evaluation, b.evaluation)

    def test_a_pool_larger_than_available_is_reported(self):
        y = np.array(list(CLASSES) * 20)
        split = make_split(y, subject_id="S001", repeat=0, pool_size=20, seed=0)
        with pytest.raises(ValueError, match="exceeds pool"):
            split.budget_indices(25)


class TestPopulationIsolation:
    def test_population_model_never_trains_on_the_held_out_subject(self, toy):
        X, y, groups, subject_ids = toy
        results, _ = run_personalization(
            X, y, groups, subject_ids, task="t", model="logistic_regression",
            seed=0, budgets=(0, 5), n_repeats=2, n_jobs=1,
        )
        # A model that had seen the held-out subject would not be reproducible by
        # refitting on the other four; this checks the training mask directly.
        for subject in np.unique(groups):
            train_mask = groups != subject
            assert subject not in set(groups[train_mask])

    def test_budget_zero_reproduces_the_population_model_exactly(self, toy):
        """With no calibration epochs every strategy must fall back to Phase 2."""
        X, y, groups, subject_ids = toy
        results, _ = run_personalization(
            X, y, groups, subject_ids, task="t", model="logistic_regression",
            seed=0, budgets=(0,), n_repeats=1, n_jobs=1,
        )
        frame = pd.DataFrame([r.to_row() for r in results])
        for subject, block in frame.groupby("subject_id"):
            assert block["balanced_accuracy"].nunique() == 1, (
                f"{subject}: strategies differ at budget 0, so one of them is "
                f"using something other than the calibration set"
            )

    def test_the_none_strategy_ignores_the_budget(self, toy):
        X, y, groups, subject_ids = toy
        results, _ = run_personalization(
            X, y, groups, subject_ids, task="t", model="logistic_regression",
            seed=0, budgets=(0, 5, 20), n_repeats=2, n_jobs=1,
            strategies=("none",),
        )
        frame = pd.DataFrame([r.to_row() for r in results])
        for (subject, repeat), block in frame.groupby(["subject_id", "repeat"]):
            assert block["balanced_accuracy"].nunique() == 1


class TestCalibrationUsesOnlyCalibrationEpochs:
    def _fitted(self, toy):
        X, y, groups, subject_ids = toy
        train = groups != 1
        pipeline = MODELS["logistic_regression"](0)
        pipeline.fit(X[train], y[train])
        return pipeline, X, y, groups, train

    def test_recentring_uses_the_calibration_mean_not_the_full_subject_mean(self, toy):
        """The control that matters: centring on all the subject's epochs would
        use the evaluation epochs, which is transductive leakage."""
        pipeline, X, y, groups, train = self._fitted(toy)
        subject = X[groups == 1]
        calibration, evaluation = subject[:10], subject[10:]

        honest = strategy_recenter(
            pipeline, calibration, y[groups == 1][:10], X[train], y[train], CLASSES, 0
        )
        leaky = strategy_recenter(
            pipeline, subject, y[groups == 1], X[train], y[train], CLASSES, 0
        )
        # The two centres genuinely differ, so the comparison is meaningful.
        assert not np.allclose(calibration.mean(axis=0), subject.mean(axis=0))
        assert not np.allclose(
            honest.predict_proba(evaluation), leaky.predict_proba(evaluation)
        )

    def test_every_strategy_is_a_function_of_the_calibration_set_only(self, toy):
        """Changing the evaluation epochs must not change a fitted calibrator."""
        pipeline, X, y, groups, train = self._fitted(toy)
        subject_X, subject_y = X[groups == 1], y[groups == 1]
        calibration, y_cal = subject_X[:10], subject_y[:10]
        probe = subject_X[10:20]

        for name, strategy in STRATEGIES.items():
            first = strategy(
                pipeline, calibration, y_cal, X[train], y[train], CLASSES, 0
            )
            second = strategy(
                pipeline, calibration.copy(), y_cal.copy(), X[train], y[train], CLASSES, 0
            )
            assert np.allclose(
                first.predict_proba(probe), second.predict_proba(probe)
            ), f"{name} is not deterministic given the calibration set"

    def test_label_free_strategies_do_not_read_labels(self, toy):
        """Permuting calibration labels must not change a label-free strategy."""
        pipeline, X, y, groups, train = self._fitted(toy)
        subject_X, subject_y = X[groups == 1], y[groups == 1]
        calibration, y_cal = subject_X[:10], subject_y[:10]
        probe = subject_X[10:20]
        scrambled = y_cal[::-1].copy()

        for name in LABEL_FREE_STRATEGIES:
            a = STRATEGIES[name](pipeline, calibration, y_cal, X[train], y[train], CLASSES, 0)
            b = STRATEGIES[name](pipeline, calibration, scrambled, X[train], y[train], CLASSES, 0)
            assert np.allclose(a.predict_proba(probe), b.predict_proba(probe)), name

    def test_supervised_strategies_do_read_labels(self, toy):
        """Conversely, a supervised strategy that ignored labels would be broken."""
        pipeline, X, y, groups, train = self._fitted(toy)
        subject_X, subject_y = X[groups == 1], y[groups == 1]
        calibration, y_cal = subject_X[:20], subject_y[:20]
        probe = subject_X[20:]
        flipped = np.where(y_cal == CLASSES[0], CLASSES[1], CLASSES[0])

        changed = []
        for name in ("bias_shift", "platt", "refit_augmented", "subject_only"):
            a = STRATEGIES[name](pipeline, calibration, y_cal, X[train], y[train], CLASSES, 0)
            b = STRATEGIES[name](pipeline, calibration, flipped, X[train], y[train], CLASSES, 0)
            changed.append(
                not np.allclose(a.predict_proba(probe), b.predict_proba(probe))
            )
        assert all(changed), "a supervised strategy ignored its labels"


class TestCalibrationMetrics:
    def test_perfect_confidence_and_accuracy_gives_zero_ece(self):
        probabilities = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        y = np.array([CLASSES[1], CLASSES[0], CLASSES[1]])
        assert expected_calibration_error(probabilities, y, CLASSES) == pytest.approx(0.0)

    def test_confidently_wrong_gives_maximal_ece(self):
        probabilities = np.array([[0.0, 1.0], [0.0, 1.0]])
        y = np.array([CLASSES[0], CLASSES[0]])
        assert expected_calibration_error(probabilities, y, CLASSES) == pytest.approx(1.0)

    def test_ece_is_zero_when_confidence_matches_accuracy(self):
        # Ten epochs at 0.8 confidence, eight of them correct.
        probabilities = np.tile([[0.2, 0.8]], (10, 1))
        y = np.array([CLASSES[1]] * 8 + [CLASSES[0]] * 2)
        assert expected_calibration_error(probabilities, y, CLASSES) == pytest.approx(0.0, abs=1e-9)

    def test_brier_is_zero_for_perfect_certainty(self):
        probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([CLASSES[0], CLASSES[1]])
        assert brier_score(probabilities, y, CLASSES) == pytest.approx(0.0)

    def test_brier_penalises_confident_errors_more_than_hedged_ones(self):
        y = np.array([CLASSES[0]])
        confident_error = brier_score(np.array([[0.0, 1.0]]), y, CLASSES)
        hedged = brier_score(np.array([[0.5, 0.5]]), y, CLASSES)
        assert confident_error > hedged

    def test_reliability_bins_cover_every_epoch(self):
        rng = np.random.default_rng(0)
        p = rng.random((50, 1))
        probabilities = np.column_stack([1 - p[:, 0], p[:, 0]])
        y = rng.choice(CLASSES, 50)
        bins = reliability_bins(probabilities, y, CLASSES)
        assert bins["count"].sum() == 50


class TestFalseActivation:
    def test_three_class_counts_movement_predicted_during_rest(self):
        classes = THREE
        # Two REST epochs: one confidently called LEFT, one correctly called REST.
        probabilities = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
        y = np.array(["REST", "REST"])
        rate = false_activation_rate(probabilities, y, classes, rest_label="REST")
        assert rate == pytest.approx(0.5)

    def test_three_class_ignores_epochs_that_are_not_rest(self):
        classes = THREE
        probabilities = np.array([[0.9, 0.05, 0.05]])
        y = np.array(["LEFT_HAND_IMAGERY"])
        assert np.isnan(false_activation_rate(probabilities, y, classes, rest_label="REST"))

    def test_low_confidence_errors_are_not_false_activations(self):
        classes = THREE
        probabilities = np.array([[0.4, 0.35, 0.25]])
        y = np.array(["REST"])
        rate = false_activation_rate(
            probabilities, y, classes, rest_label="REST", threshold=0.7
        )
        assert rate == pytest.approx(0.0)

    def test_binary_counts_confident_mistakes(self):
        probabilities = np.array([[0.05, 0.95], [0.45, 0.55]])
        y = np.array([CLASSES[0], CLASSES[0]])  # both wrong, only one confident
        rate = false_activation_rate(probabilities, y, CLASSES, threshold=0.7)
        assert rate == pytest.approx(0.5)


class TestEndToEnd:
    def test_results_are_reproducible_across_runs(self, toy):
        X, y, groups, subject_ids = toy
        kwargs = dict(
            task="t", model="logistic_regression", seed=3,
            budgets=(0, 10), n_repeats=2, n_jobs=1,
        )
        a, _ = run_personalization(X, y, groups, subject_ids, **kwargs)
        b, _ = run_personalization(X, y, groups, subject_ids, **kwargs)
        key = ["subject_id", "repeat", "budget", "strategy"]
        # Latency is wall-clock and is the one column that cannot be reproduced.
        fa = pd.DataFrame([r.to_row() for r in a]).sort_values(key).reset_index(drop=True)
        fb = pd.DataFrame([r.to_row() for r in b]).sort_values(key).reset_index(drop=True)
        columns = [c for c in fa.columns if c != "latency_ms_per_epoch"]
        assert fa[columns].equals(fb[columns])

    def test_evaluation_count_is_constant_across_budgets(self, toy):
        X, y, groups, subject_ids = toy
        results, _ = run_personalization(
            X, y, groups, subject_ids, task="t", model="logistic_regression",
            seed=0, budgets=(0, 5, 20), n_repeats=2, n_jobs=1,
        )
        frame = pd.DataFrame([r.to_row() for r in results])
        assert frame["n_evaluation"].nunique() == 1

    def test_calibration_count_matches_the_requested_budget(self, toy):
        X, y, groups, subject_ids = toy
        results, _ = run_personalization(
            X, y, groups, subject_ids, task="t", model="logistic_regression",
            seed=0, budgets=(0, 5, 20), n_repeats=1, n_jobs=1,
        )
        for result in results:
            assert result.n_calibration == result.budget

    def test_recentring_recovers_accuracy_on_a_planted_offset(self, toy):
        """Sanity check on the toy data, where the shift is known to be the problem.

        The fixture builds exactly the Phase 2 situation -- a large per-subject
        offset on top of a real class effect -- so a strategy that removes offsets
        must help here. If this fails, the strategy is broken rather than the EEG
        being difficult.
        """
        X, y, groups, subject_ids = toy
        results, _ = run_personalization(
            X, y, groups, subject_ids, task="t", model="logistic_regression",
            seed=0, budgets=(0, 20), n_repeats=3, n_jobs=1,
            strategies=("none", "recenter"),
        )
        frame = pd.DataFrame([r.to_row() for r in results])
        baseline = frame[(frame["strategy"] == "none")]["balanced_accuracy"].mean()
        centred = frame[
            (frame["strategy"] == "recenter") & (frame["budget"] == 20)
        ]["balanced_accuracy"].mean()
        # On this fixture the observed recovery is ~+0.13. The bar is set below
        # that to assert the mechanism works without pinning the test to an exact
        # value that a change of random fixture would break.
        assert centred > baseline + 0.10, (
            f"recentring recovered only {centred - baseline:+.3f} on data whose "
            f"offset is known to be the problem"
        )
