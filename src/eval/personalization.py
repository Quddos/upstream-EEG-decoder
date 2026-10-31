"""Phase 3 -- the calibration-budget protocol.

For each held-out subject the population model is trained on the other four
subjects and never sees the held-out one. The held-out subject's own epochs are
then partitioned into a **calibration pool** and an **evaluation set**, and
increasing budgets are drawn from the pool only.

Three properties of the design, each chosen to remove a specific way of fooling
oneself:

**The evaluation set is fixed across budgets.** If budget 20 were evaluated on
"whatever epochs are left" it would be tested on 25 epochs while budget 0 was
tested on 45, and the curve would confound personalisation with test-set size and
composition. Here the evaluation set is carved out first and is identical for
every budget and every strategy within a repeat, so budget-to-budget differences
are *paired* -- the same epochs, the same subject, only the calibration changes.
That is both fairer and far more sensitive than an unpaired comparison, which
matters when a fold holds 25 epochs.

**Budgets are nested.** The 5-epoch budget is a subset of the 10-epoch budget, and
so on. Without nesting, an increase from 5 to 10 would mix "more data" with "a
different draw", and at these sizes the draw is the larger effect.

**Repeats use deterministic seeds.** A single partition of 45 epochs is a coin
toss; the reported numbers are means over repeats, with the spread across repeats
reported rather than hidden.

Calibration epochs are never evaluated on, and evaluation epochs are never used to
fit anything -- not a model, not a scaler, not a threshold, not a calibrator.
:mod:`tests.test_phase3_leakage` asserts this, including against deliberately
leaky controls.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.eval.baselines import MODELS
from src.eval.calibration import (
    STRATEGIES,
    brier_score,
    expected_calibration_error,
    false_activation_rate,
    safe_roc_auc,
)

#: Calibration budgets, in labelled epochs drawn from the held-out subject.
DEFAULT_BUDGETS: Tuple[int, ...] = (0, 5, 10, 15, 20)

#: Confidence threshold at which a "false activation" is counted. Fixed in
#: advance; tuning it on evaluation epochs is exactly the leak this phase forbids.
ACTIVATION_THRESHOLD: float = 0.7


@dataclass
class Split:
    """One deterministic partition of a held-out subject's epochs."""

    subject_id: str
    repeat: int
    calibration_pool: np.ndarray   # indices, ordered; budgets take prefixes
    evaluation: np.ndarray         # indices, fixed across every budget

    def budget_indices(self, budget: int) -> np.ndarray:
        if budget > len(self.calibration_pool):
            raise ValueError(
                f"budget {budget} exceeds pool of {len(self.calibration_pool)}"
            )
        return self.calibration_pool[:budget]


def make_split(
    y: np.ndarray,
    *,
    subject_id: str,
    repeat: int,
    pool_size: int,
    seed: int,
) -> Split:
    """Partition one subject's epochs into a calibration pool and an eval set.

    The pool is drawn **stratified by class** and then ordered so that any prefix
    is also roughly balanced. A 5-epoch budget that happened to contain one class
    only would make the supervised strategies undefined, and would say more about
    the draw than about the budget.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = {c: rng.permutation(np.flatnonzero(y == c)) for c in classes}

    # Round-robin across classes so every prefix stays close to balanced.
    pool: List[int] = []
    cursor = {c: 0 for c in classes}
    while len(pool) < pool_size:
        added = False
        for c in classes:
            if len(pool) >= pool_size:
                break
            if cursor[c] < len(per_class[c]):
                pool.append(int(per_class[c][cursor[c]]))
                cursor[c] += 1
                added = True
        if not added:
            break

    pool_array = np.asarray(pool, dtype=int)
    evaluation = np.setdiff1d(np.arange(len(y)), pool_array, assume_unique=False)
    return Split(
        subject_id=subject_id,
        repeat=repeat,
        calibration_pool=pool_array,
        evaluation=evaluation,
    )


@dataclass
class BudgetResult:
    """One (subject, repeat, strategy, budget) evaluation."""

    task: str
    model: str
    strategy: str
    subject_id: str
    repeat: int
    budget: int
    n_calibration: int
    n_evaluation: int
    n_parameters: int
    balanced_accuracy: float
    accuracy: float
    macro_f1: float
    roc_auc: float
    ece: float
    brier: float
    false_activation_rate: float
    latency_ms_per_epoch: float
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    confusion: List[List[int]] = field(default_factory=list)
    classes: Tuple[str, ...] = ()

    def to_row(self) -> Dict[str, object]:
        row = {
            "task": self.task,
            "model": self.model,
            "strategy": self.strategy,
            "subject_id": self.subject_id,
            "repeat": self.repeat,
            "budget": self.budget,
            "n_calibration": self.n_calibration,
            "n_evaluation": self.n_evaluation,
            "n_parameters": self.n_parameters,
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "roc_auc": self.roc_auc,
            "ece": self.ece,
            "brier": self.brier,
            "false_activation_rate": self.false_activation_rate,
            "latency_ms_per_epoch": self.latency_ms_per_epoch,
        }
        for cls, value in self.per_class_recall.items():
            row[f"recall_{cls}"] = value
        # Stored in full so pooled confusion matrices can be summed exactly rather
        # than reconstructed from per-class recall, which for three classes cannot
        # recover how the off-diagonal mass was distributed.
        row["classes"] = json.dumps(list(self.classes))
        row["confusion"] = json.dumps(self.confusion)
        return row


def _evaluate(
    calibrated,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    classes: Sequence[str],
    rest_label: Optional[str],
) -> Tuple[Dict[str, float], Dict[str, float], List[List[int]], float]:
    start = time.perf_counter()
    probabilities = calibrated.predict_proba(X_eval)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(len(X_eval), 1)

    predicted = np.asarray(calibrated.classes)[probabilities.argmax(axis=1)]
    precision, recall, _, _ = precision_recall_fscore_support(
        y_eval, predicted, labels=list(classes), zero_division=0
    )

    metrics = {
        "accuracy": float((predicted == y_eval).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, predicted)),
        "macro_f1": float(f1_score(y_eval, predicted, average="macro", zero_division=0)),
        "roc_auc": safe_roc_auc(probabilities, y_eval, calibrated.classes),
        "ece": expected_calibration_error(probabilities, y_eval, calibrated.classes),
        "brier": brier_score(probabilities, y_eval, calibrated.classes),
        "false_activation_rate": false_activation_rate(
            probabilities,
            y_eval,
            calibrated.classes,
            rest_label=rest_label,
            threshold=ACTIVATION_THRESHOLD,
        ),
    }
    per_class = {str(c): float(recall[i]) for i, c in enumerate(classes)}
    matrix = confusion_matrix(y_eval, predicted, labels=list(classes)).tolist()
    return metrics, per_class, matrix, elapsed_ms


def run_personalization(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    subject_ids: np.ndarray,
    *,
    task: str,
    model: str,
    seed: int,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    n_repeats: int = 20,
    strategies: Optional[Sequence[str]] = None,
    rest_label: Optional[str] = None,
    n_jobs: int = -1,
) -> Tuple[List[BudgetResult], List[Dict[str, object]]]:
    """Leave-one-subject-out, with calibration budgets drawn from the held-out subject.

    Returns ``(results, reliability_records)``; the second is per-epoch
    probability/outcome pairs used to build the reliability diagram without
    re-running anything.
    """
    strategies = list(strategies or STRATEGIES)
    classes = tuple(sorted(np.unique(y)))
    pool_size = max(budgets)

    def one_subject(subject) -> Tuple[List[BudgetResult], List[Dict[str, object]]]:
        results: List[BudgetResult] = []
        reliability: List[Dict[str, object]] = []
        held_out = str(np.unique(subject_ids[groups == subject])[0])
        train_mask = groups != subject
        test_mask = groups == subject

        # Population model: fitted once per held-out subject, on the other four.
        population = MODELS[model](seed)
        population.fit(X[train_mask], y[train_mask])
        X_pop, y_pop = X[train_mask], y[train_mask]

        X_subject, y_subject = X[test_mask], y[test_mask]

        for repeat in range(n_repeats):
            split = make_split(
                y_subject,
                subject_id=held_out,
                repeat=repeat,
                pool_size=pool_size,
                seed=seed + repeat * 1009,
            )
            X_eval = X_subject[split.evaluation]
            y_eval = y_subject[split.evaluation]

            for budget in budgets:
                picks = split.budget_indices(budget)
                X_cal, y_cal = X_subject[picks], y_subject[picks]

                for strategy in strategies:
                    calibrated = STRATEGIES[strategy](
                        population, X_cal, y_cal, X_pop, y_pop, classes, seed
                    )
                    metrics, per_class, matrix, latency = _evaluate(
                        calibrated, X_eval, y_eval, classes, rest_label
                    )

                    results.append(
                        BudgetResult(
                            task=task,
                            model=model,
                            strategy=strategy,
                            subject_id=held_out,
                            repeat=repeat,
                            budget=budget,
                            n_calibration=len(picks),
                            n_evaluation=len(y_eval),
                            n_parameters=calibrated.n_parameters,
                            per_class_recall=per_class,
                            confusion=matrix,
                            classes=tuple(str(c) for c in classes),
                            latency_ms_per_epoch=latency,
                            **metrics,
                        )
                    )

                    # Keep per-epoch confidences for the reliability diagram, for
                    # a small set of conditions only -- storing all of them would
                    # be millions of rows.
                    if repeat == 0 and budget in (0, max(budgets)):
                        probabilities = calibrated.predict_proba(X_eval)
                        predicted = np.asarray(calibrated.classes)[
                            probabilities.argmax(axis=1)
                        ]
                        for confidence, ok in zip(
                            probabilities.max(axis=1), predicted == y_eval
                        ):
                            reliability.append(
                                {
                                    "task": task,
                                    "model": model,
                                    "strategy": strategy,
                                    "budget": budget,
                                    "subject_id": held_out,
                                    "confidence": float(confidence),
                                    "correct": bool(ok),
                                }
                            )
        return results, reliability

    # Subjects are independent -- each fits its own population model on the other
    # four -- so this loop parallelises cleanly. Scheduling only: every seed is
    # derived from the subject and repeat index rather than from execution order,
    # so the results are identical to the sequential version.
    chunks = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(one_subject)(subject) for subject in np.unique(groups)
    )
    results = [item for chunk, _ in chunks for item in chunk]
    reliability = [item for _, chunk in chunks for item in chunk]
    return results, reliability


def summarise_budgets(results: Sequence[BudgetResult]) -> pd.DataFrame:
    """Mean and SD across repeats and subjects, per (task, model, strategy, budget)."""
    frame = pd.DataFrame([r.to_row() for r in results])
    if frame.empty:
        return frame
    metrics = [
        "balanced_accuracy", "accuracy", "macro_f1", "roc_auc",
        "ece", "brier", "false_activation_rate", "latency_ms_per_epoch",
    ]
    grouped = frame.groupby(["task", "model", "strategy", "budget"])
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary["n_runs"] = grouped.size()
    return summary.reset_index()


def per_subject_gain(
    results: Sequence[BudgetResult], *, strategy: str, budget: int, metric: str = "balanced_accuracy"
) -> pd.DataFrame:
    """Per-subject change from budget 0 (no personalisation) to a given budget.

    The baseline is the *same strategy family's* budget-0 result, which for every
    strategy is the untouched population model -- so the gain isolates the effect
    of the calibration epochs, not of the strategy's existence.
    """
    frame = pd.DataFrame([r.to_row() for r in results])
    if frame.empty:
        return frame

    baseline = (
        frame[(frame["budget"] == 0) & (frame["strategy"] == "none")]
        .groupby("subject_id")[metric]
        .mean()
    )
    personalised = (
        frame[(frame["budget"] == budget) & (frame["strategy"] == strategy)]
        .groupby("subject_id")[metric]
        .agg(["mean", "std"])
    )
    out = personalised.join(baseline.rename("baseline"))
    out["gain"] = out["mean"] - out["baseline"]
    return out.reset_index()
