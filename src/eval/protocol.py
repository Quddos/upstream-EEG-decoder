"""Phase 2B/2C -- leave-one-subject-out evaluation.

The only splitting protocol this project uses is by **subject**. Epochs within a
run are temporally adjacent and highly correlated, so a random epoch split puts
near-duplicates on both sides of the boundary and inflates accuracy by tens of
percent. Worse for this project's purpose, it answers the wrong question: the
Mirror World controller has to work on a person it has never seen, and only a
held-out *subject* estimates that.

Statistical reality check
-------------------------
Five subjects give five folds, and a binary fold holds **45 test epochs**. At that
size a fold needs >= 29/45 (64.4%) to beat chance at p<0.05 one-sided, and a fold
scoring 60% has a 95% Wilson interval of roughly [45.5, 73.0] -- which includes
chance. Individual per-subject accuracies therefore cannot carry an argument on
their own. Two things are provided so the report does not have to pretend
otherwise: a Wilson interval on every proportion, and a label-permutation null
that re-runs the entire protocol with shuffled labels to give an empirical
p-value rather than an assumed one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut

from src.eval.baselines import MODELS, decision_scores


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Wilson rather than the normal approximation: at n=45 with p near 0.5 the
    normal interval is already poor, and near 0 or 1 it produces bounds outside
    [0, 1] entirely.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((centre - half) / denominator), float((centre + half) / denominator))


def macro_ovr_auc(y_true: np.ndarray, scores: np.ndarray, classes: Sequence) -> float:
    """Macro one-vs-rest ROC-AUC that works with any score, not just probabilities.

    ``sklearn.roc_auc_score(..., multi_class="ovr")`` insists on probability
    estimates that sum to one, which rules out a decision function. AUC only needs
    a ranking, so the one-vs-rest curves are computed per class here and averaged.
    Classes absent from ``y_true`` are skipped -- an AUC is undefined without both
    positives and negatives.
    """
    scores = np.asarray(scores)
    if scores.ndim == 1:
        # Binary: a single score column, positive class is classes[1].
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score((y_true == classes[1]).astype(int), scores))

    aucs = []
    for i, cls in enumerate(classes):
        binary = (y_true == cls).astype(int)
        if binary.min() == binary.max():
            continue
        aucs.append(roc_auc_score(binary, scores[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


@dataclass
class FoldResult:
    """One held-out subject, one model, one condition."""

    task: str
    channel_set: str
    feature_family: str
    model: str
    held_out_subject: str
    n_train: int
    n_test: int
    n_train_subjects: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    roc_auc: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion: List[List[int]] = field(default_factory=list)
    classes: Tuple[str, ...] = ()

    def to_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "task": self.task,
            "channel_set": self.channel_set,
            "feature_family": self.feature_family,
            "model": self.model,
            "held_out_subject": self.held_out_subject,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_train_subjects": self.n_train_subjects,
            "accuracy": self.accuracy,
            "accuracy_ci_low": self.accuracy_ci_low,
            "accuracy_ci_high": self.accuracy_ci_high,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "roc_auc": self.roc_auc,
        }
        for cls, scores in self.per_class.items():
            row[f"precision_{cls}"] = scores["precision"]
            row[f"recall_{cls}"] = scores["recall"]
            row[f"support_{cls}"] = scores["support"]
        # The full matrix is stored, not just per-class recall, so a pooled
        # confusion matrix can be summed back exactly. Reconstructing one from
        # recall alone would require guessing how the off-diagonal mass was
        # distributed, which for a three-class problem is not recoverable.
        row["classes"] = json.dumps(list(self.classes))
        row["confusion"] = json.dumps(self.confusion)
        return row


def losocv_folds(groups: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Leave-one-subject-out splits, in ascending subject order."""
    splitter = LeaveOneGroupOut()
    dummy = np.zeros(len(groups))
    return list(splitter.split(dummy, dummy, groups=groups))


def run_losocv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    subject_ids: np.ndarray,
    *,
    task: str,
    channel_set: str,
    feature_family: str,
    models: Sequence[str],
    seed: int,
) -> List[FoldResult]:
    """Evaluate each model with leave-one-subject-out cross-validation.

    The held-out subject's rows are used for exactly one thing: calling
    ``predict`` on an already-fitted pipeline. They never touch ``fit``, and the
    scaler inside the pipeline sees only training rows.
    """
    results: List[FoldResult] = []
    classes = tuple(sorted(np.unique(y)))

    for train_idx, test_idx in losocv_folds(groups):
        held_out = str(np.unique(subject_ids[test_idx])[0])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name in models:
            estimator = MODELS[name](seed)
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)

            try:
                auc = macro_ovr_auc(y_test, decision_scores(estimator, X_test), classes)
            except Exception:  # noqa: BLE001 - AUC is optional, never fatal
                auc = float("nan")

            precision, recall, _, support = precision_recall_fscore_support(
                y_test, y_pred, labels=list(classes), zero_division=0
            )
            correct = int((y_pred == y_test).sum())
            low, high = wilson_interval(correct, len(y_test))

            results.append(
                FoldResult(
                    task=task,
                    channel_set=channel_set,
                    feature_family=feature_family,
                    model=name,
                    held_out_subject=held_out,
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    n_train_subjects=int(len(np.unique(groups[train_idx]))),
                    accuracy=float(accuracy_score(y_test, y_pred)),
                    balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
                    macro_f1=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                    roc_auc=auc,
                    accuracy_ci_low=low,
                    accuracy_ci_high=high,
                    per_class={
                        str(cls): {
                            "precision": float(precision[i]),
                            "recall": float(recall[i]),
                            "support": int(support[i]),
                        }
                        for i, cls in enumerate(classes)
                    },
                    confusion=confusion_matrix(
                        y_test, y_pred, labels=list(classes)
                    ).tolist(),
                    classes=tuple(str(c) for c in classes),
                )
            )

    return results


def _single_threaded(estimator):
    """Force an estimator to one thread, leaving its results unchanged.

    Used only inside the permutation loop, where parallelism belongs at the
    permutation level; nested thread pools would oversubscribe the machine and
    run slower than either level alone.
    """
    model = estimator.named_steps.get("model")
    # Only the forest actually parallelises its fit. LogisticRegression still
    # accepts n_jobs but has ignored it since scikit-learn 1.8 and warns when it
    # is passed, so setting it there would add noise and change nothing.
    if isinstance(model, RandomForestClassifier):
        estimator.set_params(model__n_jobs=1)
    return estimator


def permutation_null(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    subject_ids: np.ndarray,
    *,
    model: str,
    seed: int,
    n_permutations: int = 200,
    n_jobs: int = -1,
) -> Tuple[float, np.ndarray, float]:
    """Empirical null for pooled LOSOCV accuracy under permuted labels.

    Labels are shuffled **within each subject**, which preserves both the subject
    structure and each subject's class proportions, so the null reflects "this
    protocol on this data with no relationship between features and label" rather
    than "no structure of any kind".

    Returns ``(observed_accuracy, null_accuracies, p_value)``. The p-value uses
    the ``(hits + 1) / (n + 1)`` convention, since the observed labelling is
    itself one of the possible permutations.
    """
    folds = losocv_folds(groups)

    def pooled_accuracy(labels: np.ndarray) -> float:
        correct = total = 0
        for train_idx, test_idx in folds:
            estimator = _single_threaded(MODELS[model](seed))
            estimator.fit(X[train_idx], labels[train_idx])
            correct += int((estimator.predict(X[test_idx]) == labels[test_idx]).sum())
            total += len(test_idx)
        return correct / total

    observed = pooled_accuracy(y)

    # Every shuffled labelling is drawn up front, in the same RNG order the
    # sequential version used, so the null is identical whether or not the loop
    # below is parallelised.
    rng = np.random.default_rng(seed)
    shuffles = []
    for _ in range(n_permutations):
        shuffled = y.copy()
        for subject in np.unique(groups):
            mask = groups == subject
            values = shuffled[mask]
            rng.shuffle(values)
            shuffled[mask] = values
        shuffles.append(shuffled)

    # Parallelise across permutations rather than inside the estimator. A random
    # forest on 180x96 takes ~0.7 s to fit however many threads it is given -- the
    # work per fit is too small for intra-model parallelism to help -- so 1000
    # sequential fits cost ~12 minutes. Spreading independent permutations over
    # cores instead turns that into well under a minute. Model behaviour is
    # untouched: n_jobs affects only scheduling, and a fixed random_state makes
    # the fitted forest identical either way.
    null = np.asarray(
        Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(pooled_accuracy)(labels) for labels in shuffles
        ),
        dtype=float,
    )

    p_value = float((int((null >= observed).sum()) + 1) / (n_permutations + 1))
    return observed, null, p_value


def run_within_subject_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    subject_ids: np.ndarray,
    runs: np.ndarray,
    *,
    task: str,
    channel_set: str,
    models: Sequence[str],
    seed: int,
) -> List[FoldResult]:
    """Leave-one-*run*-out cross-validation, within each subject separately.

    This is a **diagnostic, not a generalisation estimate**, and the report must
    not present it as one: the model is trained and tested on the same person, so
    it says nothing about a new user.

    It exists because a chance-level LOSOCV result is ambiguous on its own. It can
    mean the features carry no motor-imagery information at all, or it can mean
    they carry information that does not transfer between people. Those imply
    completely different Phase 3 designs -- the second is fixable by
    calibration, the first is not -- and only a within-subject reference
    separates them.

    Splitting is by run (R04/R08/R12), never randomly by epoch, so temporally
    adjacent epochs from the same run stay on the same side of the boundary.
    """
    results: List[FoldResult] = []
    classes = tuple(sorted(np.unique(y)))

    for subject in np.unique(groups):
        subject_mask = groups == subject
        held_subject = str(np.unique(subject_ids[subject_mask])[0])
        subject_runs = np.unique(runs[subject_mask])

        for held_run in subject_runs:
            train_mask = subject_mask & (runs != held_run)
            test_mask = subject_mask & (runs == held_run)
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            if len(np.unique(y[train_mask])) < 2:
                continue

            for name in models:
                estimator = MODELS[name](seed)
                estimator.fit(X[train_mask], y[train_mask])
                y_pred = estimator.predict(X[test_mask])
                y_test = y[test_mask]

                try:
                    auc = macro_ovr_auc(
                        y_test, decision_scores(estimator, X[test_mask]), classes
                    )
                except Exception:  # noqa: BLE001
                    auc = float("nan")

                precision, recall, _, support = precision_recall_fscore_support(
                    y_test, y_pred, labels=list(classes), zero_division=0
                )
                correct = int((y_pred == y_test).sum())
                low, high = wilson_interval(correct, len(y_test))

                results.append(
                    FoldResult(
                        task=task,
                        channel_set=channel_set,
                        feature_family="all",
                        model=name,
                        held_out_subject=f"{held_subject}/R{int(held_run):02d}",
                        n_train=int(train_mask.sum()),
                        n_test=int(test_mask.sum()),
                        n_train_subjects=1,
                        accuracy=float(accuracy_score(y_test, y_pred)),
                        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
                        macro_f1=float(
                            f1_score(y_test, y_pred, average="macro", zero_division=0)
                        ),
                        roc_auc=auc,
                        accuracy_ci_low=low,
                        accuracy_ci_high=high,
                        per_class={
                            str(cls): {
                                "precision": float(precision[i]),
                                "recall": float(recall[i]),
                                "support": int(support[i]),
                            }
                            for i, cls in enumerate(classes)
                        },
                        confusion=confusion_matrix(
                            y_test, y_pred, labels=list(classes)
                        ).tolist(),
                        classes=tuple(str(c) for c in classes),
                    )
                )

    return results


def summarise(results: Sequence[FoldResult]) -> pd.DataFrame:
    """Mean, SD, min and max across folds for each condition.

    The SD is across five subjects. With n=5 it is an unstable estimate and is
    reported as a description of the spread actually observed, not as an inference
    about the population of subjects.
    """
    frame = pd.DataFrame([r.to_row() for r in results])
    if frame.empty:
        return frame

    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]
    grouped = frame.groupby(["task", "channel_set", "feature_family", "model"])
    summary = grouped[metrics].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["worst_subject"] = grouped.apply(
        lambda g: g.loc[g["accuracy"].idxmin(), "held_out_subject"], include_groups=False
    )
    summary["n_folds"] = grouped.size()
    return summary.reset_index()
