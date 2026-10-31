"""Phase 3 -- subject-specific calibration strategies and confidence metrics.

Phase 2 established the problem this module exists to attack: the population
representation does not transfer. Between-subject spread of ``C4_log_mu`` is
10.4x the within-subject LEFT/RIGHT separation, and cross-subject LOSOCV sat at
51.1% against a 65.8% within-subject reference. The question here is whether a
*small* labelled sample from the new subject can close that gap.

The strategies form a deliberate ladder in cost, because "which strategy is most
efficient?" is a question about parameters-per-epoch, not only about accuracy:

============================  ==========  =========================================
strategy                      parameters  what it can fix
============================  ==========  =========================================
``none``                      0           nothing -- the Phase 2 baseline
``recenter``                  0 (labels)  a first-moment feature shift
``recenter_scale``            0 (labels)  first and second moments
``bias_shift``                1           a shifted decision boundary
``platt``                     2           boundary shift + probability scale
``refit_augmented``           all         anything, if the budget is large enough
``subject_only``              all         anything, using the budget alone
============================  ==========  =========================================

``recenter`` and ``recenter_scale`` need the calibration epochs' *features* but
not their *labels*, which matters for deployment: they could run off a short
unlabelled warm-up rather than a cued calibration block. They are the "simple
normalization baseline" the brief asks for, and they target the measured shift
directly.

Every strategy receives the calibration epochs only. None of them ever sees an
evaluation epoch -- :mod:`tests.test_phase3_leakage` asserts this against
deliberately leaky controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------------------
# Confidence and calibration metrics
# --------------------------------------------------------------------------------------


def expected_calibration_error(
    probabilities: np.ndarray, y_true: np.ndarray, classes: Sequence, n_bins: int = 10
) -> float:
    """Binned expected calibration error on the top-1 confidence.

    Confidence is ``max_k p_k``; an epoch is correct when ``argmax_k p_k`` matches
    the label. Within each equal-width confidence bin the gap between mean
    confidence and observed accuracy is measured, and the gaps are averaged
    weighted by bin population.

    A perfectly calibrated model that says "70% confident" is right 70% of the
    time. For a rehabilitation controller this matters more than accuracy alone:
    the confidence value is what a downstream policy would gate on, so a
    confidently-wrong model is worse than an accurate-but-humble one.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1 - probabilities, probabilities])
    if len(y_true) == 0:
        return float("nan")

    confidence = probabilities.max(axis=1)
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    correct = (predicted == np.asarray(y_true)).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Half-open bins, with the top bin closed so confidence == 1.0 is counted.
        in_bin = (confidence > lo) & (confidence <= hi) if hi < 1.0 else (
            (confidence > lo) & (confidence <= 1.0)
        )
        if not in_bin.any():
            continue
        weight = in_bin.mean()
        error += weight * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def reliability_bins(
    probabilities: np.ndarray, y_true: np.ndarray, classes: Sequence, n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """Per-bin confidence, accuracy and count, for a reliability diagram."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1 - probabilities, probabilities])

    confidence = probabilities.max(axis=1)
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    correct = (predicted == np.asarray(y_true)).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, accuracies, confidences, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi) if hi < 1.0 else (
            (confidence > lo) & (confidence <= 1.0)
        )
        centres.append((lo + hi) / 2)
        counts.append(int(in_bin.sum()))
        accuracies.append(float(correct[in_bin].mean()) if in_bin.any() else np.nan)
        confidences.append(float(confidence[in_bin].mean()) if in_bin.any() else np.nan)
    return {
        "bin_centre": np.asarray(centres),
        "accuracy": np.asarray(accuracies),
        "confidence": np.asarray(confidences),
        "count": np.asarray(counts),
    }


def brier_score(
    probabilities: np.ndarray, y_true: np.ndarray, classes: Sequence
) -> float:
    """Multiclass Brier score: mean squared error of the probability vector.

    Lower is better; 0 is perfect. Unlike accuracy it is a *proper* scoring rule,
    so it cannot be improved by over-confident guessing.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1 - probabilities, probabilities])
    if len(y_true) == 0:
        return float("nan")

    onehot = np.zeros_like(probabilities)
    class_index = {c: i for i, c in enumerate(classes)}
    for row, label in enumerate(y_true):
        onehot[row, class_index[label]] = 1.0
    return float(((probabilities - onehot) ** 2).sum(axis=1).mean())


def false_activation_rate(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    classes: Sequence,
    *,
    rest_label: Optional[str] = None,
    threshold: float = 0.7,
) -> float:
    """How often the system would confidently drive the wrong thing.

    Two definitions, because the two tasks mean different things by a false
    activation:

    * **Three-class** (``rest_label`` given): of the epochs where the user is
      actually at REST, the fraction the model assigns to a movement class with
      confidence at or above ``threshold``. This is the mirror world moving when
      the user is still -- the failure a rehabilitation user would notice most.
    * **Binary** (no rest class): of all epochs, the fraction where the model is
      confident at or above ``threshold`` *and* wrong. This is the mirror world
      confidently moving the wrong limb.

    Both are reported at a fixed threshold rather than optimised, because tuning
    the threshold on the evaluation epochs is exactly the leak Phase 3 forbids.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1 - probabilities, probabilities])
    if len(y_true) == 0:
        return float("nan")

    classes = list(classes)
    confidence = probabilities.max(axis=1)
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    y_true = np.asarray(y_true)

    if rest_label is not None and rest_label in classes:
        at_rest = y_true == rest_label
        if not at_rest.any():
            return float("nan")
        activated = (predicted != rest_label) & (confidence >= threshold)
        return float(activated[at_rest].mean())

    return float(((predicted != y_true) & (confidence >= threshold)).mean())


def safe_roc_auc(
    probabilities: np.ndarray, y_true: np.ndarray, classes: Sequence
) -> float:
    """Macro one-vs-rest ROC-AUC, NaN when a class is absent from ``y_true``."""
    probabilities = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true)
    classes = list(classes)

    if probabilities.ndim == 1 or probabilities.shape[1] == 2:
        scores = probabilities if probabilities.ndim == 1 else probabilities[:, 1]
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score((y_true == classes[1]).astype(int), scores))

    aucs = []
    for i, cls in enumerate(classes):
        binary = (y_true == cls).astype(int)
        if binary.min() == binary.max():
            continue
        aucs.append(roc_auc_score(binary, probabilities[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


# --------------------------------------------------------------------------------------
# Calibration strategies
# --------------------------------------------------------------------------------------


@dataclass
class CalibratedModel:
    """A population model plus whatever personalisation was fitted on top."""

    predict_proba: Callable[[np.ndarray], np.ndarray]
    classes: Tuple
    n_parameters: int
    strategy: str

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.asarray(self.classes)[probabilities.argmax(axis=1)]


def _population_proba(pipeline, X: np.ndarray) -> np.ndarray:
    return pipeline.predict_proba(X)


def strategy_none(pipeline, X_cal, y_cal, X_pop, y_pop, classes, seed) -> CalibratedModel:
    """No personalisation: the Phase 2 baseline, carried forward unchanged."""
    return CalibratedModel(
        predict_proba=lambda X: _population_proba(pipeline, X),
        classes=tuple(classes),
        n_parameters=0,
        strategy="none",
    )


def _affine_wrapper(
    pipeline,
    centre: np.ndarray,
    shift: np.ndarray,
    scale: Optional[np.ndarray] = None,
):
    """Map subject-space features into population space, then predict.

    ``x -> (x - centre) * scale + shift``.  ``centre`` is the subject's own mean
    (estimated from calibration epochs only) and ``shift`` is the population mean,
    so the transform removes the subject's offset and puts the point back where
    the population model expects it.  Forgetting ``shift`` would centre the
    subject on zero rather than on the population, which is a different and wrong
    transform -- the population scaler inside the pipeline is not centred there.
    """

    def predict_proba(X: np.ndarray) -> np.ndarray:
        adjusted = X - centre
        if scale is not None:
            adjusted = adjusted * scale
        return pipeline.predict_proba(adjusted + shift)

    return predict_proba


def strategy_recenter(pipeline, X_cal, y_cal, X_pop, y_pop, classes, seed) -> CalibratedModel:
    """Align the subject's feature mean to the population's, first moment only.

    ``x -> x - (mean_subject - mean_population)``. Uses the calibration epochs'
    features but **not their labels**, so in deployment this could be estimated
    from a short unlabelled warm-up rather than a cued block.

    This is the direct attack on the Phase 2 diagnosis: if the failure is a
    per-subject offset rather than a missing signal, removing the offset should
    recover accuracy, and the size of the recovery measures how much of the
    problem was offset.
    """
    if len(X_cal) == 0:
        return strategy_none(pipeline, X_cal, y_cal, X_pop, y_pop, classes, seed)
    return CalibratedModel(
        predict_proba=_affine_wrapper(
            pipeline, centre=X_cal.mean(axis=0), shift=X_pop.mean(axis=0)
        ),
        classes=tuple(classes),
        n_parameters=0,
        strategy="recenter",
    )


def strategy_recenter_scale(
    pipeline, X_cal, y_cal, X_pop, y_pop, classes, seed
) -> CalibratedModel:
    """Align both the mean and the per-feature spread (first and second moments).

    Also label-free. Needs more calibration epochs than :func:`strategy_recenter`
    to estimate a stable variance, which is precisely what the budget curve is
    there to reveal.
    """
    if len(X_cal) < 2:
        return strategy_none(pipeline, X_cal, y_cal, X_pop, y_pop, classes, seed)
    subject_std = X_cal.std(axis=0)
    population_std = X_pop.std(axis=0)
    # A near-constant feature in a 5-epoch calibration set would otherwise produce
    # an enormous multiplier, so the ratio is clipped. The clip is fixed in
    # advance, not tuned against any evaluation result.
    ratio = np.where(
        subject_std > 1e-12, population_std / np.maximum(subject_std, 1e-12), 1.0
    )
    ratio = np.clip(ratio, 0.2, 5.0)
    return CalibratedModel(
        predict_proba=_affine_wrapper(
            pipeline,
            centre=X_cal.mean(axis=0),
            shift=X_pop.mean(axis=0),
            scale=ratio,
        ),
        classes=tuple(classes),
        n_parameters=0,
        strategy="recenter_scale",
    )


def _decision_scores(pipeline, X: np.ndarray, classes) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    if proba.shape[1] == 2:
        # Log-odds: the quantity a bias shift acts on linearly.
        p = np.clip(proba[:, 1], 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))
    return np.log(np.clip(proba, 1e-9, 1.0))


def strategy_bias_shift(
    pipeline, X_cal, y_cal, X_pop, y_pop, classes, seed
) -> CalibratedModel:
    """Refit only the decision threshold, keeping every population weight.

    One free parameter for binary (a shift in log-odds), one per class for
    multiclass. This is the cheapest *supervised* personalisation available and
    the one most likely to be learnable from five epochs: it assumes the
    population model ranks the subject's epochs correctly but places the cut in
    the wrong place, which is exactly what a feature offset does to a linear
    boundary.
    """
    classes = list(classes)
    if len(X_cal) == 0 or len(np.unique(y_cal)) < 2:
        return strategy_none(pipeline, X_cal, y_cal, X_pop, y_pop, classes, seed)

    if len(classes) == 2:
        scores = _decision_scores(pipeline, X_cal, classes)
        positive = (np.asarray(y_cal) == classes[1]).astype(int)
        # Threshold that maximises calibration-set accuracy, taken as the midpoint
        # between adjacent candidate scores so it does not sit exactly on a point.
        order = np.argsort(scores)
        candidates = np.concatenate(
            [[scores[order[0]] - 1.0], (scores[order[:-1]] + scores[order[1:]]) / 2,
             [scores[order[-1]] + 1.0]]
        )
        best, best_threshold = -1.0, 0.0
        for threshold in candidates:
            accuracy = ((scores > threshold).astype(int) == positive).mean()
            if accuracy > best:
                best, best_threshold = accuracy, threshold

        def predict_proba(X: np.ndarray) -> np.ndarray:
            shifted = _decision_scores(pipeline, X, classes) - best_threshold
            p1 = 1.0 / (1.0 + np.exp(-shifted))
            return np.column_stack([1 - p1, p1])

        return CalibratedModel(
            predict_proba=predict_proba,
            classes=tuple(classes),
            n_parameters=1,
            strategy="bias_shift",
        )

    # Multiclass: one additive log-probability offset per class, set so the
    # predicted class distribution on the calibration set matches its labels.
    proba = pipeline.predict_proba(X_cal)
    observed = np.array([(np.asarray(y_cal) == c).mean() for c in classes])
    predicted = proba.mean(axis=0)
    offsets = np.log(np.clip(observed, 1e-6, None)) - np.log(np.clip(predicted, 1e-6, None))

    def predict_proba(X: np.ndarray) -> np.ndarray:
        adjusted = np.log(np.clip(pipeline.predict_proba(X), 1e-12, None)) + offsets
        adjusted = np.exp(adjusted - adjusted.max(axis=1, keepdims=True))
        return adjusted / adjusted.sum(axis=1, keepdims=True)

    return CalibratedModel(
        predict_proba=predict_proba,
        classes=tuple(classes),
        n_parameters=len(classes),
        strategy="bias_shift",
    )


def strategy_platt(pipeline, X_cal, y_cal, X_pop, y_pop, classes, seed) -> CalibratedModel:
    """Fit a one-dimensional logistic on the population scores (scale + bias).

    Two parameters. Unlike :func:`strategy_bias_shift` it can also *sharpen or
    soften* the probabilities, so it is the strategy aimed at the confidence
    metrics (ECE, Brier) rather than at accuracy alone.
    """
    classes = list(classes)
    if len(X_cal) == 0 or len(np.unique(y_cal)) < 2 or len(classes) != 2:
        return strategy_bias_shift(pipeline, X_cal, y_cal, X_pop, y_pop, classes, seed)

    scores = _decision_scores(pipeline, X_cal, classes).reshape(-1, 1)
    target = (np.asarray(y_cal) == classes[1]).astype(int)
    calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    calibrator.fit(scores, target)

    def predict_proba(X: np.ndarray) -> np.ndarray:
        s = _decision_scores(pipeline, X, classes).reshape(-1, 1)
        p1 = calibrator.predict_proba(s)[:, 1]
        return np.column_stack([1 - p1, p1])

    return CalibratedModel(
        predict_proba=predict_proba,
        classes=tuple(classes),
        n_parameters=2,
        strategy="platt",
    )


def strategy_refit_augmented(
    pipeline, X_cal, y_cal, X_pop, y_pop, classes, seed, weight: float = 5.0
) -> CalibratedModel:
    """Retrain on the population plus the calibration epochs, upweighted.

    The most expensive option: every parameter is refitted. ``weight`` gives each
    calibration epoch the influence of five population epochs, so a 20-epoch
    budget carries roughly the weight of 100 of the 180 population epochs. The
    weight is fixed, not tuned -- tuning it would require a validation set from
    the held-out subject, which is the leak this phase forbids.
    """
    classes = list(classes)
    if len(X_cal) == 0 or len(np.unique(y_cal)) < 2:
        return strategy_none(pipeline, X_cal, y_cal, X_pop, y_pop, classes, seed)

    X = np.vstack([X_pop, X_cal])
    y = np.concatenate([np.asarray(y_pop), np.asarray(y_cal)])
    weights = np.concatenate([np.ones(len(X_pop)), np.full(len(X_cal), weight)])

    refitted = clone(pipeline)
    try:
        refitted.fit(X, y, model__sample_weight=weights)
    except TypeError:  # estimator does not accept sample weights
        refitted.fit(X, y)

    return CalibratedModel(
        predict_proba=lambda Xq: refitted.predict_proba(Xq),
        classes=tuple(refitted.named_steps["model"].classes_),
        n_parameters=int(X.shape[1]),
        strategy="refit_augmented",
    )


def strategy_subject_only(
    pipeline, X_cal, y_cal, X_pop, y_pop, classes, seed
) -> CalibratedModel:
    """Train on the calibration epochs alone, ignoring the population entirely.

    The control that says whether the population model contributes anything at
    all. If a 20-epoch personal model matches the personalised population model,
    then the four training subjects were not helping.
    """
    classes = list(classes)
    if len(X_cal) == 0 or len(np.unique(y_cal)) < 2:
        return strategy_none(pipeline, X_cal, y_cal, X_pop, y_pop, classes, seed)

    personal = clone(pipeline)
    personal.fit(X_cal, y_cal)
    return CalibratedModel(
        predict_proba=lambda Xq: personal.predict_proba(Xq),
        classes=tuple(personal.named_steps["model"].classes_),
        n_parameters=int(X_cal.shape[1]),
        strategy="subject_only",
    )


#: Strategies evaluated in Phase 3, in ladder order (cheapest personalisation first).
STRATEGIES: Dict[str, Callable] = {
    "none": strategy_none,
    "recenter": strategy_recenter,
    "recenter_scale": strategy_recenter_scale,
    "bias_shift": strategy_bias_shift,
    "platt": strategy_platt,
    "refit_augmented": strategy_refit_augmented,
    "subject_only": strategy_subject_only,
}

#: Strategies that need no labels -- they use calibration features only.
LABEL_FREE_STRATEGIES: Tuple[str, ...] = ("none", "recenter", "recenter_scale")
