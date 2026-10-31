"""Phase 2B -- the baseline classifiers, with fixed hyperparameters.

**Nothing here is tuned.** Every hyperparameter is fixed at the value written
below and is never selected from data. That is a deliberate choice, not an
oversight: with five subjects, hyperparameter selection inside a
leave-one-subject-out loop would be driven by three or four training subjects per
inner fold, which is far too noisy to mean anything, and selecting on the outer
fold would be the exact "tuning on test subjects" the brief prohibits. The cost
is that these are honest *baselines* rather than best-achievable numbers, and the
report says so.

Each model is wrapped in a :class:`~sklearn.pipeline.Pipeline` whose first step is
a :class:`~sklearn.preprocessing.StandardScaler`. This is what keeps scaling
inside the fold: ``Pipeline.fit`` fits the scaler on the training subjects only,
and ``Pipeline.predict`` applies those stored statistics to the held-out subject.
Standardising the whole matrix before splitting would leak the test subject's
mean and variance into training, and :mod:`tests.test_leakage` asserts it does not
happen -- including a deliberately leaky control that must be caught.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#: Factory signature: ``(seed) -> fitted-able estimator``.
ModelFactory = Callable[[int], Pipeline]


def logistic_regression(seed: int) -> Pipeline:
    """L2 logistic regression.

    ``C=1.0`` is scikit-learn's default and is left there. With up to 268
    correlated band-power features against 180 training epochs, the regularisation
    is doing real work -- an unpenalised fit would be underdetermined.

    ``penalty`` is not passed: it is L2 by default and naming it explicitly is
    deprecated from scikit-learn 1.8. The behaviour is unchanged.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def linear_svm(seed: int) -> Pipeline:
    """Linear SVM.

    ``LinearSVC`` rather than ``SVC(kernel="linear")``: it scales better and, more
    importantly, exposes ``decision_function`` without Platt scaling. Fitting
    probability calibration would require an internal refit on the training fold,
    which is defensible but adds a moving part this baseline does not need -- the
    ROC-AUC here is computed from the raw decision function, which is
    rank-equivalent and therefore gives an identical AUC.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LinearSVC(C=1.0, dual="auto", max_iter=20000, random_state=seed),
            ),
        ]
    )


def random_forest(seed: int) -> Pipeline:
    """Random forest.

    500 trees, unrestricted depth -- the scikit-learn defaults apart from the tree
    count, which is raised so the out-of-bag variance between reruns is small
    enough that a fold result is reproducible. Scaling is irrelevant to a tree
    ensemble but the pipeline keeps it for uniformity; it cannot change a split.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def majority_class(seed: int) -> Pipeline:
    """The floor: always predict the training set's most frequent class.

    Included because "above 50%" is not the right yardstick for a slightly
    unbalanced problem -- this makes the actual floor explicit in every table.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", DummyClassifier(strategy="most_frequent", random_state=seed)),
        ]
    )


#: The models evaluated in Phase 2, in report order.
MODELS: Mapping[str, ModelFactory] = {
    "majority_class": majority_class,
    "logistic_regression": logistic_regression,
    "linear_svm": linear_svm,
    "random_forest": random_forest,
}

#: Models that are baselines-of-record rather than the trivial floor.
REAL_MODELS = tuple(name for name in MODELS if name != "majority_class")


def decision_scores(estimator: Pipeline, X: np.ndarray) -> np.ndarray:
    """Continuous scores for ROC-AUC, whatever the estimator exposes.

    Returns ``(n_samples,)`` for a binary problem and ``(n_samples, n_classes)``
    for a multiclass one. ``predict_proba`` is preferred where available;
    ``decision_function`` is used otherwise and is rank-equivalent, which is all
    an AUC needs.
    """
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba
    scores = estimator.decision_function(X)
    return scores
