"""Phase 4B -- CSP and Riemannian tangent-space representations.

Phase 4A left one question open: is the ~0.60 cross-subject ceiling a property of
the *handcrafted band-power features*, or of the EEG itself? Answering it needs
representations that are learned from the signal rather than specified in advance,
and the two standard ones in motor imagery are CSP and covariance/tangent-space.

Everything here is a :class:`~sklearn.pipeline.Pipeline`. That is a deliberate
safety property, not a style choice: a pipeline's ``fit`` sees only the rows it is
given, so putting the spatial filter, the covariance reference and the tangent-space
projection *inside* the pipeline makes it structurally impossible for any of them to
see the held-out subject. ``fit(X_train)`` then ``predict(X_test)`` is the whole
guarantee, and :mod:`tests.test_phase4b_leakage` attacks it with deliberately
global variants that must be caught.

Configuration, fixed in advance
-------------------------------
The brief requires the choices to be recorded before any LOSOCV result is seen, so:

``CSP_N_COMPONENTS = 6``
    Three filter pairs. Standard motor-imagery baselines use four to eight; six sits
    mid-range and yields six features against ~1,750 training epochs, so the
    classifier is nowhere near underdetermined.
``CSP regularisation = "ledoit_wolf"``
    Shrinkage on the class covariances. Without it, covariance estimates from a
    21-channel montage over pooled subjects are noisy enough that the generalised
    eigendecomposition becomes unstable.
``log-variance output``
    ``transform_into="average_power"`` with ``log=True`` -- the conventional CSP
    feature, and the one whose distribution suits a linear classifier.
``Riemannian covariance estimator = "oas"``
    Oracle-approximating shrinkage. Same reasoning as above; a plain sample
    covariance is frequently near-singular at 64 channels and the matrix logarithm
    then blows up.
``TangentSpace(metric="riemann")``
    Affine-invariant metric, the standard choice. The reference point is the
    Riemannian mean of the **training** covariances only.

No per-subject Riemannian recentring is applied. It is a common addition in the
literature and it would almost certainly help, but it estimates an alignment from
the held-out subject's own data, which this phase explicitly forbids.

Nothing here is tuned. These values were written down once and not revisited.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping

import numpy as np
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#: Number of CSP spatial components. Fixed in advance; see the module docstring.
CSP_N_COMPONENTS: int = 6

#: Covariance regularisation for CSP.
CSP_REG: str = "ledoit_wolf"

#: Covariance estimator for the Riemannian pipeline.
RIEMANN_COV_ESTIMATOR: str = "oas"

#: Metric defining the tangent-space reference point.
RIEMANN_METRIC: str = "riemann"


def _classifier(name: str, seed: int):
    """The three Phase 2 classifiers, with their frozen hyperparameters.

    Deliberately identical to :mod:`src.eval.baselines` so that a difference
    between conditions can only come from the representation. ``penalty`` is not
    passed to ``LogisticRegression`` -- L2 is the default and naming it is
    deprecated from scikit-learn 1.8.
    """
    if name == "logistic_regression":
        return LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=seed)
    if name == "linear_svm":
        return LinearSVC(C=1.0, dual="auto", max_iter=20000, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    raise ValueError(f"unknown classifier {name!r}")


def csp_pipeline(classifier: str, seed: int, n_components: int = CSP_N_COMPONENTS) -> Pipeline:
    """CSP log-variance features into a linear classifier.

    ``CSP`` is the first step, so its spatial filters are solved from the training
    fold's epochs and labels alone. Applying the pipeline to the held-out subject
    only *projects* through those filters.

    Note CSP is supervised -- it uses ``y`` -- which makes fold discipline more
    important here than for an unsupervised transform, not less.
    """
    return Pipeline(
        [
            (
                "csp",
                CSP(
                    n_components=n_components,
                    reg=CSP_REG,
                    log=True,
                    transform_into="average_power",
                    cov_est="concat",
                ),
            ),
            ("scale", StandardScaler()),
            ("model", _classifier(classifier, seed)),
        ]
    )


def riemann_pipeline(classifier: str, seed: int) -> Pipeline:
    """Covariance -> tangent space -> linear classifier.

    ``Covariances`` is per-epoch and stateless, so it cannot leak. ``TangentSpace``
    *is* stateful: it learns the Riemannian mean that defines the projection
    origin, and that mean is computed from the training fold only because the step
    lives inside the pipeline.
    """
    return Pipeline(
        [
            ("cov", Covariances(estimator=RIEMANN_COV_ESTIMATOR)),
            ("tangent", TangentSpace(metric=RIEMANN_METRIC)),
            ("scale", StandardScaler()),
            ("model", _classifier(classifier, seed)),
        ]
    )


def frozen_pipeline(classifier: str, seed: int) -> Pipeline:
    """Scaler + classifier, for the pre-computed Phase 2 feature matrix.

    Condition A. The representation is already fixed, so the only fold-local step
    is standardisation -- exactly what Phase 2 did.
    """
    return Pipeline([("scale", StandardScaler()), ("model", _classifier(classifier, seed))])


#: ``{representation: {classifier: factory}}``. ``frozen`` consumes a feature
#: matrix; ``csp`` and ``riemann`` consume raw epochs, so the caller must pass the
#: right array -- :mod:`src.cli.phase4b_compare` does that from one aligned load.
REPRESENTATIONS: Mapping[str, Callable[[str, int], Pipeline]] = {
    "frozen": frozen_pipeline,
    "csp": csp_pipeline,
    "riemann": riemann_pipeline,
}

#: Which representations take epoch time-series rather than a feature matrix.
EPOCH_INPUT_REPRESENTATIONS = ("csp", "riemann")

#: Primary classifiers for the Phase 4B comparison; random forest is secondary.
PRIMARY_CLASSIFIERS = ("logistic_regression", "linear_svm")


def config_record() -> Dict[str, object]:
    """The pre-registered configuration, for the provenance record."""
    return {
        "csp_n_components": CSP_N_COMPONENTS,
        "csp_reg": CSP_REG,
        "csp_transform": "average_power+log",
        "csp_cov_est": "concat",
        "riemann_cov_estimator": RIEMANN_COV_ESTIMATOR,
        "riemann_metric": RIEMANN_METRIC,
        "riemann_recentering": "none (would require held-out subject data)",
        "classifier_hyperparameters": "identical to frozen Phase 2 baselines; untuned",
    }
