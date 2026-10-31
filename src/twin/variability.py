"""Phase 6H -- observation variance estimated from observed dispersion.

The frozen estimator uses a constant ``R = 0.025``. Because the Kalman variance
update is ``P_post = (1-K)P_pred`` with ``K = P/(P+R)``, the innovation moves the
*mean* and never the *variance*, so posterior width follows a fixed schedule and
floors at ``sd* ~ (R*Q)^(1/4)``. Phase 6H asks whether ``R`` can instead reflect
what the observations actually did.

Model D, as audited
-------------------
::

    d_t      = z_t - z_{t-1}                    successive differences
    MAD      = median(|d - median(d)|)          robust scale of the differences
    sigma_d  = 1.4826 * MAD                     normal-consistent conversion
    R_hat    = sigma_d^2 / 2                    <- see "the factor of two" below
    R        = clip(lambda*R_hat + (1-lambda)*R0, R_MIN, R_MAX)

The factor of two, and what it assumes
--------------------------------------
For independent, identically distributed observation noise,
``Var(z_t - z_{t-1}) = Var(z_t) + Var(z_{t-1}) = 2 Var(z)``. Dividing by two is
what converts a dispersion of *differences* into a dispersion of *observations*.
Omitting it would overstate ``R`` by a factor of two and understate confidence
everywhere.

**That derivation assumes iid noise, and this project's data is not iid.** Three
reasons, all of which mean the number below is an *observed-dispersion estimate*
and not an identification of measurement noise:

1. Successive trials are not independent -- the simulated participant has a
   capability that drifts, and a real one would have fatigue and learning.
2. Differencing removes a locally linear trend, which is why it is used here, but
   it does not remove curvature. A smoothly accelerating change still leaks into
   the dispersion estimate.
3. ``z`` is a clipped weighted composite of a Bernoulli success flag and two
   bounded scores. It is not Gaussian, so the 1.4826 factor -- which is the
   normal-consistency constant for the MAD -- is an approximation whose error
   depends on the shape of the distribution.

Under-dispersion of differences relative to ``2 Var(z)`` therefore means
*something about the sequence*, not necessarily *something about the sensor*. The
report says so in those words, and the claim made anywhere in Phase 6H is
"responsive to observed dispersion", never "measures physiological variability".

Why MAD rather than the sample variance
---------------------------------------
A sample variance has a breakdown point of zero: one extreme value moves it
without limit. The MAD's breakdown point is 50%. Since ``sd* ~ R^(1/4)``, an
``R`` inflated by a single outlier would widen every interval for as long as that
outlier stays in the window, and an ``R`` deflated by a lucky run would narrow
them -- so robustness matters more here than efficiency.

Everything in this module is a pure function of its arguments. No clock, no RNG,
no global state.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

#: Normal-consistency constant: for Gaussian data, ``1.4826 * MAD -> sigma``.
MAD_TO_SIGMA: float = 1.4826

#: The frozen baseline observation variance this model shrinks towards.
#: Read from Phase 6A rather than restated, so the two cannot drift apart.
BASELINE_R: float = 0.025


class VariabilityConfigError(ValueError):
    """Raised when a configuration is inadmissible. Never silently repaired."""


@dataclass(frozen=True)
class VariabilityConfig:
    """Model D's parameters. **Every value here was declared before the primary
    experiment was run**, and each is justified from the baseline scale rather
    than chosen against any downstream outcome.

    Phase 6G's precedent is followed deliberately: this is a *separate* object
    from ``LatentModelConfig``, so the frozen latent fingerprint
    ``42dcd25435726422`` is unchanged whenever this is absent, and the adaptive
    path carries an identity of its own when it is present.
    """

    model_id: str = "phase6h.model_d.v1"

    #: Number of successive differences in the window. Eight differences (nine
    #: observations) is long enough for a MAD to mean something and short enough
    #: to follow a session that changes. Not swept against outcomes.
    window: int = 8

    #: Differences required before the adaptive estimate is used at all. Below
    #: this the baseline R is used unchanged -- a dispersion estimate from three
    #: points is not a dispersion estimate.
    min_differences: int = 5

    #: Weight on the empirical estimate against the baseline. 0.5 is the neutral
    #: choice: with only ``window`` differences the empirical estimate is itself
    #: noisy, and equal weighting says so without pretending to know better.
    #: **This value directly compresses the achievable effect** (see the report),
    #: and is reported as a declared secondary sensitivity rather than tuned.
    shrinkage: float = 0.5

    #: Hard floor on R. One quarter of the baseline. Because ``sd* ~ R^(1/4)``,
    #: an underestimated R buys confidence cheaply, and this is the only
    #: structural defence against that. Symmetric in log-space with the ceiling.
    r_min: float = BASELINE_R / 4.0

    #: Hard ceiling on R. Four times the baseline. Stops one pathological window
    #: from making the twin permanently ignorant.
    r_max: float = BASELINE_R * 4.0

    #: The value shrunk towards, and the value used during warm-up.
    baseline_r: float = BASELINE_R

    def __post_init__(self) -> None:
        if self.window < 2:
            raise VariabilityConfigError("window needs at least 2 differences")
        if not 1 <= self.min_differences <= self.window:
            raise VariabilityConfigError(
                "min_differences must lie in [1, window]")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise VariabilityConfigError("shrinkage must lie in [0, 1]")
        if not 0.0 < self.r_min <= self.r_max:
            raise VariabilityConfigError("need 0 < r_min <= r_max")
        if not self.r_min <= self.baseline_r <= self.r_max:
            raise VariabilityConfigError(
                "the baseline must lie inside [r_min, r_max]; otherwise warm-up "
                "and the adaptive path would disagree about what is admissible")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------- estimators


def median(values: Sequence[float]) -> float:
    """Median of a non-empty sequence. Deterministic for even counts."""
    if not values:
        raise VariabilityConfigError("median of an empty sequence")
    ordered = sorted(values)
    n = len(ordered)
    middle = n // 2
    if n % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def median_absolute_deviation(values: Sequence[float]) -> float:
    """MAD about the median. Zero for a constant sequence, which is correct."""
    if not values:
        raise VariabilityConfigError("MAD of an empty sequence")
    centre = median(values)
    return median([abs(v - centre) for v in values])


def successive_differences(values: Sequence[float]) -> Tuple[float, ...]:
    return tuple(values[i] - values[i - 1] for i in range(1, len(values)))


@dataclass(frozen=True)
class VarianceEstimate:
    """The observation variance actually used, and everything behind it."""

    #: What was passed to ``Estimate.updated``.
    r: float
    #: The empirical part before shrinkage and clipping, or ``None`` in warm-up.
    r_empirical: Optional[float]
    #: Robust scale of the differences, or ``None`` in warm-up.
    sigma_delta: Optional[float]
    mad_delta: Optional[float]
    n_differences: int
    #: Which branch produced ``r``: ``warm_up`` | ``adaptive``.
    source: str
    #: Whether a bound bit, and which.
    clipped: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def observation_variance(window: Sequence[float],
                         config: VariabilityConfig) -> VarianceEstimate:
    """Model D. Returns the ``R`` to use and the full derivation behind it.

    ``window`` is the recent pseudo-observation history **excluding the current
    observation**. Excluding it is deliberate: letting a measurement set its own
    trust is circular, and would let a single surprising value simultaneously
    move the mean and excuse itself by inflating the variance.
    """
    differences = successive_differences(tuple(window))

    if len(differences) < config.min_differences:
        # Warm-up. The baseline is used unchanged rather than a dispersion being
        # fabricated from too few points.
        return VarianceEstimate(
            r=config.baseline_r, r_empirical=None, sigma_delta=None,
            mad_delta=None, n_differences=len(differences), source="warm_up")

    recent = differences[-config.window:]
    mad = median_absolute_deviation(recent)
    sigma_delta = MAD_TO_SIGMA * mad
    # Var(z_t - z_{t-1}) = 2 Var(z) under iid noise; see the module docstring for
    # what that assumption does and does not license.
    r_empirical = (sigma_delta ** 2) / 2.0

    blended = config.shrinkage * r_empirical + (1.0 - config.shrinkage) * config.baseline_r

    clipped: Optional[str] = None
    r = blended
    if r < config.r_min:
        r, clipped = config.r_min, "r_min"
    elif r > config.r_max:
        r, clipped = config.r_max, "r_max"

    return VarianceEstimate(
        r=r, r_empirical=r_empirical, sigma_delta=sigma_delta, mad_delta=mad,
        n_differences=len(recent), source="adaptive", clipped=clipped)

