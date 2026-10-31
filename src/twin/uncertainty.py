"""Uncertainty representation for the digital twin.

The single most important scientific constraint on this project is that EEG does
not measure fatigue, attention, recovery or rehabilitation state. Those live in
the twin as *estimates*, and an estimate without an uncertainty is a claim.

:class:`Estimate` is therefore the only container the twin has for a latent
value. There is no code path that stores a latent quantity as a bare float, and
:mod:`tests.test_twin_states` asserts that structurally. If a future component
wants a point value it has to reach through ``.value`` and, in doing so, say so
out loud.

Belief model
------------
Each latent variable is a scalar Gaussian belief truncated to a bounded interval
(``[0, 1]`` for every latent in Phase 6A). The Gaussian is an approximation: near
a boundary the true truncated posterior is skewed and its variance is smaller
than the untruncated one, so :meth:`Estimate.clamped` records that the mean was
clipped rather than pretending the belief is still Gaussian. Nothing downstream
currently depends on the difference, but the flag exists so that a later
component cannot silently assume symmetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

#: Variance assigned to a latent that has never been observed. A standard
#: deviation of ~0.29 is the variance of a uniform distribution on [0, 1]
#: (1/12), i.e. "we know nothing beyond the bounds".
PRIOR_VARIANCE: float = 1.0 / 12.0

#: Floor on the posterior variance. A belief that collapses to zero variance can
#: never be moved again by an observation, which is a modelling artefact rather
#: than knowledge. 1e-4 corresponds to a standard deviation of 0.01.
MIN_VARIANCE: float = 1e-4

#: Ceiling on the posterior variance, at the uniform-prior value. Without it, a
#: long gap in observations would inflate the variance past total ignorance.
MAX_VARIANCE: float = PRIOR_VARIANCE


@dataclass(frozen=True)
class Estimate:
    """A scalar latent quantity with its uncertainty and its provenance.

    Parameters
    ----------
    value
        Posterior mean, clipped to ``bounds``.
    variance
        Posterior variance, clipped to ``[MIN_VARIANCE, MAX_VARIANCE]``.
    n_updates
        How many observations have informed this estimate. Zero means the value
        is the prior and carries no evidence at all.
    last_observed_at
        Simulation time, in seconds, of the most recent observation that
        informed this estimate; ``None`` if it has never been observed. Used to
        detect staleness without consulting a wall clock.
    clamped
        ``True`` if the posterior mean was clipped to a bound on the most recent
        update, which means the Gaussian approximation is at its weakest.
    """

    value: float
    variance: float = PRIOR_VARIANCE
    n_updates: int = 0
    last_observed_at: Optional[float] = None
    clamped: bool = False

    #: Support of the latent. Every Phase 6A latent is a bounded score.
    bounds: Tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        lo, hi = self.bounds
        if not lo < hi:
            raise ValueError(f"bounds must be increasing, got {self.bounds}")
        if not math.isfinite(self.value):
            raise ValueError(f"estimate value must be finite, got {self.value!r}")
        if not math.isfinite(self.variance) or self.variance <= 0:
            raise ValueError(f"variance must be positive and finite, got {self.variance!r}")
        if self.n_updates < 0:
            raise ValueError("n_updates cannot be negative")

    # ------------------------------------------------------------------ views

    @property
    def sd(self) -> float:
        """Posterior standard deviation."""
        return math.sqrt(self.variance)

    @property
    def ci95(self) -> Tuple[float, float]:
        """Central 95% interval, truncated to the support.

        Truncation is what makes this honest at the boundary: an estimate of
        0.02 does not get an interval reaching to -0.5.
        """
        lo, hi = self.bounds
        half = 1.959963984540054 * self.sd
        return (max(lo, self.value - half), min(hi, self.value + half))

    @property
    def is_informed(self) -> bool:
        """Has any observation moved this estimate off its prior?"""
        return self.n_updates > 0

    def age(self, now: float) -> Optional[float]:
        """Seconds since the last informing observation, or ``None`` if never."""
        if self.last_observed_at is None:
            return None
        return max(0.0, now - self.last_observed_at)

    def is_stale(self, now: float, max_age: float) -> bool:
        """Whether this estimate is older than ``max_age`` seconds.

        An uninformed estimate is always stale: never having been observed is
        not the same as having been observed recently.
        """
        seconds = self.age(now)
        return True if seconds is None else seconds > max_age

    # ------------------------------------------------------------- operations

    def clipped(self) -> "Estimate":
        """Return a copy with the mean forced into ``bounds``.

        Sets :attr:`clamped` when the clip actually bites, so a consumer can see
        that the Gaussian approximation is being stretched.
        """
        lo, hi = self.bounds
        clipped_value = min(hi, max(lo, self.value))
        variance = min(MAX_VARIANCE, max(MIN_VARIANCE, self.variance))
        return replace(
            self,
            value=clipped_value,
            variance=variance,
            clamped=clipped_value != self.value,
        )

    def predicted(self, *, drift: float, process_variance: float) -> "Estimate":
        """Time-update: move the mean by ``drift`` and inflate the variance.

        This is the half of the filter that runs even when nothing is observed,
        which is what makes an unobserved latent decay towards ignorance instead
        of staying confidently wrong.
        """
        if process_variance < 0:
            raise ValueError("process variance cannot be negative")
        return replace(
            self,
            value=self.value + drift,
            variance=self.variance + process_variance,
        ).clipped()

    def updated(self, observation: float, observation_variance: float,
                *, at: float) -> "Estimate":
        """Measurement-update with a scalar pseudo-observation.

        ``observation_variance`` is how much the caller distrusts the
        pseudo-observation it derived from the measured variables. A large value
        makes the update nearly a no-op, which is the correct behaviour when the
        measured channel that produced it was missing or degraded.
        """
        if observation_variance <= 0:
            raise ValueError("observation variance must be positive")
        if not math.isfinite(observation):
            raise ValueError("observation must be finite")
        gain = self.variance / (self.variance + observation_variance)
        return replace(
            self,
            value=self.value + gain * (observation - self.value),
            variance=(1.0 - gain) * self.variance,
            n_updates=self.n_updates + 1,
            last_observed_at=at,
        ).clipped()

    # --------------------------------------------------------- serialisation

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "variance": self.variance,
            "sd": self.sd,
            "ci95_low": self.ci95[0],
            "ci95_high": self.ci95[1],
            "n_updates": self.n_updates,
            "last_observed_at": self.last_observed_at,
            "clamped": self.clamped,
            "bounds": list(self.bounds),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Estimate":
        """Inverse of :meth:`to_dict`; derived fields are ignored, not trusted."""
        return cls(
            value=float(payload["value"]),
            variance=float(payload["variance"]),
            n_updates=int(payload.get("n_updates", 0)),
            last_observed_at=(None if payload.get("last_observed_at") is None
                              else float(payload["last_observed_at"])),
            clamped=bool(payload.get("clamped", False)),
            bounds=tuple(payload.get("bounds", (0.0, 1.0))),  # type: ignore[arg-type]
        )


def uninformed(bounds: Tuple[float, float] = (0.0, 1.0)) -> Estimate:
    """A prior estimate: the midpoint of the support, at maximum variance."""
    lo, hi = bounds
    return Estimate(value=0.5 * (lo + hi), variance=PRIOR_VARIANCE, bounds=bounds)
