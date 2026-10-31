"""Read-only view of the twin for the future decision engine.

Phase 6A does not build a decision engine, an NPC therapist or any policy. What
it does build is the *boundary* they will sit behind, because the boundary is
where the scientific constraint has to be enforced. A decision engine that can
read ``fatigue_estimate`` as a bare float will eventually be written as though
fatigue were measured.

:class:`DecisionEngineView` therefore exposes:

* measured quantities as plain values;
* latent quantities only as :class:`~src.twin.uncertainty.Estimate` objects,
  which cannot exist without a variance;
* an explicit :meth:`DecisionEngineView.latent_is_usable` predicate, so that
  "the estimate is too uncertain or too stale to act on" is a first-class answer
  rather than something each policy re-invents.

Nothing here decides anything. It is a projection of the twin, and it is the
only surface Phase 6C onwards should read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from src.twin.states import LatentState, VirtualPatientState
from src.twin.twin import DigitalTwin
from src.twin.uncertainty import Estimate

#: Default posterior SD above which a latent is considered too uncertain to act
#: on. 0.20 is a little over two thirds of the uniform-prior SD (0.289), so it
#: rejects beliefs that have barely moved off the prior.
DEFAULT_MAX_SD: float = 0.20

#: Default staleness limit, in seconds.
DEFAULT_MAX_AGE_S: float = 30.0


@dataclass(frozen=True)
class LatentReadout:
    """A latent estimate plus the twin's own verdict on whether to use it."""

    name: str
    estimate: Estimate
    usable: bool
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "usable": self.usable, "reason": self.reason,
                **self.estimate.to_dict()}


class DecisionEngineView:
    """Everything a policy may read, and nothing it may write.

    Parameters
    ----------
    twin
        The twin to project. Held by reference: the view is a window, not a copy,
        so a policy always reads the current state.
    max_sd, max_age_s
        Thresholds used by :meth:`latent_is_usable`. They are policy parameters,
        not scientific ones, and are exposed so a policy can tighten them.
    """

    def __init__(self, twin: DigitalTwin, *, max_sd: float = DEFAULT_MAX_SD,
                 max_age_s: float = DEFAULT_MAX_AGE_S) -> None:
        self._twin = twin
        self.max_sd = max_sd
        self.max_age_s = max_age_s

    # ------------------------------------------------------------- measured

    @property
    def now(self) -> float:
        return self._twin.state.t

    @property
    def measured(self) -> Dict[str, object]:
        """Measured quantities only, flattened. Never contains a latent.

        A policy that only needs measured evidence -- trial success, error,
        difficulty -- can read this and be structurally unable to touch an
        estimate by accident.
        """
        state: VirtualPatientState = self._twin.state
        out: Dict[str, object] = {
            "t": state.t,
            "tick": state.tick,
            "observed_channels": state.observed_channels,
            "task.task_id": state.task.task_id,
            "task.difficulty": state.task.difficulty,
            "task.assistance_level": state.task.assistance_level,
            "task.target_side": state.task.target_side,
            "performance.trials": state.performance.trials,
            "performance.successes": state.performance.successes,
            "performance.success_rate": state.performance.success_rate,
            "performance.time_on_task_s": state.performance.time_on_task_s,
            "session.consecutive_failures": self._twin.memory.consecutive_failures,
        }
        if state.neuro is not None:
            out.update({
                "neuro.left_probability": state.neuro.left_probability,
                "neuro.right_probability": state.neuro.right_probability,
                "neuro.predicted_class": state.neuro.predicted_class,
                "neuro.decoder_confidence": state.neuro.decoder_confidence,
                "neuro.uncertainty": state.neuro.uncertainty,
                "neuro.lateralisation": state.neuro.lateralisation,
                "neuro.calibrated": state.neuro.calibrated,
            })
        if state.motor is not None:
            out.update({
                "motor.task_success": state.motor.task_success,
                "motor.movement_error": state.motor.movement_error,
                "motor.completion_time": state.motor.completion_time,
                "motor.smoothness": state.motor.smoothness,
                "motor.reaction_time": state.motor.reaction_time,
            })
        return out

    # ------------------------------------------------------------ estimated

    def latent(self, name: str) -> LatentReadout:
        """One latent, with its uncertainty and a usability verdict."""
        if name not in LatentState.FIELDS:
            raise KeyError(
                f"unknown latent {name!r}; expected one of {LatentState.FIELDS}"
            )
        estimate: Estimate = getattr(self._twin.state.latent, name)
        usable, reason = self._usability(estimate)
        return LatentReadout(name=name, estimate=estimate, usable=usable, reason=reason)

    def latents(self) -> Dict[str, LatentReadout]:
        return {name: self.latent(name) for name in LatentState.FIELDS}

    def latent_is_usable(self, name: str) -> bool:
        return self.latent(name).usable

    def _usability(self, estimate: Estimate) -> Tuple[bool, str]:
        if not estimate.is_informed:
            return False, "uninformed: no observation has moved this estimate"
        if estimate.sd > self.max_sd:
            return False, f"too uncertain: sd {estimate.sd:.3f} > {self.max_sd:.3f}"
        if estimate.is_stale(self.now, self.max_age_s):
            age = estimate.age(self.now)
            return False, f"stale: last informed {age:.1f}s ago"
        return True, "usable"

    # ------------------------------------------------------------- summary

    def summary(self) -> Dict[str, object]:
        """A single dictionary suitable for logging or for an NPC's context.

        Latents appear as nested dictionaries carrying ``value``, ``sd``,
        ``ci95`` and ``usable``. There is deliberately no flattened
        ``fatigue_estimate: 0.7`` key anywhere in the payload.
        """
        return {
            "patient_id": self._twin.config.patient_id,
            "config_fingerprint": self._twin.config.fingerprint,
            "measured": self.measured,
            "estimated": {name: readout.to_dict()
                          for name, readout in self.latents().items()},
            "caveat": (
                "Latent values are model estimates produced by a simulator for "
                "system development. They are not measurements and have not been "
                "validated against human data."
            ),
        }
