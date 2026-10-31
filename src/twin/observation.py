"""The input record for one digital-twin update.

An :class:`Observation` is everything the outside world hands the twin at one
decision point: what the neuro-state encoder said, what the virtual patient did,
and what the environment had configured. Any of the first two may be ``None``,
because a real session drops frames, loses tracking and produces windows the
encoder refuses to score. Handling that is not an edge case to bolt on later --
it is the normal path, and :meth:`Observation.channels` is what lets the latent
estimator widen its uncertainty instead of pretending.

The observation is also the replay unit. A sequence of observations plus a
:class:`~src.twin.latent.LatentModelConfig` fully determines every state the twin
will ever hold, which is what makes :func:`~src.twin.twin.DigitalTwin.replay`
exact rather than approximate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from src.twin.states import MotorState, NeuroState, TaskState

#: Channel names used in :attr:`Observation.channels` and recorded on every
#: state so that a consumer can tell "not observed" from "not recorded".
CHANNEL_NEURO = "neuro"
CHANNEL_MOTOR = "motor"


@dataclass(frozen=True)
class Observation:
    """One tick of input to the twin.

    Parameters
    ----------
    t
        Simulation time in seconds since session start. Must be non-decreasing
        across a sequence; the twin rejects time going backwards rather than
        silently reordering, because a decision engine that acts on a stale
        state is a safety problem and not a tidiness problem.
    task
        The configuration the environment presented. Always required: the system
        knows what it asked for even when every sensor fails.
    neuro
        Encoder output, or ``None`` if this window produced none.
    motor
        Trial outcome, or ``None`` if no trial completed in this window. A twin
        tick is not the same thing as a trial: several ticks may pass while the
        patient is mid-reach.
    label
        Optional free-form tag carried through to the log. Used by the scenario
        generators to mark phase boundaries ("fatigue onset") so a figure can
        annotate them without re-deriving them.
    """

    t: float
    task: TaskState
    neuro: Optional[NeuroState] = None
    motor: Optional[MotorState] = None
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.t) or self.t < 0:
            raise ValueError(f"observation time must be finite and >= 0, got {self.t!r}")

    @property
    def channels(self) -> Tuple[str, ...]:
        """Which measured channels are actually present on this observation."""
        present = []
        if self.neuro is not None:
            present.append(CHANNEL_NEURO)
        if self.motor is not None:
            present.append(CHANNEL_MOTOR)
        return tuple(present)

    @property
    def is_empty(self) -> bool:
        """True when nothing was measured. The twin still ticks: time passed."""
        return self.neuro is None and self.motor is None

    def to_dict(self) -> Dict[str, object]:
        return {
            "t": self.t,
            "task": self.task.to_dict(),
            "neuro": None if self.neuro is None else self.neuro.to_dict(),
            "motor": None if self.motor is None else self.motor.to_dict(),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Observation":
        return cls(
            t=float(payload["t"]),  # type: ignore[arg-type]
            task=TaskState.from_dict(payload["task"]),  # type: ignore[arg-type]
            neuro=(None if payload.get("neuro") is None
                   else NeuroState.from_dict(payload["neuro"])),  # type: ignore[arg-type]
            motor=(None if payload.get("motor") is None
                   else MotorState.from_dict(payload["motor"])),  # type: ignore[arg-type]
            label=(None if payload.get("label") is None else str(payload["label"])),
        )
