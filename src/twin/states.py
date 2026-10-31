"""State containers for the virtual patient digital twin.

The type system carries the scientific constraint. There are two kinds of
quantity in this package and they are not interchangeable:

**Measured** -- something an instrument or the VR task actually produced. It is a
plain float on :class:`NeuroState`, :class:`MotorState` or :class:`TaskState`.
The decoder's class posterior is measured in this sense: it is a number the
decoder emitted, and calling it measured says nothing about what it means.

**Estimated** -- an inferred quantity that no instrument produced. It lives on
:class:`LatentState` and its type is :class:`~src.twin.uncertainty.Estimate`, so
it cannot be stored without a variance. Attention, fatigue, recovery and the
twin's self-confidence are all of this kind.

The distinction exists because EEG does not measure fatigue, attention, recovery
or clinical rehabilitation state. Phase 6A models them as latent simulation
variables so that a decision engine has something to reason over; that is a
software design choice, not a physiological claim, and nothing in this package
has been validated against human data.

Derived quantities
------------------
:attr:`NeuroState.decoder_confidence` and :attr:`NeuroState.uncertainty` are
computed from the class posterior rather than stored. Storing them separately
invites the three numbers to disagree, and a state object that can be internally
inconsistent is a bug waiting for a decision engine to find.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Optional, Tuple

from src.twin.uncertainty import Estimate, uninformed

#: Class labels the twin understands, matching the frozen Paper 1 task. The
#: names are duplicated rather than imported so that this package has no
#: dependency on the EEG pipeline; :mod:`tests.test_twin_states` pins them.
LEFT = "LEFT"
RIGHT = "RIGHT"
REST = "REST"

#: Tolerance on the class posterior summing to one, generous enough for float32
#: round-tripping through JSON and tight enough to catch a real bug.
PROBABILITY_TOLERANCE: float = 1e-6


def _check_unit(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
    return float(value)


def _check_non_negative(name: str, value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return float(value)


# ============================================================== measured state


@dataclass(frozen=True)
class NeuroState:
    """What the neuro-state encoder reported for one decision window.

    Every field here is *measured* in the sense that the encoder emitted it. In
    Phase 6A these values are supplied by a simulator; in Phase 6B they will come
    from the frozen Paper 1 representation. The container does not change.

    Parameters
    ----------
    left_probability, right_probability
        Class posterior over the binary LEFT/RIGHT task, summing to one.
    lateralisation
        The measured mu-band lateralisation index, ``log(P_C4 / P_C3)``, or
        ``None`` when the encoder did not produce one. Signed and unbounded;
        only the left-versus-right *difference* is interpretable, so an absolute
        threshold on this value would be an error (Paper 1, Sec. 3.9).
    calibrated
        Whether the posterior came from a calibrated decoder. This matters more
        than it looks: Paper 1 found that calibration substantially improves
        reliability without improving accuracy, so an uncalibrated posterior is a
        worse guide to trust even at identical accuracy. The latent estimator
        inflates its observation variance when this is ``False``.
    encoder_id
        Free-form identifier of the encoder that produced the values, recorded
        for provenance. Phase 6A uses ``"simulator"``.
    """

    left_probability: float
    right_probability: float
    lateralisation: Optional[float] = None
    calibrated: bool = False
    encoder_id: str = "simulator"

    def __post_init__(self) -> None:
        _check_unit("left_probability", self.left_probability)
        _check_unit("right_probability", self.right_probability)
        total = self.left_probability + self.right_probability
        if abs(total - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError(
                f"class posterior must sum to 1, got {total!r}. "
                "The twin never renormalises for you: a posterior that does not "
                "sum to one is an encoder bug, not a rounding detail."
            )
        if self.lateralisation is not None and not math.isfinite(self.lateralisation):
            raise ValueError("lateralisation must be finite or None")

    @property
    def decoder_confidence(self) -> float:
        """Posterior mass on the predicted class, in ``[0.5, 1]``."""
        return max(self.left_probability, self.right_probability)

    @property
    def predicted_class(self) -> str:
        """``LEFT`` or ``RIGHT``. Ties resolve to ``LEFT`` deterministically."""
        return LEFT if self.left_probability >= self.right_probability else RIGHT

    @property
    def uncertainty(self) -> float:
        """Normalised binary entropy of the posterior, in ``[0, 1]``.

        One means a coin flip. This is the decoder's *own* uncertainty about the
        class and is a computed property of the posterior, not an estimate of
        anything about the person.
        """
        p = self.left_probability
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))

    @property
    def margin(self) -> float:
        """``|p_left - p_right|``, in ``[0, 1]``. Zero at chance."""
        return abs(self.left_probability - self.right_probability)

    def to_dict(self) -> Dict[str, object]:
        return {
            "left_probability": self.left_probability,
            "right_probability": self.right_probability,
            "lateralisation": self.lateralisation,
            "calibrated": self.calibrated,
            "encoder_id": self.encoder_id,
            "decoder_confidence": self.decoder_confidence,
            "predicted_class": self.predicted_class,
            "uncertainty": self.uncertainty,
            "margin": self.margin,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "NeuroState":
        return cls(
            left_probability=float(payload["left_probability"]),  # type: ignore[arg-type]
            right_probability=float(payload["right_probability"]),  # type: ignore[arg-type]
            lateralisation=(None if payload.get("lateralisation") is None
                            else float(payload["lateralisation"])),  # type: ignore[arg-type]
            calibrated=bool(payload.get("calibrated", False)),
            encoder_id=str(payload.get("encoder_id", "simulator")),
        )


@dataclass(frozen=True)
class MotorState:
    """What the virtual patient actually did on one trial.

    All measured. ``smoothness`` is normalised so that larger is better, which is
    the opposite convention to the jerk-based metrics it would be computed from
    in a real implementation; the normalisation is the simulator's
    responsibility and is documented in the assumptions list.
    """

    task_success: bool
    movement_error: float
    completion_time: float
    smoothness: float
    reaction_time: float
    repetitions: int = 1

    def __post_init__(self) -> None:
        _check_non_negative("movement_error", self.movement_error)
        _check_non_negative("completion_time", self.completion_time)
        _check_unit("smoothness", self.smoothness)
        _check_non_negative("reaction_time", self.reaction_time)
        if self.repetitions < 0:
            raise ValueError("repetitions cannot be negative")

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_success": self.task_success,
            "movement_error": self.movement_error,
            "completion_time": self.completion_time,
            "smoothness": self.smoothness,
            "reaction_time": self.reaction_time,
            "repetitions": self.repetitions,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MotorState":
        return cls(
            task_success=bool(payload["task_success"]),
            movement_error=float(payload["movement_error"]),  # type: ignore[arg-type]
            completion_time=float(payload["completion_time"]),  # type: ignore[arg-type]
            smoothness=float(payload["smoothness"]),  # type: ignore[arg-type]
            reaction_time=float(payload["reaction_time"]),  # type: ignore[arg-type]
            repetitions=int(payload.get("repetitions", 1)),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class TaskState:
    """The configuration the environment presented for this trial.

    This is measured in the strongest sense available: it is what the system
    itself chose, so it is known exactly. It is separated from
    :class:`MotorState` because a future decision engine writes here and reads
    there, and keeping the two apart makes that direction of flow explicit.
    """

    task_id: str
    difficulty: float
    assistance_level: float
    target_side: Optional[str] = None
    target_configuration: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_unit("difficulty", self.difficulty)
        _check_unit("assistance_level", self.assistance_level)
        if self.target_side not in (None, LEFT, RIGHT):
            raise ValueError(f"target_side must be {LEFT!r}, {RIGHT!r} or None")

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "assistance_level": self.assistance_level,
            "target_side": self.target_side,
            "target_configuration": dict(self.target_configuration),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TaskState":
        return cls(
            task_id=str(payload["task_id"]),
            difficulty=float(payload["difficulty"]),  # type: ignore[arg-type]
            assistance_level=float(payload["assistance_level"]),  # type: ignore[arg-type]
            target_side=(None if payload.get("target_side") is None
                         else str(payload["target_side"])),
            target_configuration=dict(payload.get("target_configuration", {})),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PerformanceState:
    """Aggregated measured performance over the session so far.

    Every field is a running summary of :class:`MotorState` values the twin has
    already seen, so it is measured, not estimated. It exists so that a decision
    engine does not have to re-derive session-level context from the history
    buffer on every tick.
    """

    trials: int = 0
    successes: int = 0
    total_repetitions: int = 0
    mean_movement_error: float = 0.0
    mean_completion_time: float = 0.0
    mean_smoothness: float = 0.0
    time_on_task_s: float = 0.0

    @property
    def success_rate(self) -> Optional[float]:
        """Fraction of trials succeeded, or ``None`` before the first trial.

        ``None`` rather than 0.0: no trials is not the same as no successes, and
        a decision engine that cannot tell them apart will escalate difficulty
        for a patient who has not started.
        """
        return None if self.trials == 0 else self.successes / self.trials

    @property
    def consecutive_failure_capable(self) -> bool:
        """Whether enough trials exist for a failure streak to be meaningful."""
        return self.trials >= 2

    def to_dict(self) -> Dict[str, object]:
        return {
            "trials": self.trials,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "total_repetitions": self.total_repetitions,
            "mean_movement_error": self.mean_movement_error,
            "mean_completion_time": self.mean_completion_time,
            "mean_smoothness": self.mean_smoothness,
            "time_on_task_s": self.time_on_task_s,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PerformanceState":
        return cls(
            trials=int(payload.get("trials", 0)),  # type: ignore[arg-type]
            successes=int(payload.get("successes", 0)),  # type: ignore[arg-type]
            total_repetitions=int(payload.get("total_repetitions", 0)),  # type: ignore[arg-type]
            mean_movement_error=float(payload.get("mean_movement_error", 0.0)),  # type: ignore[arg-type]
            mean_completion_time=float(payload.get("mean_completion_time", 0.0)),  # type: ignore[arg-type]
            mean_smoothness=float(payload.get("mean_smoothness", 0.0)),  # type: ignore[arg-type]
            time_on_task_s=float(payload.get("time_on_task_s", 0.0)),  # type: ignore[arg-type]
        )

    def extended(self, motor: MotorState, *, dt: float) -> "PerformanceState":
        """Fold one trial in, returning a new instance.

        Means are updated incrementally rather than recomputed from history, so
        the result does not depend on how much history the buffer happens to
        retain. That is what keeps a bounded ring buffer from silently changing
        session-level statistics.
        """
        n = self.trials + 1

        def running(previous: float, new: float) -> float:
            return previous + (new - previous) / n

        return PerformanceState(
            trials=n,
            successes=self.successes + (1 if motor.task_success else 0),
            total_repetitions=self.total_repetitions + motor.repetitions,
            mean_movement_error=running(self.mean_movement_error, motor.movement_error),
            mean_completion_time=running(self.mean_completion_time, motor.completion_time),
            mean_smoothness=running(self.mean_smoothness, motor.smoothness),
            time_on_task_s=self.time_on_task_s + max(0.0, dt),
        )


# ============================================================= estimated state


@dataclass(frozen=True)
class LatentState:
    """Inferred quantities. Every field carries its own uncertainty.

    None of these is measured, none has been validated against human data, and
    none should be reported as a physiological or clinical fact. They are
    simulation variables that give a future decision engine something to reason
    over, and they are named ``*_estimate`` so that no downstream string
    formatting can quietly drop the qualifier.

    ``motor_state_estimate`` is the twin's belief about how well the person is
    currently expressing the trained motor pattern. It is deliberately separate
    from :class:`NeuroState`: the decoder posterior is about one 4-second window,
    this is about the session.
    """

    motor_state_estimate: Estimate = field(default_factory=uninformed)
    attention_estimate: Estimate = field(default_factory=uninformed)
    fatigue_estimate: Estimate = field(default_factory=uninformed)
    recovery_estimate: Estimate = field(default_factory=uninformed)
    confidence_estimate: Estimate = field(default_factory=uninformed)

    #: Field order used by every serialiser and by the schema document.
    FIELDS: Tuple[str, ...] = (
        "motor_state_estimate",
        "attention_estimate",
        "fatigue_estimate",
        "recovery_estimate",
        "confidence_estimate",
    )

    def as_mapping(self) -> Dict[str, Estimate]:
        return {name: getattr(self, name) for name in self.FIELDS}

    @property
    def mean_uncertainty(self) -> float:
        """Mean posterior SD across the latents; a one-number staleness signal."""
        values = [e.sd for e in self.as_mapping().values()]
        return float(sum(values) / len(values))

    @property
    def any_uninformed(self) -> bool:
        """True while at least one latent is still sitting on its prior."""
        return any(not e.is_informed for e in self.as_mapping().values())

    def to_dict(self) -> Dict[str, object]:
        return {name: est.to_dict() for name, est in self.as_mapping().items()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LatentState":
        return cls(**{name: Estimate.from_dict(payload[name])  # type: ignore[arg-type]
                      for name in cls.FIELDS})


# =============================================================== the composite


@dataclass(frozen=True)
class VirtualPatientState:
    """One timestamped snapshot of everything the twin believes and observed.

    ``t`` is simulation time in seconds since the start of the session, not a
    wall clock. Determinism is the point: replaying the same observation
    sequence must reproduce this object exactly, and a wall clock would make
    that impossible.
    """

    t: float
    tick: int
    neuro: Optional[NeuroState]
    motor: Optional[MotorState]
    task: TaskState
    performance: PerformanceState
    latent: LatentState

    #: Which measured channels were present on the update that produced this
    #: state. Recorded rather than inferred, because "absent" and "absent and
    #: also I forgot to record it" must not look the same downstream.
    observed_channels: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.t) or self.t < 0:
            raise ValueError(f"t must be a finite non-negative number, got {self.t!r}")
        if self.tick < 0:
            raise ValueError("tick cannot be negative")

    def with_latent(self, latent: LatentState) -> "VirtualPatientState":
        return replace(self, latent=latent)

    def to_dict(self) -> Dict[str, object]:
        return {
            "t": self.t,
            "tick": self.tick,
            "neuro": None if self.neuro is None else self.neuro.to_dict(),
            "motor": None if self.motor is None else self.motor.to_dict(),
            "task": self.task.to_dict(),
            "performance": self.performance.to_dict(),
            "latent": self.latent.to_dict(),
            "observed_channels": list(self.observed_channels),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "VirtualPatientState":
        return cls(
            t=float(payload["t"]),  # type: ignore[arg-type]
            tick=int(payload["tick"]),  # type: ignore[arg-type]
            neuro=(None if payload.get("neuro") is None
                   else NeuroState.from_dict(payload["neuro"])),  # type: ignore[arg-type]
            motor=(None if payload.get("motor") is None
                   else MotorState.from_dict(payload["motor"])),  # type: ignore[arg-type]
            task=TaskState.from_dict(payload["task"]),  # type: ignore[arg-type]
            performance=PerformanceState.from_dict(payload["performance"]),  # type: ignore[arg-type]
            latent=LatentState.from_dict(payload["latent"]),  # type: ignore[arg-type]
            observed_channels=tuple(payload.get("observed_channels", ())),  # type: ignore[arg-type]
        )


#: Names of every quantity the twin treats as measured, and every quantity it
#: treats as estimated. Exported so that the schema document, the report and
#: :mod:`tests.test_twin_states` all read from one place instead of three.
MEASURED_VARIABLES: Tuple[str, ...] = (
    "neuro.left_probability",
    "neuro.right_probability",
    "neuro.lateralisation",
    "neuro.decoder_confidence",
    "neuro.uncertainty",
    "neuro.margin",
    "motor.task_success",
    "motor.movement_error",
    "motor.completion_time",
    "motor.smoothness",
    "motor.reaction_time",
    "motor.repetitions",
    "task.difficulty",
    "task.assistance_level",
    "task.target_side",
    "performance.trials",
    "performance.successes",
    "performance.success_rate",
    "performance.time_on_task_s",
)

ESTIMATED_VARIABLES: Tuple[str, ...] = LatentState.FIELDS
