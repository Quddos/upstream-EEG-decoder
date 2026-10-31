"""The digital twin: state container, update loop and replay.

:class:`DigitalTwin` owns the current :class:`~src.twin.states.VirtualPatientState`,
the history buffer and the estimator memory, and exposes exactly three verbs:
:meth:`~DigitalTwin.observe`, :meth:`~DigitalTwin.reset` and
:meth:`~DigitalTwin.snapshot`. Everything else is a read.

Determinism
-----------
The twin never reads a clock, never touches a global random number generator and
never mutates a state object in place. Time arrives on the observation; states
are frozen dataclasses and are replaced, not edited. The consequence is that

    ``DigitalTwin(config).run(observations)``

is a pure function of ``(config, observations)``, verified by
:meth:`~DigitalTwin.trajectory_hash` and by
:mod:`tests.test_twin_determinism`. That property is what makes Phase 6B able to
swap the simulator for the frozen Paper 1 encoder and still be able to tell
whether a behaviour change came from the encoder or from the twin.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.twin.history import DEFAULT_CAPACITY, StateHistory
from src.twin.latent import (
    DEFAULT_LATENT_CONFIG,
    EstimatorMemory,
    LatentEstimator,
    LatentModelConfig,
    PseudoObservation,
)
from src.twin.observation import Observation
from src.twin.states import (
    LatentState,
    PerformanceState,
    TaskState,
    VirtualPatientState,
)


@dataclass(frozen=True)
class TwinConfig:
    """Configuration of the twin itself, distinct from the latent model.

    Split deliberately: :class:`~src.twin.latent.LatentModelConfig` holds the
    scientific assumptions and is what a reader should scrutinise, while this
    holds engineering choices that cannot change a conclusion.
    """

    patient_id: str = "VP000"
    history_capacity: int = DEFAULT_CAPACITY
    latent: LatentModelConfig = field(default_factory=lambda: DEFAULT_LATENT_CONFIG)

    @property
    def fingerprint(self) -> str:
        payload = {
            "patient_id": self.patient_id,
            "history_capacity": self.history_capacity,
            "latent": self.latent.fingerprint,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TwinSnapshot:
    """Everything needed to resume a twin exactly where it was left.

    The current state alone is not enough: the estimator's within-subject
    baseline and its previous-tick memories also determine the future, so they
    travel together or resumption is not exact.
    """

    state: VirtualPatientState
    memory: EstimatorMemory
    tick: int

    def to_dict(self) -> Dict[str, object]:
        return {"state": self.state.to_dict(), "memory": self.memory.to_dict(),
                "tick": self.tick}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TwinSnapshot":
        return cls(
            state=VirtualPatientState.from_dict(payload["state"]),  # type: ignore[arg-type]
            memory=EstimatorMemory.from_dict(payload["memory"]),  # type: ignore[arg-type]
            tick=int(payload["tick"]),  # type: ignore[arg-type]
        )


#: Task configuration used before the first observation arrives. A twin has to
#: hold *some* task state to be a valid object, and inventing a plausible one
#: would be worse than an explicitly named placeholder.
IDLE_TASK = TaskState(task_id="idle", difficulty=0.0, assistance_level=0.0)


class DigitalTwin:
    """A virtual patient's evolving state.

    Parameters
    ----------
    config
        Twin and latent-model configuration. Frozen and hashed; the fingerprint
        is written into every artifact the twin produces.
    """

    def __init__(self, config: TwinConfig = TwinConfig()) -> None:
        self.config = config
        self._estimator = LatentEstimator(config.latent)
        self._history = StateHistory(capacity=config.history_capacity)
        self._state: VirtualPatientState = self._initial_state()
        self._memory = EstimatorMemory()
        self._tick = 0
        self._last_pseudo: Dict[str, Optional[PseudoObservation]] = {}

    # ------------------------------------------------------------- lifecycle

    def _initial_state(self) -> VirtualPatientState:
        return VirtualPatientState(
            t=0.0,
            tick=0,
            neuro=None,
            motor=None,
            task=IDLE_TASK,
            performance=PerformanceState(),
            latent=LatentState(),
            observed_channels=(),
        )

    def reset(self) -> None:
        """Return to the initial state, clearing history and estimator memory.

        A reset must be total. Leaving the estimator baseline behind would make
        the second session of a twin behave differently from the first, and the
        difference would be invisible in the state object.
        """
        self._history.clear()
        self._state = self._initial_state()
        self._memory = EstimatorMemory()
        self._tick = 0
        self._last_pseudo = {}

    # ---------------------------------------------------------------- update

    def observe(self, obs: Observation) -> VirtualPatientState:
        """Fold one observation in and return the new state.

        The order matters and is fixed: measured performance is updated first,
        because the fatigue pseudo-observation reads ``time_on_task_s``; then the
        latent beliefs are advanced; then the snapshot is appended. Reversing the
        first two would make fatigue lag performance by one tick for no reason.
        """
        if obs.t < self._state.t:
            raise ValueError(
                f"observation time {obs.t} precedes current state time {self._state.t}; "
                "the twin does not reorder observations"
            )
        dt = obs.t - self._state.t

        performance = self._state.performance
        if obs.motor is not None:
            performance = performance.extended(obs.motor, dt=dt)
        else:
            performance = PerformanceState(
                trials=performance.trials,
                successes=performance.successes,
                total_repetitions=performance.total_repetitions,
                mean_movement_error=performance.mean_movement_error,
                mean_completion_time=performance.mean_completion_time,
                mean_smoothness=performance.mean_smoothness,
                time_on_task_s=performance.time_on_task_s + max(0.0, dt),
            )

        latent, memory, pseudo = self._estimator.update(
            self._state.latent, self._memory, obs,
            dt=dt, time_on_task_s=performance.time_on_task_s,
        )

        self._tick += 1
        self._memory = memory
        self._last_pseudo = pseudo
        self._state = VirtualPatientState(
            t=obs.t,
            tick=self._tick,
            neuro=obs.neuro,
            motor=obs.motor,
            task=obs.task,
            performance=performance,
            latent=latent,
            observed_channels=obs.channels,
        )
        self._history.append(self._state)
        return self._state

    def run(self, observations: Iterable[Observation]) -> List[VirtualPatientState]:
        """Fold in a whole sequence, returning every intermediate state."""
        return [self.observe(obs) for obs in observations]

    # ----------------------------------------------------------------- reads

    @property
    def state(self) -> VirtualPatientState:
        return self._state

    @property
    def history(self) -> StateHistory:
        return self._history

    @property
    def memory(self) -> EstimatorMemory:
        return self._memory

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def last_pseudo_observations(self) -> Dict[str, Optional[PseudoObservation]]:
        """The pseudo-observations that produced the most recent update.

        Exposed so that a trajectory can be audited: if a latent moved, this says
        which measured quantities moved it and how much they were trusted.
        """
        return dict(self._last_pseudo)

    def snapshot(self) -> TwinSnapshot:
        return TwinSnapshot(state=self._state, memory=self._memory, tick=self._tick)

    def restore(self, snapshot: TwinSnapshot) -> None:
        """Resume from a snapshot. History is *not* restored, only the state.

        Restoring history would make it ambiguous whether a snapshot is a
        resumption point or a whole session recording. It is the former.
        """
        self._state = snapshot.state
        self._memory = snapshot.memory
        self._tick = snapshot.tick
        self._history.clear()
        self._history.append(snapshot.state)

    # ------------------------------------------------------------ determinism

    def trajectory_hash(self) -> str:
        """Stable hash of every state the twin has held.

        Two runs agree if and only if this agrees. Floats are serialised with
        ``repr`` through ``json``, which is exact for IEEE doubles, so the hash
        distinguishes trajectories that differ in the last bit.
        """
        blob = json.dumps(
            {"config": self.config.fingerprint, "states": self._history.to_list()},
            sort_keys=True, separators=(",", ":"), default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @classmethod
    def replay(cls, observations: Sequence[Observation],
               config: TwinConfig = TwinConfig()) -> "DigitalTwin":
        """Build a fresh twin and fold in a whole observation sequence.

        This is the canonical way to reproduce a session from a log: the log
        stores observations, not states, so a replay re-derives every state and
        would surface any accidental dependence on hidden context.
        """
        twin = cls(config)
        twin.run(observations)
        return twin
