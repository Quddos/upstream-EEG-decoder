"""Temporal state buffer for the digital twin.

A bounded ring buffer of :class:`~src.twin.states.VirtualPatientState` snapshots,
plus the queries a decision engine or a figure actually needs: the last *n*
ticks, everything inside a time window, and a flat table for analysis.

Bounded on purpose. A VR session running at a few hertz for an hour produces
tens of thousands of snapshots, and an unbounded buffer would make the twin's
memory footprint a function of session length. The bound is why
:class:`~src.twin.states.PerformanceState` accumulates incrementally instead of
being recomputed from history: dropping old snapshots must not change any
session-level number.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence

import pandas as pd

from src.twin.states import VirtualPatientState

#: Default capacity. At 2 Hz this holds roughly 40 minutes of session.
DEFAULT_CAPACITY: int = 4096


class StateHistory:
    """Append-only, bounded, ordered store of twin snapshots."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._buffer: Deque[VirtualPatientState] = deque(maxlen=capacity)
        self._appended = 0

    # ------------------------------------------------------------- container

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[VirtualPatientState]:
        return iter(self._buffer)

    def __getitem__(self, index: int) -> VirtualPatientState:
        return list(self._buffer)[index]

    @property
    def capacity(self) -> int:
        return self._buffer.maxlen or 0

    @property
    def total_appended(self) -> int:
        """How many snapshots were ever appended, including evicted ones.

        Distinct from ``len``: a caller that wants to know whether history has
        been truncated compares the two, rather than guessing.
        """
        return self._appended

    @property
    def truncated(self) -> bool:
        return self._appended > len(self._buffer)

    # ---------------------------------------------------------------- writes

    def append(self, state: VirtualPatientState) -> None:
        """Add a snapshot. Time must not go backwards."""
        if self._buffer and state.t < self._buffer[-1].t:
            raise ValueError(
                f"history time must be non-decreasing: {state.t} follows "
                f"{self._buffer[-1].t}"
            )
        self._buffer.append(state)
        self._appended += 1

    def clear(self) -> None:
        self._buffer.clear()
        self._appended = 0

    # ----------------------------------------------------------------- reads

    @property
    def latest(self) -> Optional[VirtualPatientState]:
        return self._buffer[-1] if self._buffer else None

    @property
    def first(self) -> Optional[VirtualPatientState]:
        return self._buffer[0] if self._buffer else None

    def last(self, n: int) -> List[VirtualPatientState]:
        """The most recent ``n`` snapshots, oldest first."""
        if n < 0:
            raise ValueError("n cannot be negative")
        return list(self._buffer)[-n:] if n else []

    def window(self, seconds: float, *, now: Optional[float] = None
               ) -> List[VirtualPatientState]:
        """Snapshots within ``seconds`` of ``now`` (default: the latest tick)."""
        if seconds < 0:
            raise ValueError("window length cannot be negative")
        if not self._buffer:
            return []
        end = self._buffer[-1].t if now is None else now
        return [s for s in self._buffer if end - s.t <= seconds]

    def at_or_before(self, t: float) -> Optional[VirtualPatientState]:
        """The most recent snapshot no later than ``t``, or ``None``.

        The twin does not interpolate between snapshots: a state at a time it
        never computed is a fabrication, and a decision engine asking for one
        should get the real previous state and know how old it is.
        """
        candidate = None
        for state in self._buffer:
            if state.t <= t:
                candidate = state
            else:
                break
        return candidate

    # ------------------------------------------------------------- analysis

    def to_frame(self) -> pd.DataFrame:
        """Flat table: one row per snapshot, latent estimates with their SDs.

        Column names mirror the dotted paths in
        :data:`~src.twin.states.MEASURED_VARIABLES` and
        :data:`~src.twin.states.ESTIMATED_VARIABLES` so that a reader can tell at
        a glance which side of the measured/estimated line a column sits on.
        """
        rows: List[Dict[str, object]] = []
        for state in self._buffer:
            row: Dict[str, object] = {
                "t": state.t,
                "tick": state.tick,
                "observed_channels": "|".join(state.observed_channels),
                "task.task_id": state.task.task_id,
                "task.difficulty": state.task.difficulty,
                "task.assistance_level": state.task.assistance_level,
                "task.target_side": state.task.target_side,
                "performance.trials": state.performance.trials,
                "performance.successes": state.performance.successes,
                "performance.success_rate": state.performance.success_rate,
                "performance.mean_movement_error": state.performance.mean_movement_error,
                "performance.mean_completion_time": state.performance.mean_completion_time,
                "performance.mean_smoothness": state.performance.mean_smoothness,
                "performance.time_on_task_s": state.performance.time_on_task_s,
            }
            if state.neuro is not None:
                row.update({
                    "neuro.left_probability": state.neuro.left_probability,
                    "neuro.right_probability": state.neuro.right_probability,
                    "neuro.lateralisation": state.neuro.lateralisation,
                    "neuro.decoder_confidence": state.neuro.decoder_confidence,
                    "neuro.uncertainty": state.neuro.uncertainty,
                    "neuro.predicted_class": state.neuro.predicted_class,
                    "neuro.calibrated": state.neuro.calibrated,
                })
            if state.motor is not None:
                row.update({
                    "motor.task_success": state.motor.task_success,
                    "motor.movement_error": state.motor.movement_error,
                    "motor.completion_time": state.motor.completion_time,
                    "motor.smoothness": state.motor.smoothness,
                    "motor.reaction_time": state.motor.reaction_time,
                    "motor.repetitions": state.motor.repetitions,
                })
            for name, estimate in state.latent.as_mapping().items():
                row[f"latent.{name}"] = estimate.value
                row[f"latent.{name}.sd"] = estimate.sd
                row[f"latent.{name}.n_updates"] = estimate.n_updates
            rows.append(row)
        return pd.DataFrame(rows)

    def to_list(self) -> List[Dict[str, object]]:
        return [state.to_dict() for state in self._buffer]

    @classmethod
    def from_states(cls, states: Sequence[VirtualPatientState],
                    capacity: int = DEFAULT_CAPACITY) -> "StateHistory":
        history = cls(capacity=capacity)
        for state in states:
            history.append(state)
        return history
