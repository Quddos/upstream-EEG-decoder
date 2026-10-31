"""Phase 6A -- virtual patient digital twin: data model and state engine.

This package is the first component of Phase 6 (digital twin and adaptive VR).
It holds no EEG code, no Unity code, no policy and no learned model. It is a
state container with an update rule, and it runs headless.

Relationship to Paper 1
-----------------------
Paper 1 (the cross-subject motor-imagery study on PhysioNet EEGMMIDB S001--S040)
is frozen. Nothing in :mod:`src.twin` imports from ``src.features``,
``src.eval``, ``src.preprocessing`` or ``src.data``, reads a Paper 1 artifact, or
can change one. Two of Paper 1's findings are nonetheless *design inputs* here,
and both are recorded where they bite:

* Calibrated and uncalibrated posteriors are not equally trustworthy at equal
  accuracy, so :class:`~src.twin.states.NeuroState` records which it holds and
  the estimator inflates its observation variance for the uncalibrated case.
* An absolute cross-subject threshold on an amplitude-like quantity fails, so
  the fatigue term is computed against the performer's **own** early-session
  baseline rather than a cohort value.

The measured/estimated boundary
-------------------------------
EEG does not measure fatigue, attention, recovery or clinical rehabilitation
state. This package enforces that structurally rather than by convention:
measured quantities are plain floats on :class:`~src.twin.states.NeuroState`,
:class:`~src.twin.states.MotorState` and :class:`~src.twin.states.TaskState`;
estimated quantities exist only as :class:`~src.twin.uncertainty.Estimate`
objects on :class:`~src.twin.states.LatentState`, and an ``Estimate`` cannot be
constructed without a variance. The lists are exported as
:data:`~src.twin.states.MEASURED_VARIABLES` and
:data:`~src.twin.states.ESTIMATED_VARIABLES` so that the schema document, the
report and the tests all read from one place.

None of the latent variables has been validated against human data. They are
simulation variables for system development.
"""

from src.twin.history import StateHistory
from src.twin.interfaces import DecisionEngineView, LatentReadout
from src.twin.latent import (
    DEFAULT_LATENT_CONFIG,
    EstimatorMemory,
    LatentEstimator,
    LatentModelConfig,
    PseudoObservation,
)
from src.twin.observation import Observation
from src.twin.states import (
    ESTIMATED_VARIABLES,
    LEFT,
    MEASURED_VARIABLES,
    REST,
    RIGHT,
    LatentState,
    MotorState,
    NeuroState,
    PerformanceState,
    TaskState,
    VirtualPatientState,
)
from src.twin.twin import DigitalTwin, TwinConfig, TwinSnapshot
from src.twin.uncertainty import Estimate, uninformed

__all__ = [
    "DEFAULT_LATENT_CONFIG",
    "ESTIMATED_VARIABLES",
    "LEFT",
    "MEASURED_VARIABLES",
    "REST",
    "RIGHT",
    "DecisionEngineView",
    "DigitalTwin",
    "Estimate",
    "EstimatorMemory",
    "LatentEstimator",
    "LatentModelConfig",
    "LatentReadout",
    "LatentState",
    "MotorState",
    "NeuroState",
    "Observation",
    "PerformanceState",
    "PseudoObservation",
    "StateHistory",
    "TaskState",
    "TwinConfig",
    "TwinSnapshot",
    "VirtualPatientState",
    "uninformed",
]
