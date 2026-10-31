"""Authoritative description of the PhysioNet EEG Motor Movement/Imagery Database.

This module is the *single source of truth* for every dataset property used by the
Phase 1 pipeline.  Nothing about the dataset may be hard-coded anywhere else in the
codebase; if a fact about EEGMMIDB is needed, it is defined here and imported.

Reference
---------
EEG Motor Movement/Imagery Dataset, version 1.0.0
https://physionet.org/content/eegmmidb/1.0.0/
Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R.
"BCI2000: A General-Purpose Brain-Computer Interface (BCI) System."
IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.

Experimental protocol (as documented by PhysioNet)
--------------------------------------------------
Each of the 109 subjects performed 14 runs:

  * Run  1 -- one-minute baseline, eyes open
  * Run  2 -- one-minute baseline, eyes closed
  * Then three repetitions of four two-minute task conditions, in this order:
      - Task 1: open and close **left or right fist**            (motor EXECUTION)
      - Task 2: **imagine** opening and closing left or right fist (motor IMAGERY)
      - Task 3: open and close **both fists or both feet**       (motor EXECUTION)
      - Task 4: **imagine** opening/closing both fists or both feet (motor IMAGERY)

    giving runs 3,4,5,6 / 7,8,9,10 / 11,12,13,14.

Annotation semantics -- THIS IS RUN-DEPENDENT
---------------------------------------------
    T0  rest -- in every run.
    T1  onset of motion (real or imagined) of
            the LEFT FIST   in runs 3, 4, 7, 8, 11, 12
            BOTH FISTS      in runs 5, 6, 9, 10, 13, 14
    T2  onset of motion (real or imagined) of
            the RIGHT FIST  in runs 3, 4, 7, 8, 11, 12
            BOTH FEET       in runs 5, 6, 9, 10, 13, 14

It is therefore **incorrect** to globally assume ``T1 == left`` and ``T2 == right``.
The mapping is resolved per run by :func:`event_map_for_run`.

For unilateral *motor imagery* -- the paradigm this project decodes -- the only
valid runs are **4, 8 and 12**.  Runs 3, 7 and 11 are the unilateral *execution*
counterparts and are exposed separately so they can be used later as an auxiliary
or transfer-learning source, but they are NOT motor imagery and are excluded from
the Phase 1 default configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

# --------------------------------------------------------------------------------------
# Recording properties
# --------------------------------------------------------------------------------------

#: Official dataset identity, recorded in every provenance record.
DATASET_NAME: str = "PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)"
DATASET_VERSION: str = "1.0.0"
DATASET_DOI: str = "10.13026/C28G6P"
PHYSIONET_RECORD_URL: str = "https://physionet.org/content/eegmmidb/1.0.0/"
PHYSIONET_FILES_URL: str = "https://physionet.org/files/eegmmidb/1.0.0"
#: Official checksum manifest published alongside the record.
PHYSIONET_CHECKSUMS_URL: str = f"{PHYSIONET_FILES_URL}/SHA256SUMS.txt"

#: Nominal sampling frequency of every EEGMMIDB recording, in Hz.
SAMPLING_RATE_HZ: float = 160.0

#: Number of EEG channels (international 10-10 system, minus the nasion/inion pair
#: that the 10-10 layout does not include).
N_EEG_CHANNELS: int = 64

#: Total number of subjects in the complete release.
N_SUBJECTS_TOTAL: int = 109

#: Total number of runs recorded per subject.
N_RUNS_PER_SUBJECT: int = 14

#: Power-line frequency at the recording site (Wadsworth Center, New York, USA).
POWERLINE_FREQ_HZ: float = 60.0

#: Channel labels exactly as they appear in the EDF+ headers, in file order.
#: EDF pads every label to four characters with '.', which is why the trailing dots
#: are part of the on-disk name.  :func:`~src.data.loader.standardize_channel_names`
#: converts these to montage-compatible 10-05 names.
EDF_CHANNEL_NAMES: Tuple[str, ...] = (
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.",
    "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..",
    "Cp5.", "Cp3.", "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.",
    "Fp1.", "Fpz.", "Fp2.",
    "Af7.", "Af3.", "Afz.", "Af4.", "Af8.",
    "F7..", "F5..", "F3..", "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..",
    "Ft7.", "Ft8.",
    "T7..", "T8..", "T9..", "T10.",
    "Tp7.", "Tp8.",
    "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..", "P4..", "P6..", "P8..",
    "Po7.", "Po3.", "Poz.", "Po4.", "Po8.",
    "O1..", "Oz..", "O2..",
    "Iz..",
)

#: Electrodes overlying sensorimotor cortex.  Used for diagnostic plots and as an
#: optional channel subset; the default pipeline keeps all 64 channels.
#:
#: These 21 electrodes are the **decoder region of interest** agreed in Phase 1.7
#: (see :data:`SENSORIMOTOR_ROI` for the full rationale).  Three rows over the hand
#: area: fronto-central (FC), central (C) and centro-parietal (CP), each spanning
#: left hemisphere to right through the midline.
SENSORIMOTOR_CHANNELS: Tuple[str, ...] = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)

#: The channel set the first neural decoder will be trained on.
#:
#: Why a 21-channel ROI rather than all 64
#: ---------------------------------------
#: Motor-imagery ERD is a sensorimotor phenomenon: mu (8-13 Hz) and beta (13-30 Hz)
#: power drops over the hand area *contralateral* to the imagined movement.  The
#: electrodes above cover that area and its immediate surround.  Restricting to them
#: cuts the input dimension by 3x, which matters when a subject contributes only 69
#: epochs -- a 64-channel CSP or EEGNet has far more parameters than the data can
#: constrain, and the surplus channels are where a model finds shortcuts.
#:
#: Phase 1.6 gives the empirical argument.  Across S001 R04 every one of the 150 µV
#: threshold crossings fell outside this ROI: 36% frontopolar, 35% occipital, 22%
#: frontal, and **zero** on any FC/C/CP electrode.  The ocular and posterior-alpha
#: activity that dominates the amplitude distribution is largely outside the ROI,
#: so an ROI-restricted decoder is structurally less able to learn from it.
#:
#: What this is NOT
#: ----------------
#: This is a documented default, not a finding, and not a preprocessing change.
#: The stored arrays keep **all 64 channels** (see ``docs/PREPROCESSING_DECISIONS.md``);
#: the ROI is applied at model-input time by :func:`roi_indices`, so the 64-channel
#: vs 21-channel comparison remains available as a Phase 2 ablation.  Whether the ROI
#: actually helps is an empirical question that has not been tested -- no model has
#: been trained.
SENSORIMOTOR_ROI: Tuple[str, ...] = SENSORIMOTOR_CHANNELS

#: Human-readable name for the ROI, used in reports and provenance records.
SENSORIMOTOR_ROI_NAME: str = "sensorimotor_21"


def roi_indices(
    channels: Sequence[str], roi: Sequence[str] = SENSORIMOTOR_ROI
) -> Tuple[int, ...]:
    """Positions of ``roi`` within ``channels``, in ROI order.

    Returns indices rather than a sliced array so the caller controls copying, and
    so the same mapping can be applied to a data array, a channel-name list and a
    montage position table without them drifting apart.

    Raises
    ------
    ValueError
        If any ROI channel is absent.  A silently shortened ROI would change the
        model input without changing anything visible in the config, so a missing
        electrode has to be loud.
    """
    lookup = {name: i for i, name in enumerate(channels)}
    missing = [name for name in roi if name not in lookup]
    if missing:
        raise ValueError(
            f"ROI channels absent from the montage: {missing}; "
            f"montage has {len(channels)} channels"
        )
    return tuple(lookup[name] for name in roi)

#: Left- and right-hemisphere hand-knob electrodes.  Motor imagery of one hand
#: produces mu/beta event-related desynchronisation over the *contralateral*
#: electrode, so C3 is the informative site for RIGHT-hand imagery and C4 for
#: LEFT-hand imagery.  Used by the left/right PSD comparison figure.
CONTRALATERAL_HAND_CHANNELS: Mapping[str, str] = {
    "LEFT_HAND_IMAGERY": "C4",
    "RIGHT_HAND_IMAGERY": "C3",
}


# --------------------------------------------------------------------------------------
# Task taxonomy
# --------------------------------------------------------------------------------------


class TaskType(str, Enum):
    """The five kinds of run in EEGMMIDB."""

    BASELINE_EYES_OPEN = "baseline_eyes_open"
    BASELINE_EYES_CLOSED = "baseline_eyes_closed"
    UNILATERAL_EXECUTION = "unilateral_execution"      # left / right fist, performed
    UNILATERAL_IMAGERY = "unilateral_imagery"          # left / right fist, imagined
    BILATERAL_EXECUTION = "bilateral_execution"        # both fists / both feet, performed
    BILATERAL_IMAGERY = "bilateral_imagery"            # both fists / both feet, imagined


#: Run number -> task type.  Derived directly from the PhysioNet protocol description.
RUN_TASK_TYPES: Mapping[int, TaskType] = {
    1: TaskType.BASELINE_EYES_OPEN,
    2: TaskType.BASELINE_EYES_CLOSED,
    3: TaskType.UNILATERAL_EXECUTION,
    4: TaskType.UNILATERAL_IMAGERY,
    5: TaskType.BILATERAL_EXECUTION,
    6: TaskType.BILATERAL_IMAGERY,
    7: TaskType.UNILATERAL_EXECUTION,
    8: TaskType.UNILATERAL_IMAGERY,
    9: TaskType.BILATERAL_EXECUTION,
    10: TaskType.BILATERAL_IMAGERY,
    11: TaskType.UNILATERAL_EXECUTION,
    12: TaskType.UNILATERAL_IMAGERY,
    13: TaskType.BILATERAL_EXECUTION,
    14: TaskType.BILATERAL_IMAGERY,
}

#: The runs this project decodes: unilateral (left vs right hand) MOTOR IMAGERY.
UNILATERAL_IMAGERY_RUNS: Tuple[int, ...] = (4, 8, 12)

#: Unilateral motor EXECUTION runs.  Same left/right labelling as the imagery runs
#: but the movement is actually performed.  Not used in Phase 1.
UNILATERAL_EXECUTION_RUNS: Tuple[int, ...] = (3, 7, 11)

#: Bilateral runs (both fists vs both feet).  Not a left/right paradigm.
BILATERAL_IMAGERY_RUNS: Tuple[int, ...] = (6, 10, 14)
BILATERAL_EXECUTION_RUNS: Tuple[int, ...] = (5, 9, 13)

#: Resting-state baseline runs (no T1/T2 events at all).
BASELINE_RUNS: Tuple[int, ...] = (1, 2)


# --------------------------------------------------------------------------------------
# Class labels
# --------------------------------------------------------------------------------------

#: Canonical label for rest.  Shared by every run.
REST = "REST"
LEFT_HAND_IMAGERY = "LEFT_HAND_IMAGERY"
RIGHT_HAND_IMAGERY = "RIGHT_HAND_IMAGERY"
LEFT_HAND_EXECUTION = "LEFT_HAND_EXECUTION"
RIGHT_HAND_EXECUTION = "RIGHT_HAND_EXECUTION"
BOTH_FISTS_IMAGERY = "BOTH_FISTS_IMAGERY"
BOTH_FEET_IMAGERY = "BOTH_FEET_IMAGERY"
BOTH_FISTS_EXECUTION = "BOTH_FISTS_EXECUTION"
BOTH_FEET_EXECUTION = "BOTH_FEET_EXECUTION"

#: The three classes Phase 1 produces, in a fixed order.  The integer index of a
#: class is its position in this tuple and must never be reordered, because it is
#: written into the on-disk ``y`` arrays.
PHASE1_CLASSES: Tuple[str, ...] = (REST, LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY)

#: class label -> integer code used in the stored ``y`` arrays.
CLASS_TO_INDEX: Mapping[str, int] = {name: i for i, name in enumerate(PHASE1_CLASSES)}
INDEX_TO_CLASS: Mapping[int, str] = {i: name for name, i in CLASS_TO_INDEX.items()}

#: The EDF+ annotation descriptions used by EEGMMIDB.
EVENT_CODES: Tuple[str, ...] = ("T0", "T1", "T2")


# Per-task-type resolution of the run-dependent T1/T2 semantics.
_EVENT_MAPS: Mapping[TaskType, Mapping[str, str]] = {
    TaskType.BASELINE_EYES_OPEN: {"T0": REST},
    TaskType.BASELINE_EYES_CLOSED: {"T0": REST},
    TaskType.UNILATERAL_EXECUTION: {
        "T0": REST,
        "T1": LEFT_HAND_EXECUTION,
        "T2": RIGHT_HAND_EXECUTION,
    },
    TaskType.UNILATERAL_IMAGERY: {
        "T0": REST,
        "T1": LEFT_HAND_IMAGERY,
        "T2": RIGHT_HAND_IMAGERY,
    },
    TaskType.BILATERAL_EXECUTION: {
        "T0": REST,
        "T1": BOTH_FISTS_EXECUTION,
        "T2": BOTH_FEET_EXECUTION,
    },
    TaskType.BILATERAL_IMAGERY: {
        "T0": REST,
        "T1": BOTH_FISTS_IMAGERY,
        "T2": BOTH_FEET_IMAGERY,
    },
}


def task_type_for_run(run: int) -> TaskType:
    """Return the :class:`TaskType` of ``run``.

    Raises
    ------
    ValueError
        If ``run`` is not in 1..14.
    """
    try:
        return RUN_TASK_TYPES[run]
    except KeyError:
        raise ValueError(
            f"run {run!r} is not a valid EEGMMIDB run; expected 1..{N_RUNS_PER_SUBJECT}"
        ) from None


def event_map_for_run(run: int) -> Dict[str, str]:
    """Map EDF+ annotation descriptions to class labels **for this specific run**.

    This is the function that prevents the ``T1 == left`` fallacy: the returned
    mapping depends on which task the run belongs to.

    >>> event_map_for_run(4)["T1"]
    'LEFT_HAND_IMAGERY'
    >>> event_map_for_run(6)["T1"]
    'BOTH_FISTS_IMAGERY'
    """
    return dict(_EVENT_MAPS[task_type_for_run(run)])


#: Human-readable task name per run type, used by :func:`resolve_run_task`.
TASK_NAMES: Mapping[TaskType, str] = {
    TaskType.BASELINE_EYES_OPEN: "baseline_eyes_open",
    TaskType.BASELINE_EYES_CLOSED: "baseline_eyes_closed",
    TaskType.UNILATERAL_EXECUTION: "unilateral_fist_execution",
    TaskType.UNILATERAL_IMAGERY: "unilateral_fist_imagery",
    TaskType.BILATERAL_EXECUTION: "bilateral_fist_foot_execution",
    TaskType.BILATERAL_IMAGERY: "bilateral_fist_foot_imagery",
}


def resolve_run_task(run: Union[int, str]) -> Dict[str, Optional[str]]:
    """Fully resolve what a run is and what its annotations mean.

    This is the explicit, inspectable answer to "which runs are unilateral motor
    imagery, and what do T0/T1/T2 mean in this particular run?".  It exists so the
    run type is *checked* rather than inferred from the word "imagery" -- runs 6,
    10 and 14 are also imagery runs, but bilateral ones, and their T1/T2 mean
    something completely different.

    Parameters
    ----------
    run
        Run number (``4``) or run identifier (``"R04"`` / ``"r04"``).

    Returns
    -------
    dict
        ``run``, ``run_number``, ``task``, ``is_unilateral_imagery``, and the
        resolved ``t0``/``t1``/``t2`` labels (``None`` where the code is unused).

    Examples
    --------
    >>> info = resolve_run_task("R04")
    >>> info["task"], info["t1"], info["t2"]
    ('unilateral_fist_imagery', 'LEFT_HAND_IMAGERY', 'RIGHT_HAND_IMAGERY')

    >>> resolve_run_task(6)["t1"]        # bilateral imagery -- NOT left hand
    'BOTH_FISTS_IMAGERY'
    """
    if isinstance(run, str):
        token = run.strip().upper().lstrip("R")
        if not token.isdigit():
            raise ValueError(f"cannot parse run identifier {run!r}; expected e.g. 'R04'")
        number = int(token)
    else:
        number = int(run)

    task_type = task_type_for_run(number)
    mapping = event_map_for_run(number)

    return {
        "run": run_id(number),
        "run_number": number,
        "task": TASK_NAMES[task_type],
        "task_type": task_type.value,
        "is_unilateral_imagery": task_type is TaskType.UNILATERAL_IMAGERY,
        "t0": mapping.get("T0"),
        "t1": mapping.get("T1"),
        "t2": mapping.get("T2"),
    }


def is_unilateral_imagery_run(run: int) -> bool:
    """True iff ``run`` is a left-hand vs right-hand *imagery* run (4, 8 or 12)."""
    return task_type_for_run(run) is TaskType.UNILATERAL_IMAGERY


def subject_id(subject: int) -> str:
    """Format a subject number as the canonical PhysioNet identifier (``S001``)."""
    if not 1 <= subject <= N_SUBJECTS_TOTAL:
        raise ValueError(
            f"subject {subject!r} out of range 1..{N_SUBJECTS_TOTAL}"
        )
    return f"S{subject:03d}"


def run_id(run: int) -> str:
    """Format a run number as it appears in EEGMMIDB filenames (``R04``)."""
    task_type_for_run(run)  # validates
    return f"R{run:02d}"


def edf_filename(subject: int, run: int) -> str:
    """Return the EEGMMIDB EDF filename for ``subject``/``run`` (``S001R04.edf``)."""
    return f"{subject_id(subject)}{run_id(run)}.edf"


# --------------------------------------------------------------------------------------
# Known data-quality caveats
# --------------------------------------------------------------------------------------

#: Subjects that the BCI literature repeatedly reports as inconsistent with the
#: nominal 160 Hz / 64-channel / 14-run format (S088, S092 and S100 are reported as
#: recorded at a different sampling rate, S089 as damaged).
#:
#: IMPORTANT: this list is a *hint*, not an assertion of fact.  It could not be
#: verified against the official PhysioNet record in the environment this pipeline
#: was authored in (physionet.org was unreachable).  The pipeline therefore does
#: **not** rely on it: every recording is validated at load time against
#: :data:`SAMPLING_RATE_HZ` and :data:`N_EEG_CHANNELS` by
#: :func:`~src.data.loader.validate_raw`, and non-conforming recordings are
#: rejected on their measured properties regardless of subject number.
#:
#: Treat this constant as documentation of a *suspicion to check*, and see
#: ``docs/DATASET_ASSUMPTIONS.md``.
REPORTED_INCONSISTENT_SUBJECTS: Tuple[int, ...] = (88, 89, 92, 100)


@dataclass(frozen=True)
class RunSpec:
    """Fully-resolved description of one recording, used for provenance records."""

    subject: int
    run: int

    @property
    def subject_id(self) -> str:
        return subject_id(self.subject)

    @property
    def run_id(self) -> str:
        return run_id(self.run)

    @property
    def task_type(self) -> TaskType:
        return task_type_for_run(self.run)

    @property
    def event_map(self) -> Dict[str, str]:
        return event_map_for_run(self.run)

    @property
    def filename(self) -> str:
        return edf_filename(self.subject, self.run)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.subject_id}{self.run_id} ({self.task_type.value})"


def iter_run_specs(
    subjects: Sequence[int], runs: Sequence[int]
) -> Tuple[RunSpec, ...]:
    """Cartesian product of ``subjects`` x ``runs`` as validated :class:`RunSpec`."""
    return tuple(
        RunSpec(subject=s, run=r)
        for s in subjects
        for r in runs
        # subject_id / task_type_for_run validate and raise on bad input
        if subject_id(s) and task_type_for_run(r)
    )
