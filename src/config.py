"""Configuration for the Phase 1 EEG dataset pipeline.

Every knob the pipeline honours lives here.  Scaling from the 5-subject development
subset to all 109 subjects is a one-line change to :attr:`DatasetConfig.subjects`
(or ``--subjects all`` on the command line); no other code needs to be touched.

The config is serialised into every artifact the pipeline writes, and hashed, so a
processed dataset can always be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from src.data.physionet_spec import (
    N_SUBJECTS_TOTAL,
    POWERLINE_FREQ_HZ,
    SAMPLING_RATE_HZ,
    UNILATERAL_IMAGERY_RUNS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Development subset.  Five subjects is enough to exercise every branch of the
#: pipeline (multi-subject aggregation, per-run event mapping, class balance
#: checks) while keeping a full run to well under a minute.
DEV_SUBJECTS: Tuple[int, ...] = (1, 2, 3, 4, 5)

#: Every subject in the release.  Use this to scale up.
ALL_SUBJECTS: Tuple[int, ...] = tuple(range(1, N_SUBJECTS_TOTAL + 1))


#: Directory prefix applied to every artifact built from synthetic fixtures.
#: Real PhysioNet artifacts live at the top level (``data/raw``), simulated ones
#: under a sibling tree (``data/synthetic/raw``).  The two can therefore never
#: overwrite one another, and no build can silently mix them.
SYNTHETIC_SUBDIR = "synthetic"


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem layout.  All paths are absolute once :meth:`resolve` has run."""

    root: Path = PROJECT_ROOT
    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    metadata: Path = Path("data/metadata")
    figures: Path = Path("reports/figures")

    def resolve(self) -> "PathsConfig":
        """Return a copy with every relative path anchored at :attr:`root`."""
        root = self.root.resolve()
        return replace(
            self,
            root=root,
            raw=root / self.raw,
            processed=root / self.processed,
            metadata=root / self.metadata,
            figures=root / self.figures,
        )

    def for_source(self, source: str) -> "PathsConfig":
        """Scope the layout to a data source.

        ``"physionet"`` keeps the canonical locations.  ``"synthetic"`` diverts
        everything into a parallel tree:

            data/raw/S001/S001R04.edf              <- real recording
            data/synthetic/raw/S001/S001R04.edf    <- simulated fixture

        Real and simulated recordings share a filename, so without this split a
        synthetic fixture sitting in ``data/raw`` would be picked up by a
        ``--source physionet`` build and silently reported as real data.
        """
        if source != "synthetic":
            return self
        return replace(
            self,
            raw=Path("data") / SYNTHETIC_SUBDIR / "raw",
            processed=Path("data") / SYNTHETIC_SUBDIR / "processed",
            metadata=Path("data") / SYNTHETIC_SUBDIR / "metadata",
            figures=Path("reports") / f"figures_{SYNTHETIC_SUBDIR}",
        )

    def mkdirs(self) -> None:
        for p in (self.raw, self.processed, self.metadata, self.figures):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Signal-processing parameters.

    Defaults are the conservative, widely-used choices for sensorimotor-rhythm
    motor imagery.  Each one is justified in ``docs/PREPROCESSING_DECISIONS.md``;
    the short version is given inline.
    """

    #: Band-pass passband in Hz.  8-30 Hz spans the mu (8-13 Hz) and beta
    #: (13-30 Hz) sensorimotor rhythms whose event-related desynchronisation
    #: carries the left/right hand information.  It also removes drift, EOG
    #: (<4 Hz) and most EMG (>40 Hz) without subject-specific artifact rejection.
    l_freq: float = 8.0
    h_freq: float = 30.0

    #: Zero-phase FIR.  Zero-phase matters because we epoch on event onsets and a
    #: causal filter would smear the ERD onset in time.
    filter_method: str = "fir"
    fir_design: str = "firwin"
    filter_phase: str = "zero"
    #: 'auto' lets MNE pick transition bandwidths of max(l_freq*0.25, 2) and
    #: min(h_freq*0.25, ...) Hz, which is reproducible given the passband.
    filter_length: str = "auto"

    #: Notch filtering is OFF by default: the 8-30 Hz band-pass already places
    #: 60 Hz deep in the stopband, so a notch would be redundant signal surgery.
    #: Kept configurable for wider passbands (e.g. a future broadband variant).
    notch_enabled: bool = False
    notch_freqs: Tuple[float, ...] = (POWERLINE_FREQ_HZ,)

    #: Common average reference.  EEGMMIDB is recorded against a single physical
    #: reference; CAR removes the reference-dependent common component and
    #: sharpens the spatial contrast between C3 and C4, which is what a
    #: left/right hand decoder relies on.
    common_average_reference: bool = True

    #: Epoch window relative to event onset, in seconds.  The cue/task period in
    #: EEGMMIDB is ~4.1 s, so [0, 4) captures the task without bleeding into the
    #: following rest period.  ``include_tmax=False`` is used so the window is
    #: exactly ``4.0 * 160 = 640`` samples -- a clean, fixed length for EEGNet.
    epoch_tmin: float = 0.0
    epoch_tmax: float = 4.0
    include_tmax: bool = False

    #: Baseline correction window, or ``None`` to disable.  Disabled by default:
    #: the band-pass already removes DC and slow drift, and any pre-onset window
    #: would fall inside the *previous* trial, importing its class into the
    #: current epoch.  That is a subtle form of label leakage.
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None

    #: Amplitude rejection threshold in volts, or ``None`` for no rejection.
    #: Disabled by default -- the brief is explicit that artifacts should not be
    #: aggressively removed.  A per-epoch peak-to-peak flag is *recorded* in the
    #: metadata instead (see ``reject_log_threshold_v``) so a later phase can
    #: filter without the data having been silently discarded here.
    reject_peak_to_peak_v: Optional[float] = None
    #: Epochs whose peak-to-peak amplitude exceeds this are flagged (not dropped).
    reject_log_threshold_v: float = 150e-6

    #: Per-epoch, per-channel z-scoring.  Applied at *load* time rather than
    #: baked into the stored arrays, so the on-disk artifact keeps physical units
    #: (volts) and remains inspectable -- PSD plots need real units.  Per-epoch
    #: statistics are computed within a single epoch only, so this introduces no
    #: cross-epoch or cross-subject leakage.
    normalization: str = "zscore_per_channel"

    #: Keep every EEG channel by default.  Set to a tuple of montage-standard
    #: names (e.g. ``SENSORIMOTOR_CHANNELS``) to restrict the montage.
    channel_subset: Optional[Tuple[str, ...]] = None

    #: Montage used for topographic plots and channel-position sanity checks.
    montage: str = "standard_1005"

    @property
    def expected_epoch_samples(self) -> int:
        """Number of samples an epoch must contain, given a 160 Hz recording."""
        n = (self.epoch_tmax - self.epoch_tmin) * SAMPLING_RATE_HZ
        return int(round(n)) + (1 if self.include_tmax else 0)


@dataclass(frozen=True)
class DatasetConfig:
    """Top-level pipeline configuration."""

    #: *** Change this one field to scale up. ***  ``DEV_SUBJECTS`` for the
    #: 5-subject development subset, ``ALL_SUBJECTS`` for the full release.
    subjects: Tuple[int, ...] = DEV_SUBJECTS

    #: Runs to process.  Defaults to the unilateral motor-imagery runs (4, 8, 12).
    runs: Tuple[int, ...] = UNILATERAL_IMAGERY_RUNS

    #: Include T0 (rest) epochs as a third class.  Phase 3's neuro-state engine
    #: needs a neutral state, so rest is kept.
    include_rest: bool = True

    #: Cap on rest epochs kept per run.  Rest is roughly as frequent as the task
    #: events combined, which would make it a 50% majority class.  ``None`` keeps
    #: all of them; the default keeps a per-run count close to one task class.
    max_rest_epochs_per_run: Optional[int] = 8

    #: Where the EDF files come from.  ``"physionet"`` downloads the real dataset;
    #: ``"synthetic"`` generates structurally-identical fixtures locally, which is
    #: what makes the pipeline runnable and testable without network access.
    source: str = "physionet"

    #: Abort the whole run if any single recording fails, rather than skipping it.
    strict: bool = False

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    #: Seed for every stochastic step (synthetic data generation, rest subsampling).
    seed: int = 20260815

    def __post_init__(self) -> None:
        if self.source not in {"physionet", "synthetic"}:
            raise ValueError(
                f"source must be 'physionet' or 'synthetic', got {self.source!r}"
            )
        if not self.subjects:
            raise ValueError("subjects must not be empty")
        if not self.runs:
            raise ValueError("runs must not be empty")

    @property
    def resolved_paths(self) -> PathsConfig:
        """Absolute, source-scoped filesystem layout.

        Always use this rather than ``config.paths.resolve()``: it applies the
        real/synthetic split, which is what keeps the two datasets apart on disk.
        """
        return self.paths.for_source(self.source).resolve()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view of the config, used for provenance records."""

        def _encode(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, dict):
                return {k: _encode(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_encode(v) for v in obj]
            return obj

        return _encode(asdict(self))

    @property
    def fingerprint(self) -> str:
        """Stable 16-hex-character hash of everything that affects the output.

        Paths are excluded so that moving the project directory does not
        invalidate an otherwise identical dataset.
        """
        payload = self.to_dict()
        payload.pop("paths", None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def dev_config(**overrides: Any) -> DatasetConfig:
    """The 5-subject development configuration, with optional field overrides."""
    return replace(DatasetConfig(), **overrides) if overrides else DatasetConfig()


def full_config(**overrides: Any) -> DatasetConfig:
    """The full 109-subject configuration, with optional field overrides."""
    return replace(DatasetConfig(subjects=ALL_SUBJECTS), **overrides)


def parse_subjects(value: str) -> Tuple[int, ...]:
    """Parse a CLI subject specification.

    Accepts ``all``, ``dev``, a comma-separated list (``1,2,7``), inclusive ranges
    (``1-5``), or any combination (``1-5,20,88``).
    """
    token = value.strip().lower()
    if token == "all":
        return ALL_SUBJECTS
    if token == "dev":
        return DEV_SUBJECTS

    subjects: list[int] = []
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"invalid subject range {part!r}: {lo} > {hi}")
            subjects.extend(range(lo, hi + 1))
        else:
            subjects.append(int(part))

    out = tuple(sorted(set(subjects)))
    if not out:
        raise ValueError(f"no subjects parsed from {value!r}")
    bad = [s for s in out if not 1 <= s <= N_SUBJECTS_TOTAL]
    if bad:
        raise ValueError(f"subjects out of range 1..{N_SUBJECTS_TOTAL}: {bad}")
    return out


def parse_runs(value: str) -> Tuple[int, ...]:
    """Parse a CLI run specification (``imagery``, ``execution``, or ``4,8,12``)."""
    token = value.strip().lower()
    if token in {"imagery", "mi"}:
        return UNILATERAL_IMAGERY_RUNS
    runs = tuple(sorted({int(p) for p in token.split(",") if p.strip()}))
    if not runs:
        raise ValueError(f"no runs parsed from {value!r}")
    return runs
