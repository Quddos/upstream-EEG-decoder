"""Loading and inspection of EEGMMIDB EDF+ recordings.

Responsibilities
----------------
* Resolve an EDF path for a (subject, run) pair, downloading from PhysioNet or
  generating a synthetic fixture depending on the configured source.
* Normalise the EDF channel labels to montage-compatible 10-05 names.
* Validate every recording against the properties declared in
  :mod:`src.data.physionet_spec` *on measured values*, so a non-conforming
  recording is rejected on its own merits rather than on a hard-coded blacklist.
* Report the annotations of a run resolved through the run-dependent event map.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd

from src.config import DatasetConfig
from src.data.physionet_spec import (
    DATASET_VERSION,
    EDF_CHANNEL_NAMES,
    EVENT_CODES,
    N_EEG_CHANNELS,
    PHYSIONET_RECORD_URL,
    SAMPLING_RATE_HZ,
    RunSpec,
    edf_filename,
    event_map_for_run,
    run_id,
    subject_id,
)

log = logging.getLogger(__name__)


class RecordingValidationError(RuntimeError):
    """Raised when a recording does not conform to the EEGMMIDB specification."""


class DataSourceMismatchError(RuntimeError):
    """Raised when a recording's provenance disagrees with the configured source.

    Real and synthetic recordings share a filename (``S001R04.edf``).  Source-scoped
    paths keep them in separate trees, but a stray copy -- or a hand-placed file --
    could still put a fixture where a real recording belongs.  Reporting simulated
    data as ``source: physionet`` would be a scientific integrity failure, so it is
    made impossible rather than merely unlikely.
    """


# --------------------------------------------------------------------------------------
# Channel names
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _montage_lookup(montage: str) -> Dict[str, str]:
    """UPPERCASE electrode name -> canonical montage spelling."""
    positions = mne.channels.make_standard_montage(montage)
    return {name.upper(): name for name in positions.ch_names}


def standardize_channel_names(
    names: Sequence[str], montage: str = "standard_1005"
) -> List[str]:
    """Convert EDF+ channel labels to montage-compatible 10-05 names.

    EEGMMIDB stores labels padded to four characters with ``.`` and with
    inconsistent capitalisation (``Fc5.``, ``Cp1.``, ``Iz..``).  Neither form
    matches a standard montage, so channel positions -- and therefore every
    topographic plot and any spatial filter -- would silently fail without this
    step.

    The mapping is derived from the montage itself by case-insensitive match
    rather than from a hand-written table, so it cannot drift out of sync with
    MNE's electrode definitions.

    >>> standardize_channel_names(["Fc5.", "Cz..", "Iz.."])
    ['FC5', 'Cz', 'Iz']
    """
    lookup = _montage_lookup(montage)
    out: List[str] = []
    for raw_name in names:
        key = raw_name.strip().rstrip(".").replace(".", "").upper()
        out.append(lookup.get(key, raw_name.strip().rstrip(".")))
    return out


# --------------------------------------------------------------------------------------
# File resolution
# --------------------------------------------------------------------------------------


def local_edf_path(spec: RunSpec, raw_dir: Path) -> Path:
    """Canonical on-disk location of one recording inside ``data/raw``."""
    return raw_dir / spec.subject_id / spec.filename


def resolve_edf(spec: RunSpec, config: DatasetConfig) -> Path:
    """Return a local path to the EDF for ``spec``, fetching or generating it.

    * If the file already exists under ``data/raw`` it is reused (the pipeline is
      idempotent and never re-downloads).
    * ``source="physionet"`` downloads the official record via
      :mod:`mne.datasets.eegbci` and copies it into ``data/raw``.
    * ``source="synthetic"`` writes a structurally-identical local fixture.
    """
    paths = config.resolved_paths
    target = local_edf_path(spec, paths.raw)
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)

    if config.source == "synthetic":
        from src.data.synthetic import write_synthetic_run

        return write_synthetic_run(spec, target, seed=config.seed)

    from src.data.download import download_run

    return download_run(spec, target)


def is_synthetic_recording(raw: mne.io.BaseRaw) -> bool:
    """True if this recording was produced by :mod:`src.data.synthetic`.

    The fixture writer stamps :data:`~src.data.synthetic.SYNTHETIC_MARKER` into the
    EDF patient-identification field, which survives the EDF round-trip and is
    readable from the header alone.
    """
    from src.data.synthetic import SYNTHETIC_MARKER

    info = raw.info.get("subject_info") or {}
    his_id = str(info.get("his_id", "") or "")
    return SYNTHETIC_MARKER in his_id


def verify_source(raw: mne.io.BaseRaw, spec: RunSpec, config: DatasetConfig) -> None:
    """Assert the file's provenance matches the configured source.

    Guards both directions:

    * ``source="physionet"`` reading a fixture -> simulated data would be reported
      as real. This is the dangerous direction.
    * ``source="synthetic"`` reading a real recording -> a "unit test" would
      silently depend on a 2.5 MB download.
    """
    synthetic_file = is_synthetic_recording(raw)
    path = Path(raw.filenames[0]) if raw.filenames else "?"

    if config.source == "physionet" and synthetic_file:
        raise DataSourceMismatchError(
            f"{spec}: {path} carries the synthetic-fixture marker but the pipeline "
            f"is configured with source='physionet'. Refusing to report simulated "
            f"data as real. Synthetic fixtures belong under data/synthetic/raw/."
        )
    if config.source == "synthetic" and not synthetic_file:
        raise DataSourceMismatchError(
            f"{spec}: {path} does not carry the synthetic-fixture marker but the "
            f"pipeline is configured with source='synthetic'. This looks like a real "
            f"recording; real data belongs under data/raw/ and should be built with "
            f"--source physionet."
        )


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, for provenance records."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_raw(
    spec: RunSpec,
    config: DatasetConfig,
    *,
    preload: bool = True,
    set_montage: bool = True,
) -> mne.io.BaseRaw:
    """Load one EEGMMIDB recording as an MNE :class:`~mne.io.Raw`.

    The returned object has standardised channel names, an attached montage, and
    has been checked against the dataset specification.
    """
    path = resolve_edf(spec, config)
    raw = mne.io.read_raw_edf(path, preload=preload, verbose="ERROR")

    rename = dict(
        zip(raw.ch_names, standardize_channel_names(raw.ch_names, config.preprocessing.montage))
    )
    raw.rename_channels(rename)
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    if set_montage:
        montage = mne.channels.make_standard_montage(config.preprocessing.montage)
        # on_missing="warn": a recording with an unexpected label still loads, and
        # validate_raw() below is what decides whether to keep it.
        raw.set_montage(montage, on_missing="warn", verbose="ERROR")

    verify_source(raw, spec, config)
    validate_raw(raw, spec)
    return raw


def validate_raw(raw: mne.io.BaseRaw, spec: RunSpec) -> None:
    """Check a recording against the EEGMMIDB specification.

    Validation is done on *measured* properties.  This is what protects the
    pipeline from the subjects that are reported to deviate from the nominal
    format: they are caught here by their actual sampling rate or channel count,
    with no need to trust a blacklist.

    Raises
    ------
    RecordingValidationError
        If the sampling rate, channel count, or annotation vocabulary is wrong.
    """
    problems: List[str] = []

    if not np.isclose(raw.info["sfreq"], SAMPLING_RATE_HZ):
        problems.append(
            f"sampling rate is {raw.info['sfreq']} Hz, expected {SAMPLING_RATE_HZ} Hz"
        )

    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    if n_eeg != N_EEG_CHANNELS:
        problems.append(f"{n_eeg} EEG channels, expected {N_EEG_CHANNELS}")

    descriptions = set(raw.annotations.description)
    unexpected = descriptions - set(EVENT_CODES)
    if unexpected:
        problems.append(f"unexpected annotation descriptions: {sorted(unexpected)}")

    if raw.n_times == 0:
        problems.append("recording contains no samples")

    if problems:
        raise RecordingValidationError(
            f"{spec} ({Path(raw.filenames[0]).name if raw.filenames else '?'}): "
            + "; ".join(problems)
        )


# --------------------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordingSummary:
    """Everything the exploration step reports about a single recording."""

    subject_id: str
    run_id: str
    task_type: str
    path: str
    sha256: str
    # --- provenance ---
    source: str                      # "physionet" or "synthetic"
    dataset_version: str
    download_source: str
    file_bytes: int
    acquisition_date: Optional[str]  # from the EDF header, if recorded
    sampling_rate: float
    n_channels: int
    n_eeg_channels: int
    n_samples: int
    duration_s: float
    n_annotations: int
    event_counts: Dict[str, int]
    class_counts: Dict[str, int]
    channel_names: Tuple[str, ...]

    def to_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "subject_id": self.subject_id,
            "run_id": self.run_id,
            "task_type": self.task_type,
            "source": self.source,
            "dataset_version": self.dataset_version,
            "download_source": self.download_source,
            "acquisition_date": self.acquisition_date,
            "file_bytes": self.file_bytes,
            "sampling_rate": self.sampling_rate,
            "n_channels": self.n_channels,
            "n_eeg_channels": self.n_eeg_channels,
            "n_samples": self.n_samples,
            "duration_s": round(self.duration_s, 3),
            "n_annotations": self.n_annotations,
            "sha256": self.sha256,
            "path": self.path,
        }
        for code in EVENT_CODES:
            row[f"n_{code}"] = self.event_counts.get(code, 0)
        for label, count in sorted(self.class_counts.items()):
            row[f"n_{label}"] = count
        return row


def summarize_recording(
    spec: RunSpec, config: DatasetConfig, raw: Optional[mne.io.BaseRaw] = None
) -> RecordingSummary:
    """Inspect channels, sampling rate, annotations and duration of one recording."""
    if raw is None:
        raw = load_raw(spec, config, preload=False)
    path = Path(raw.filenames[0]) if raw.filenames else resolve_edf(spec, config)

    descriptions = list(raw.annotations.description)
    event_counts = {code: descriptions.count(code) for code in EVENT_CODES}

    mapping = event_map_for_run(spec.run)
    class_counts: Dict[str, int] = {}
    for code in descriptions:
        label = mapping.get(code)
        if label is not None:
            class_counts[label] = class_counts.get(label, 0) + 1

    meas_date = raw.info.get("meas_date")

    return RecordingSummary(
        subject_id=spec.subject_id,
        run_id=spec.run_id,
        task_type=spec.task_type.value,
        path=str(path),
        sha256=sha256(path),
        source=config.source,
        dataset_version=DATASET_VERSION if config.source == "physionet" else "n/a (simulated)",
        download_source=(
            PHYSIONET_RECORD_URL if config.source == "physionet" else "locally generated fixture"
        ),
        file_bytes=path.stat().st_size,
        acquisition_date=meas_date.isoformat() if meas_date is not None else None,
        sampling_rate=float(raw.info["sfreq"]),
        n_channels=len(raw.ch_names),
        n_eeg_channels=len(mne.pick_types(raw.info, eeg=True)),
        n_samples=int(raw.n_times),
        duration_s=float(raw.n_times) / float(raw.info["sfreq"]),
        n_annotations=len(raw.annotations),
        event_counts=event_counts,
        class_counts=class_counts,
        channel_names=tuple(raw.ch_names),
    )


def annotations_table(spec: RunSpec, raw: mne.io.BaseRaw) -> pd.DataFrame:
    """Annotation timeline of one run with class labels resolved for that run.

    Columns: ``onset_time``, ``duration``, ``event_code``, ``class_label``,
    ``onset_sample``.
    """
    mapping = event_map_for_run(spec.run)
    sfreq = float(raw.info["sfreq"])
    first_samp_time = raw.first_samp / sfreq

    records = []
    for onset, duration, description in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        # Annotation onsets are absolute; subtract first_samp to get a time
        # relative to the start of the data array.
        relative = float(onset) - first_samp_time
        records.append(
            {
                "subject_id": spec.subject_id,
                "run_id": spec.run_id,
                "onset_time": relative,
                "duration": float(duration),
                "event_code": str(description),
                "class_label": mapping.get(str(description)),
                "onset_sample": int(round(relative * sfreq)),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "subject_id",
            "run_id",
            "onset_time",
            "duration",
            "event_code",
            "class_label",
            "onset_sample",
        ],
    )
