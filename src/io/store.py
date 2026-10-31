"""Reproducible on-disk format for the processed epoch dataset.

Layout
------
``data/processed/<SUBJECT>.npz``
    One compressed archive per subject, holding

    ``X``            float32 ``(n_epochs, n_channels, n_times)``, in **volts**
    ``y``            int64   ``(n_epochs,)`` class indices into ``PHASE1_CLASSES``
    ``subject``      int64   ``(n_epochs,)`` subject number, repeated
    ``run``          int64   ``(n_epochs,)`` run number
    ``onset_sample`` int64   ``(n_epochs,)``
    ``channels``     unicode ``(n_channels,)`` channel names, in array order
    ``times``        float64 ``(n_times,)`` epoch time vector in seconds

``data/metadata/epoch_index.csv``
    One row per epoch across the whole dataset, with the full provenance record.
    This is the file to read for any subject-wise split.

``data/metadata/dataset_summary.json``
    Counts, class balance, per-subject breakdown, the exact config used, library
    versions and per-file checksums.

``data/metadata/recordings.csv``
    One row per source recording, from the inspection step.

Why per-subject files
---------------------
The eventual evaluation must split by **subject**, never by window: epochs from one
subject's run are temporally adjacent and highly correlated, so a random window
split leaks near-duplicate data across the train/test boundary and inflates
accuracy by tens of percent.  Storing one file per subject makes the subject the
natural unit of access, and :func:`load_dataset` returns the ``groups`` array that
``GroupKFold`` / ``LeaveOneGroupOut`` need.

Epochs are stored **un-normalised, in volts**.  Per-epoch normalisation is applied
on load (see :func:`load_dataset`) so that the artifact keeps physical units and
stays scientifically inspectable -- a PSD in volts²/Hz is meaningful, one computed
from z-scored data is not.  Because the normalisation statistics are computed
within a single epoch, applying it at load time is equivalent to baking it in, and
leaks nothing.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import DatasetConfig
from src.data.physionet_spec import CLASS_TO_INDEX, PHASE1_CLASSES
from src.preprocessing.filters import normalize_epochs

log = logging.getLogger(__name__)

EPOCH_INDEX_FILENAME = "epoch_index.csv"
SUMMARY_FILENAME = "dataset_summary.json"
RECORDINGS_FILENAME = "recordings.csv"


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------


def subject_archive_path(processed_dir: Path, subject_id: str) -> Path:
    return processed_dir / f"{subject_id}.npz"


def save_subject(
    processed_dir: Path,
    subject_id: str,
    X: np.ndarray,
    metadata: pd.DataFrame,
    channels: Sequence[str],
    times: np.ndarray,
) -> Path:
    """Write one subject's epochs and return the archive path."""
    if X.shape[0] != len(metadata):
        raise ValueError(
            f"{subject_id}: {X.shape[0]} epochs but {len(metadata)} metadata rows"
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    path = subject_archive_path(processed_dir, subject_id)
    np.savez_compressed(
        path,
        X=X.astype(np.float32),
        y=metadata["class_index"].to_numpy(dtype=np.int64),
        subject=metadata["subject"].to_numpy(dtype=np.int64),
        run=metadata["run"].to_numpy(dtype=np.int64),
        onset_sample=metadata["onset_sample"].to_numpy(dtype=np.int64),
        channels=np.asarray(list(channels), dtype="<U8"),
        times=np.asarray(times, dtype=np.float64),
    )
    return path


def save_epoch_index(metadata_dir: Path, index: pd.DataFrame) -> Path:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / EPOCH_INDEX_FILENAME
    index.to_csv(path, index=False)
    return path


def save_recordings(metadata_dir: Path, recordings: pd.DataFrame) -> Path:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / RECORDINGS_FILENAME
    recordings.to_csv(path, index=False)
    return path


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def _library_versions() -> Dict[str, str]:
    import mne
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "mne": mne.__version__,
    }


def build_summary(
    config: DatasetConfig,
    index: pd.DataFrame,
    recordings: pd.DataFrame,
    channels: Sequence[str],
    failures: Sequence[Dict[str, str]] = (),
) -> Dict[str, object]:
    """Assemble the dataset-level metadata summary."""
    by_subject = (
        index.groupby("subject_id")["class_label"]
        .value_counts()
        .unstack(fill_value=0)
        .sort_index()
    )

    return {
        "dataset": "PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB) v1.0.0",
        "dataset_url": "https://physionet.org/content/eegmmidb/1.0.0/",
        "source": config.source,
        "synthetic": config.source == "synthetic",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_fingerprint": config.fingerprint,
        "git_commit": _git_commit(),
        "library_versions": _library_versions(),
        "config": config.to_dict(),
        "classes": list(PHASE1_CLASSES),
        "class_to_index": dict(CLASS_TO_INDEX),
        "n_subjects": int(index["subject_id"].nunique()),
        "n_runs": int(index.groupby(["subject_id", "run_id"]).ngroups),
        "n_epochs": int(len(index)),
        "n_channels": len(channels),
        "channels": list(channels),
        "sampling_rate": float(index["sampling_rate"].iloc[0]) if len(index) else None,
        "epoch_samples": int(index["n_times"].iloc[0]) if len(index) else None,
        "epoch_seconds": (
            float(config.preprocessing.epoch_tmax - config.preprocessing.epoch_tmin)
        ),
        "class_counts": index["class_label"].value_counts().sort_index().to_dict(),
        "epochs_per_subject": index["subject_id"].value_counts().sort_index().to_dict(),
        "class_counts_per_subject": by_subject.to_dict(orient="index"),
        "subjects_processed": sorted(index["subject_id"].unique().tolist()),
        "recordings_processed": int(len(recordings)),
        "failures": list(failures),
        "artifact_flagged_epochs": int(index["artifact_flag"].sum())
        if "artifact_flag" in index
        else 0,
        "notes": [
            "Epochs are stored in volts, un-normalised; normalisation is applied "
            "at load time by load_dataset(normalize=True).",
            "Splits must be made by subject_id, never by random epoch, to avoid "
            "leakage between temporally adjacent windows.",
        ],
    }


def save_summary(metadata_dir: Path, summary: Dict[str, object]) -> Path:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / SUMMARY_FILENAME
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EEGDataset:
    """In-memory view of the processed dataset."""

    X: np.ndarray            # (n_epochs, n_channels, n_times)
    y: np.ndarray            # (n_epochs,) class indices
    groups: np.ndarray       # (n_epochs,) subject numbers -- the split unit
    metadata: pd.DataFrame   # (n_epochs, ...) full provenance
    channels: Tuple[str, ...]
    times: np.ndarray
    classes: Tuple[str, ...] = PHASE1_CLASSES

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def subject_ids(self) -> np.ndarray:
        return self.metadata["subject_id"].to_numpy()

    def class_counts(self) -> Dict[str, int]:
        return self.metadata["class_label"].value_counts().sort_index().to_dict()

    def select_channels(self, names: Sequence[str]) -> "EEGDataset":
        """Return a copy restricted to ``names``, in the order given.

        Use this for the Phase 2 decoder input rather than slicing ``X`` directly:
        it reorders ``channels`` in lockstep, so a later ``list(ds.channels).index("C3")``
        cannot silently point at the wrong row of the array.
        """
        from src.data.physionet_spec import roi_indices

        picks = list(roi_indices(self.channels, list(names)))
        return EEGDataset(
            X=self.X[:, picks, :],
            y=self.y,
            groups=self.groups,
            metadata=self.metadata,
            channels=tuple(self.channels[i] for i in picks),
            times=self.times,
            classes=self.classes,
        )

    def sensorimotor_roi(self) -> "EEGDataset":
        """The documented 21-channel sensorimotor ROI (see ``SENSORIMOTOR_ROI``).

        The full 64-channel dataset is unchanged and still on disk; this is a view
        for model input, which keeps the 64-vs-21 ablation available.
        """
        from src.data.physionet_spec import SENSORIMOTOR_ROI

        return self.select_channels(SENSORIMOTOR_ROI)

    def filter_classes(self, keep: Sequence[str]) -> "EEGDataset":
        """Return a copy restricted to ``keep`` (e.g. drop REST for binary work)."""
        mask = self.metadata["class_label"].isin(list(keep)).to_numpy()
        return EEGDataset(
            X=self.X[mask],
            y=self.y[mask],
            groups=self.groups[mask],
            metadata=self.metadata.loc[mask].reset_index(drop=True),
            channels=self.channels,
            times=self.times,
            classes=self.classes,
        )


def load_dataset(
    config: DatasetConfig,
    *,
    subjects: Optional[Sequence[int]] = None,
    normalize: bool = True,
) -> EEGDataset:
    """Load the processed dataset from disk.

    Parameters
    ----------
    subjects
        Restrict to these subject numbers.  Defaults to every subject present.
    normalize
        Apply the per-epoch normalisation named in the config.  Safe to leave on:
        the statistics come from within one epoch, so nothing leaks across a
        future train/test boundary.
    """
    paths = config.resolved_paths
    index_path = paths.metadata / EPOCH_INDEX_FILENAME
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found -- run `python -m src.cli.build_dataset` first"
        )

    index = pd.read_csv(index_path)
    if subjects is not None:
        index = index[index["subject"].isin(list(subjects))].reset_index(drop=True)
    if index.empty:
        raise ValueError("no epochs match the requested subjects")

    blocks: List[np.ndarray] = []
    frames: List[pd.DataFrame] = []
    channels: Optional[Tuple[str, ...]] = None
    times: Optional[np.ndarray] = None

    for subject_id, rows in index.groupby("subject_id", sort=True):
        archive = subject_archive_path(paths.processed, str(subject_id))
        if not archive.exists():
            raise FileNotFoundError(f"{archive} referenced by the index is missing")

        with np.load(archive, allow_pickle=False) as bundle:
            X = bundle["X"]
            file_channels = tuple(str(c) for c in bundle["channels"])
            file_times = bundle["times"]

        if len(rows) != X.shape[0]:
            raise ValueError(
                f"{subject_id}: index lists {len(rows)} epochs but the archive holds "
                f"{X.shape[0]} -- the dataset is inconsistent, rebuild it"
            )

        if channels is None:
            channels, times = file_channels, file_times
        elif file_channels != channels:
            raise ValueError(f"{subject_id}: channel order differs from other subjects")

        blocks.append(X)
        frames.append(rows.reset_index(drop=True))

    X = np.concatenate(blocks, axis=0)
    metadata = pd.concat(frames, ignore_index=True)

    if normalize:
        X = normalize_epochs(X.astype(np.float64), config.preprocessing.normalization)

    return EEGDataset(
        X=X.astype(np.float32),
        y=metadata["class_index"].to_numpy(dtype=np.int64),
        groups=metadata["subject"].to_numpy(dtype=np.int64),
        metadata=metadata,
        channels=channels or (),
        times=times if times is not None else np.array([]),
    )


def load_summary(config: DatasetConfig) -> Dict[str, object]:
    path = config.resolved_paths.metadata / SUMMARY_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- build the dataset first")
    return json.loads(path.read_text(encoding="utf-8"))
