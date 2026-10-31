"""End-to-end orchestration: EDF files in, processed epoch dataset out.

    for each subject:
        for each run in (4, 8, 12):
            resolve EDF  ->  load + validate  ->  preprocess  ->  epoch
        concatenate the subject's runs  ->  data/processed/<SUBJECT>.npz
    concatenate every subject's metadata  ->  data/metadata/epoch_index.csv
                                          ->  data/metadata/dataset_summary.json

Nothing in this module knows what ``T1`` means; it delegates to
:mod:`src.data.physionet_spec` for every dataset fact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import DatasetConfig
from src.data.loader import (
    RecordingValidationError,
    load_raw,
    summarize_recording,
)
from src.data.physionet_spec import RunSpec, iter_run_specs
from src.io import store
from src.preprocessing.epoching import RunEpochs, extract_epochs
from src.preprocessing.filters import preprocess_raw

log = logging.getLogger(__name__)


@dataclass
class BuildReport:
    """What a pipeline run produced and what it could not."""

    config: DatasetConfig
    n_subjects: int = 0
    n_recordings: int = 0
    n_epochs: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    archives: List[Path] = field(default_factory=list)
    epoch_index_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    recordings_path: Optional[Path] = None
    failures: List[Dict[str, str]] = field(default_factory=list)

    def render(self) -> str:  # pragma: no cover - reporting only
        lines = [
            "Phase 1 dataset build",
            f"  source             : {self.config.source}",
            f"  config fingerprint : {self.config.fingerprint}",
            f"  subjects           : {self.n_subjects}",
            f"  recordings         : {self.n_recordings}",
            f"  epochs             : {self.n_epochs}",
            "  class balance      : "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.class_counts.items())),
        ]
        if self.epoch_index_path:
            lines.append(f"  epoch index        : {self.epoch_index_path}")
        if self.summary_path:
            lines.append(f"  summary            : {self.summary_path}")
        if self.failures:
            lines.append(f"  FAILURES           : {len(self.failures)}")
            for failure in self.failures:
                lines.append(f"    - {failure['recording']}: {failure['error']}")
        return "\n".join(lines)


def process_run(spec: RunSpec, config: DatasetConfig) -> RunEpochs:
    """Load, preprocess and epoch a single recording."""
    raw = load_raw(spec, config)
    raw = preprocess_raw(raw, config.preprocessing)
    return extract_epochs(raw, spec, config)


def process_subject(
    subject: int, config: DatasetConfig
) -> Tuple[Optional[np.ndarray], pd.DataFrame, List[Dict[str, str]], Optional[Tuple[str, ...]], Optional[np.ndarray]]:
    """Process every configured run of one subject.

    Returns ``(X, metadata, failures, channels, times)``.  ``X`` is ``None`` when
    the subject produced no usable epochs.
    """
    blocks: List[np.ndarray] = []
    frames: List[pd.DataFrame] = []
    failures: List[Dict[str, str]] = []
    channels: Optional[Tuple[str, ...]] = None
    times: Optional[np.ndarray] = None

    for spec in iter_run_specs([subject], config.runs):
        try:
            result = process_run(spec, config)
        except (RecordingValidationError, Exception) as exc:  # noqa: BLE001
            if config.strict:
                raise
            log.warning("skipping %s: %s", spec, exc)
            failures.append({"recording": str(spec), "error": str(exc)})
            continue

        run_channels = tuple(result.epochs.ch_names)
        if channels is None:
            channels, times = run_channels, result.epochs.times
        elif run_channels != channels:
            message = "channel order differs from earlier runs of the same subject"
            if config.strict:
                raise RuntimeError(f"{spec}: {message}")
            log.warning("skipping %s: %s", spec, message)
            failures.append({"recording": str(spec), "error": message})
            continue

        blocks.append(result.data)
        frames.append(result.metadata)

    if not blocks:
        return None, pd.DataFrame(), failures, channels, times

    return (
        np.concatenate(blocks, axis=0),
        pd.concat(frames, ignore_index=True),
        failures,
        channels,
        times,
    )


def build_dataset(config: DatasetConfig) -> BuildReport:
    """Run the full pipeline and write every artifact."""
    paths = config.resolved_paths
    paths.mkdirs()

    report = BuildReport(config=config)
    index_frames: List[pd.DataFrame] = []
    recording_rows: List[Dict[str, object]] = []
    channels: Optional[Tuple[str, ...]] = None

    for subject in config.subjects:
        X, metadata, failures, subject_channels, times = process_subject(subject, config)
        report.failures.extend(failures)

        if X is None or metadata.empty:
            log.warning("subject %03d produced no epochs", subject)
            continue

        if channels is None:
            channels = subject_channels
        elif subject_channels != channels:
            message = f"S{subject:03d}: channel order differs from earlier subjects"
            if config.strict:
                raise RuntimeError(message)
            log.warning("%s -- skipping subject", message)
            report.failures.append({"recording": f"S{subject:03d}", "error": message})
            continue

        subject_id = metadata["subject_id"].iloc[0]
        archive = store.save_subject(
            paths.processed, subject_id, X, metadata, subject_channels or (), times
        )
        report.archives.append(archive)
        index_frames.append(metadata)
        report.n_subjects += 1
        report.n_recordings += int(metadata["run_id"].nunique())

        # Inspection row per recording, for data/metadata/recordings.csv.
        for spec in iter_run_specs([subject], config.runs):
            if spec.run_id not in set(metadata["run_id"]):
                continue
            try:
                recording_rows.append(summarize_recording(spec, config).to_row())
            except Exception as exc:  # noqa: BLE001 - inspection must not break a build
                log.warning("could not summarise %s: %s", spec, exc)

    if not index_frames:
        raise RuntimeError(
            "the pipeline produced no epochs at all; see the logged failures above"
        )

    index = pd.concat(index_frames, ignore_index=True)
    index.insert(0, "epoch_uid", [f"E{i:07d}" for i in range(len(index))])

    recordings = pd.DataFrame.from_records(recording_rows)
    report.epoch_index_path = store.save_epoch_index(paths.metadata, index)
    report.recordings_path = store.save_recordings(paths.metadata, recordings)
    summary = store.build_summary(
        config, index, recordings, channels or (), report.failures
    )
    report.summary_path = store.save_summary(paths.metadata, summary)

    report.n_epochs = len(index)
    report.class_counts = index["class_label"].value_counts().sort_index().to_dict()
    return report
