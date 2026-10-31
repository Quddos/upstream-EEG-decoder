"""Event resolution and fixed-length epoch extraction.

The critical correctness property enforced here is that ``T1``/``T2`` are decoded
through :func:`~src.data.physionet_spec.event_map_for_run`, i.e. **per run**.  No
code path in this module can produce a left/right label without going through that
function.

Every epoch carries a full provenance row (subject, run, event code, class,
onset sample, onset time, window bounds).  MNE's :class:`~mne.Epochs` metadata
mechanism is used to attach it, which guarantees the rows stay aligned with the
data array even when MNE drops epochs that run past the end of the recording.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd

from src.config import DatasetConfig
from src.data.physionet_spec import (
    CLASS_TO_INDEX,
    PHASE1_CLASSES,
    REST,
    RunSpec,
    event_map_for_run,
)

log = logging.getLogger(__name__)


class EpochingError(RuntimeError):
    """Raised when epochs cannot be extracted from a recording."""


#: Columns every epoch metadata table must contain, in order.  The first ten are
#: the fields the Phase 1 brief requires; the rest are provenance and diagnostics.
METADATA_COLUMNS: Tuple[str, ...] = (
    "subject_id",
    "run_id",
    "event_code",
    "class_label",
    "onset_sample",
    "onset_time",
    "sampling_rate",
    "channel_count",
    "epoch_start",
    "epoch_end",
    # --- additional provenance ---
    "subject",
    "run",
    "task_type",
    "class_index",
    "epoch_tmin",
    "epoch_tmax",
    "epoch_start_sample",
    "epoch_end_sample",
    "n_times",
    "source",
)


def build_events(
    raw: mne.io.BaseRaw, spec: RunSpec, keep_classes: Sequence[str]
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build an MNE events array whose codes are Phase 1 class indices.

    The annotation description -> class label step uses the run-specific map, so
    ``T1`` becomes ``LEFT_HAND_IMAGERY`` in run 4 and ``BOTH_FISTS_IMAGERY`` in
    run 6.  Annotations whose class is not in ``keep_classes`` are dropped.

    Returns
    -------
    events
        ``(n_events, 3)`` array; column 2 holds the class index.
    event_id
        ``{class_label: class_index}`` for the classes actually present.
    """
    run_map = event_map_for_run(spec.run)
    keep = set(keep_classes)

    # Refuse the run outright if it produces labels outside the Phase 1 class set.
    # This is checked *before* the keep-list filter on purpose: filtering first
    # would quietly discard BOTH_FISTS/BOTH_FEET (or the execution labels) and
    # emit a REST-only dataset for run 6, which looks like success but is not.
    foreign = sorted(
        {label for label in run_map.values() if label not in CLASS_TO_INDEX}
    )
    if foreign:
        raise EpochingError(
            f"run {spec.run} is a {spec.task_type.value} run and produces "
            f"{foreign}, which are not a Phase 1 class; Phase 1 accepts only the "
            f"unilateral motor-imagery runs {(4, 8, 12)}"
        )

    # description -> class index, restricted to the classes we want.
    description_to_code: Dict[str, int] = {}
    event_id: Dict[str, int] = {}
    for description, label in run_map.items():
        if label not in keep:
            continue
        code = CLASS_TO_INDEX[label]
        description_to_code[description] = code
        event_id[label] = code

    present = set(raw.annotations.description)
    usable = {d: c for d, c in description_to_code.items() if d in present}
    if not usable:
        raise EpochingError(
            f"{spec}: none of the expected annotations {sorted(description_to_code)} "
            f"are present; found {sorted(present)}"
        )

    events, _ = mne.events_from_annotations(raw, event_id=usable, verbose="ERROR")
    event_id = {
        label: code for label, code in event_id.items() if code in set(events[:, 2])
    }
    return events, event_id


def subsample_rest(
    events: np.ndarray, max_per_run: Optional[int], seed: int
) -> np.ndarray:
    """Thin the REST events down to ``max_per_run``, preserving temporal spread.

    Rest occurs between every pair of task trials, so keeping all of it makes REST
    roughly half the dataset.  Rather than sampling randomly -- which would cluster
    -- rest events are taken at evenly spaced positions through the run, so the
    retained rest still covers the whole recording (early trials, late trials, and
    the fatigue drift in between).
    """
    if max_per_run is None:
        return events

    rest_code = CLASS_TO_INDEX[REST]
    rest_rows = np.flatnonzero(events[:, 2] == rest_code)
    if len(rest_rows) <= max_per_run:
        return events

    picks = np.linspace(0, len(rest_rows) - 1, num=max_per_run).round().astype(int)
    keep_rest = set(rest_rows[np.unique(picks)].tolist())
    mask = np.array(
        [(i in keep_rest) or (code != rest_code) for i, code in enumerate(events[:, 2])]
    )
    return events[mask]


def build_metadata(
    events: np.ndarray, raw: mne.io.BaseRaw, spec: RunSpec, config: DatasetConfig
) -> pd.DataFrame:
    """One provenance row per event, in :data:`METADATA_COLUMNS` order."""
    pre = config.preprocessing
    sfreq = float(raw.info["sfreq"])
    n_channels = len(raw.ch_names)
    inverse_map = {v: k for k, v in CLASS_TO_INDEX.items()}
    run_map = event_map_for_run(spec.run)
    label_to_description = {label: desc for desc, label in run_map.items()}

    rows: List[Dict[str, object]] = []
    for sample, _, code in events:
        # events[:, 0] is absolute; first_samp makes it relative to the data array.
        onset_sample = int(sample) - int(raw.first_samp)
        onset_time = onset_sample / sfreq
        label = inverse_map[int(code)]
        start_sample = onset_sample + int(round(pre.epoch_tmin * sfreq))

        rows.append(
            {
                "subject_id": spec.subject_id,
                "run_id": spec.run_id,
                "event_code": label_to_description[label],
                "class_label": label,
                "onset_sample": onset_sample,
                "onset_time": onset_time,
                "sampling_rate": sfreq,
                "channel_count": n_channels,
                "epoch_start": onset_time + pre.epoch_tmin,
                "epoch_end": onset_time + pre.epoch_tmax,
                "subject": spec.subject,
                "run": spec.run,
                "task_type": spec.task_type.value,
                "class_index": int(code),
                "epoch_tmin": pre.epoch_tmin,
                "epoch_tmax": pre.epoch_tmax,
                "epoch_start_sample": start_sample,
                "epoch_end_sample": start_sample + pre.expected_epoch_samples,
                "n_times": pre.expected_epoch_samples,
                "source": config.source,
            }
        )

    return pd.DataFrame.from_records(rows, columns=list(METADATA_COLUMNS))


def mne_tmax(config: DatasetConfig, sfreq: float) -> float:
    """Translate the configured epoch window into MNE's ``tmax``.

    MNE's ``Epochs`` window is *closed* -- it includes the sample at ``tmax`` --
    so ``tmin=0, tmax=4.0`` at 160 Hz yields 641 samples, not 640.  The config
    expresses the window as half-open ``[tmin, tmax)``, which is the convention
    that gives a round 640-sample epoch, so one sample period is subtracted here
    when ``include_tmax`` is false.
    """
    pre = config.preprocessing
    if pre.include_tmax:
        return pre.epoch_tmax
    return pre.epoch_tmax - 1.0 / sfreq


@dataclass
class RunEpochs:
    """Epochs and aligned metadata for one recording."""

    spec: RunSpec
    epochs: mne.Epochs
    metadata: pd.DataFrame
    n_dropped: int

    @property
    def data(self) -> np.ndarray:
        """``(n_epochs, n_channels, n_times)`` in volts."""
        return self.epochs.get_data(copy=True)

    @property
    def labels(self) -> np.ndarray:
        """Integer class indices, aligned with :attr:`data`."""
        return self.metadata["class_index"].to_numpy(dtype=np.int64)


def extract_epochs(
    raw: mne.io.BaseRaw, spec: RunSpec, config: DatasetConfig
) -> RunEpochs:
    """Extract fixed-length epochs around the events of one recording.

    ``include_tmax=False`` makes the window half-open, ``[tmin, tmax)``, so the
    epoch length is exactly ``(tmax - tmin) * sfreq`` samples.  With the default
    4 s window at 160 Hz that is 640 samples -- a power-of-two-friendly length that
    every epoch shares, which is what a fixed-input model such as EEGNet requires.
    """
    pre = config.preprocessing
    keep_classes = list(PHASE1_CLASSES) if config.include_rest else [
        c for c in PHASE1_CLASSES if c != REST
    ]

    events, event_id = build_events(raw, spec, keep_classes)
    events = subsample_rest(events, config.max_rest_epochs_per_run, config.seed)
    if events.size == 0:
        raise EpochingError(f"{spec}: no events left after class filtering")

    metadata = build_metadata(events, raw, spec, config)

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={k: v for k, v in event_id.items() if v in set(events[:, 2])},
        tmin=pre.epoch_tmin,
        tmax=mne_tmax(config, float(raw.info["sfreq"])),
        baseline=pre.baseline,
        picks="eeg",
        preload=True,
        reject=(
            {"eeg": pre.reject_peak_to_peak_v}
            if pre.reject_peak_to_peak_v is not None
            else None
        ),
        # Epochs running past the end of the recording are dropped by MNE; the
        # attached metadata is subset in lockstep so rows never desynchronise.
        metadata=metadata,
        on_missing="warn",
        verbose="ERROR",
    )

    n_dropped = len(events) - len(epochs)
    if n_dropped:
        log.info("%s: %d epoch(s) dropped by MNE (window past end of run)", spec, n_dropped)

    aligned = epochs.metadata.reset_index(drop=True).copy()

    data = epochs.get_data(copy=False)
    if data.shape[0] == 0:
        raise EpochingError(f"{spec}: all epochs were dropped")
    if data.shape[2] != pre.expected_epoch_samples:
        raise EpochingError(
            f"{spec}: epochs have {data.shape[2]} samples, expected "
            f"{pre.expected_epoch_samples}"
        )

    # Record, but do not act on, per-epoch amplitude excursions.  The brief asks
    # for no aggressive artifact removal, so noisy epochs are flagged and kept;
    # a later phase can filter on this column without the data having been lost.
    # Peak-to-peak is per channel, then maximised over channels -- the conventional
    # definition, and the one MNE's `reject` uses.  (Phase 1 originally took the
    # global max-minus-min across all channels at once, which is >= this value and
    # not what "peak-to-peak" normally means.  On real data both flag the same
    # epochs; the definition is corrected here for correctness, not to change any
    # outcome.  The threshold itself is deliberately left untouched.)
    aligned["peak_to_peak_v"] = (data.max(axis=2) - data.min(axis=2)).max(axis=1)
    aligned["artifact_flag"] = aligned["peak_to_peak_v"] > pre.reject_log_threshold_v
    aligned["n_times"] = data.shape[2]
    aligned["channel_count"] = data.shape[1]

    return RunEpochs(spec=spec, epochs=epochs, metadata=aligned, n_dropped=n_dropped)
