"""Locally-generated EDF+ fixtures with EEGMMIDB structure.

Why this exists
---------------
The Phase 1 deliverable is a *pipeline*, and a pipeline that can only be executed
where ``physionet.org`` is reachable cannot be tested in CI, cannot be tested in a
sandbox, and cannot be demonstrated to a reviewer offline.  This module writes EDF+
files that are structurally indistinguishable from real EEGMMIDB recordings:

* 64 channels carrying the exact EDF labels of the real record (``Fc5.``, ``Iz..``),
* 160 Hz sampling rate,
* ~125 s runs,
* EDF+ annotations using the real ``T0``/``T1``/``T2`` vocabulary, with the real
  alternating rest/task timing (4.2 s rest, 4.1 s task),
* run-dependent T1/T2 semantics, taken from :mod:`src.data.physionet_spec` -- the
  same source of truth the pipeline itself reads, so the fixture cannot encode a
  different convention than the loader decodes.

The *signal* is simulated, not real EEG.  It is built from a 1/f background plus
mu (10 Hz) and beta (22 Hz) sensorimotor rhythms whose amplitude drops over the
hemisphere **contralateral** to the imagined hand -- the event-related
desynchronisation that motor-imagery decoding actually exploits.  That makes the
fixture a genuine end-to-end test: if the pipeline mislabels left and right, or
loses channel positions, or epochs on the wrong samples, the left/right PSD
comparison figure stops showing the expected C3/C4 asymmetry.

.. warning::
   Data produced here is **synthetic** and is written with a
   ``SYNTHETIC-EEGMMIDB-FIXTURE`` marker in the EDF recording-identification
   field.  It must never be used to make a scientific claim about human
   neurophysiology, and it must never be mixed with real recordings in an
   analysis.  Datasets built from it are stamped ``source="synthetic"`` in their
   metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np

from src.data.physionet_spec import (
    EDF_CHANNEL_NAMES,
    LEFT_HAND_EXECUTION,
    LEFT_HAND_IMAGERY,
    REST,
    RIGHT_HAND_EXECUTION,
    RIGHT_HAND_IMAGERY,
    SAMPLING_RATE_HZ,
    RunSpec,
    event_map_for_run,
)

log = logging.getLogger(__name__)

#: Marker written into the EDF header so synthetic files are always identifiable.
SYNTHETIC_MARKER = "SYNTHETIC-EEGMMIDB-FIXTURE"

#: Real EEGMMIDB task runs are two minutes long.
RUN_DURATION_S = 125.0
#: Rest (T0) and task (T1/T2) cue durations in the real record.
REST_DURATION_S = 4.2
TASK_DURATION_S = 4.1

#: Amplitude of the 1/f background, in volts.
BACKGROUND_AMPLITUDE_V = 12e-6
#: Peak amplitude of the sensorimotor rhythms at their focus, in volts.
MU_AMPLITUDE_V = 22e-6
BETA_AMPLITUDE_V = 9e-6
#: Fractional amplitude remaining during ERD (0.45 -> a 55% power-carrying
#: amplitude drop, comparable to a strong real ERD).
ERD_DEPTH = 0.45

MU_FREQ_HZ = 10.0
BETA_FREQ_HZ = 22.0


def _clean_names() -> List[str]:
    """EDF labels with padding removed, for hemisphere lookups."""
    return [name.rstrip(".").replace(".", "") for name in EDF_CHANNEL_NAMES]


def _hemisphere(name: str) -> str:
    """``'left'``, ``'right'`` or ``'midline'`` for a 10-10 electrode label.

    In the 10-10 convention odd trailing numbers are left-hemisphere, even are
    right, and 'z' is midline.
    """
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        return "midline"
    return "left" if int(digits) % 2 == 1 else "right"


def _sensorimotor_weight(name: str) -> float:
    """How strongly an electrode carries the sensorimotor rhythm, in ``[0, 1]``.

    Peaks over the C row (the hand knob), falls off through CP/FC to the frontal
    and occipital edges of the montage.
    """
    upper = name.upper()
    prefix = "".join(ch for ch in upper if ch.isalpha())
    weights = {
        "C": 1.0,
        "CP": 0.8,
        "FC": 0.8,
        "P": 0.45,
        "F": 0.35,
        "T": 0.3,
        "TP": 0.3,
        "FT": 0.3,
        "PO": 0.2,
        "O": 0.15,
        "AF": 0.15,
        "FP": 0.1,
        "IZ": 0.1,
    }
    base = weights.get(prefix, 0.2)

    # Lateral electrodes over the hand knob (C3/C4, index 3-4) are the focus;
    # far-lateral (C5/C6, T7/T8) and midline (Cz) sites carry less hand signal.
    digits = "".join(ch for ch in upper if ch.isdigit())
    if prefix == "C" and digits in {"3", "4"}:
        base *= 1.0
    elif prefix == "C" and digits in {"1", "2"}:
        base *= 0.8
    elif prefix == "C" and digits in {"5", "6"}:
        base *= 0.7
    return float(base)


def _pink_noise(n_channels: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """1/f-shaped noise, the dominant component of a real EEG background."""
    spectrum = rng.normal(size=(n_channels, n_samples // 2 + 1)) + 1j * rng.normal(
        size=(n_channels, n_samples // 2 + 1)
    )
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / SAMPLING_RATE_HZ)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    scale[0] = 0.0  # no DC
    noise = np.fft.irfft(spectrum * scale, n=n_samples)
    noise /= np.std(noise, axis=1, keepdims=True)
    return noise


def build_event_sequence(run: int) -> List[Tuple[float, float, str]]:
    """Alternating rest/task annotation sequence for one run.

    Returns ``(onset_s, duration_s, description)`` triples using the real EEGMMIDB
    timing.  Task cues alternate between ``T1`` and ``T2``.
    """
    events: List[Tuple[float, float, str]] = []
    onset = 0.0
    task_codes = ("T1", "T2")
    index = 0

    while onset + REST_DURATION_S + TASK_DURATION_S <= RUN_DURATION_S:
        events.append((onset, REST_DURATION_S, "T0"))
        onset += REST_DURATION_S
        events.append((onset, TASK_DURATION_S, task_codes[index % 2]))
        onset += TASK_DURATION_S
        index += 1

    # Trailing rest, as in the real recordings.
    if onset < RUN_DURATION_S:
        events.append((onset, RUN_DURATION_S - onset, "T0"))
    return events


def _erd_envelope(
    events: List[Tuple[float, float, str]],
    run: int,
    n_samples: int,
    hemispheres: List[str],
) -> np.ndarray:
    """Per-channel, per-sample amplitude envelope encoding contralateral ERD.

    Motor imagery of one hand suppresses the mu/beta rhythm over the *opposite*
    hemisphere.  The envelope starts at 1.0 (idle rhythm present) and dips toward
    :data:`ERD_DEPTH` over the contralateral side during task events.
    """
    n_channels = len(hemispheres)
    envelope = np.ones((n_channels, n_samples), dtype=np.float64)
    mapping = event_map_for_run(run)

    #: Which hemisphere desynchronises for each class label.
    contralateral: Dict[str, str] = {
        LEFT_HAND_IMAGERY: "right",
        LEFT_HAND_EXECUTION: "right",
        RIGHT_HAND_IMAGERY: "left",
        RIGHT_HAND_EXECUTION: "left",
    }

    for onset, duration, code in events:
        label = mapping.get(code)
        if label is None or label == REST:
            continue
        side = contralateral.get(label)
        if side is None:
            # Bilateral runs: both hemispheres desynchronise.
            affected = [i for i, h in enumerate(hemispheres) if h != "midline"]
        else:
            affected = [i for i, h in enumerate(hemispheres) if h == side]
        if not affected:
            continue

        start = int(round(onset * SAMPLING_RATE_HZ))
        stop = min(n_samples, int(round((onset + duration) * SAMPLING_RATE_HZ)))
        if stop <= start:
            continue

        # Smooth onset/offset: ERD builds over ~0.5 s rather than switching
        # instantaneously, which keeps the simulated spectrum physiological.
        length = stop - start
        ramp = np.ones(length)
        ramp_len = min(int(0.5 * SAMPLING_RATE_HZ), length // 2)
        if ramp_len > 0:
            ramp[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
            ramp[-ramp_len:] = np.linspace(1.0, 0.0, ramp_len)
        envelope[affected, start:stop] = 1.0 - (1.0 - ERD_DEPTH) * ramp

    return envelope


def simulate_run(spec: RunSpec, seed: int) -> Tuple[np.ndarray, List[Tuple[float, float, str]]]:
    """Simulate the signal and annotations of one run.

    Returns ``(data, events)`` where ``data`` is ``(64, n_samples)`` in volts.
    """
    # Seed derived from subject and run so every fixture is deterministic and
    # different runs are not identical copies.
    rng = np.random.default_rng(seed + spec.subject * 100 + spec.run)

    names = _clean_names()
    n_channels = len(names)
    n_samples = int(round(RUN_DURATION_S * SAMPLING_RATE_HZ))
    times = np.arange(n_samples) / SAMPLING_RATE_HZ

    events = build_event_sequence(spec.run)
    hemispheres = [_hemisphere(name) for name in names]
    weights = np.array([_sensorimotor_weight(name) for name in names])[:, None]
    envelope = _erd_envelope(events, spec.run, n_samples, hemispheres)

    # Independent phase per channel keeps channels from being perfectly coherent.
    mu_phase = rng.uniform(0, 2 * np.pi, size=(n_channels, 1))
    beta_phase = rng.uniform(0, 2 * np.pi, size=(n_channels, 1))

    mu = MU_AMPLITUDE_V * weights * envelope * np.sin(
        2 * np.pi * MU_FREQ_HZ * times[None, :] + mu_phase
    )
    beta = BETA_AMPLITUDE_V * weights * envelope * np.sin(
        2 * np.pi * BETA_FREQ_HZ * times[None, :] + beta_phase
    )
    background = BACKGROUND_AMPLITUDE_V * _pink_noise(n_channels, n_samples, rng)

    # Occipital alpha, present regardless of task -- a realistic distractor that a
    # decoder must not confuse with the sensorimotor mu rhythm.
    occipital = np.array(
        [1.0 if name.upper().startswith(("O", "PO", "IZ")) else 0.0 for name in names]
    )[:, None]
    alpha = (
        14e-6
        * occipital
        * np.sin(2 * np.pi * 10.5 * times[None, :] + rng.uniform(0, 2 * np.pi))
    )

    data = mu + beta + alpha + background
    return data.astype(np.float64), events


def write_synthetic_run(spec: RunSpec, target: Path, seed: int = 0) -> Path:
    """Write one synthetic EEGMMIDB-shaped EDF+ file to ``target``."""
    if target.exists():
        return target

    data, events = simulate_run(spec, seed)

    info = mne.create_info(
        ch_names=list(EDF_CHANNEL_NAMES),
        sfreq=SAMPLING_RATE_HZ,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(
            onset=[e[0] for e in events],
            duration=[e[1] for e in events],
            description=[e[2] for e in events],
        )
    )
    # Flag the file as simulated in a place that survives every downstream read.
    with raw.info._unlock():
        raw.info["subject_info"] = {"his_id": f"{SYNTHETIC_MARKER}-{spec.subject_id}"}

    target.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(target, raw, fmt="edf", overwrite=True, verbose="ERROR")
    log.info("wrote synthetic fixture %s", target)
    return target


def write_synthetic_dataset(
    specs: List[RunSpec], raw_dir: Path, seed: int = 0
) -> List[Path]:
    """Write synthetic fixtures for many recordings."""
    return [
        write_synthetic_run(spec, raw_dir / spec.subject_id / spec.filename, seed=seed)
        for spec in specs
    ]
