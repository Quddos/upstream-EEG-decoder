"""Continuous-signal preprocessing for motor-imagery EEG.

Every operation here is applied to the *continuous* recording, before epoching, and
every one is justified in ``docs/PREPROCESSING_DECISIONS.md``.  The order is fixed
and deliberate:

    1. channel selection      -- decide what is in the montage before anything else
    2. notch (optional)       -- narrowband line noise, if the passband admits it
    3. band-pass              -- restrict to the sensorimotor rhythms
    4. common average ref     -- spatial: remove the reference-common component

Filtering before re-referencing matters: CAR averages across channels, so any
channel still carrying large out-of-band drift would inject that drift into every
other channel through the average.  Band-passing first removes it.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import mne
import numpy as np

from src.config import PreprocessingConfig

log = logging.getLogger(__name__)


class PreprocessingError(RuntimeError):
    """Raised when a preprocessing step cannot be applied as configured."""


def select_channels(
    raw: mne.io.BaseRaw, subset: Optional[Sequence[str]]
) -> mne.io.BaseRaw:
    """Restrict the recording to EEG channels, optionally to a named subset.

    With ``subset=None`` (the default) all 64 EEG channels are kept.  Phase 1 does
    not reduce the montage: CSP and EEGNet both learn their own spatial filters in
    Phase 2, and hand-picking motor channels now would discard the information
    those methods exist to find.
    """
    raw.pick("eeg")

    if subset is None:
        return raw

    missing = [ch for ch in subset if ch not in raw.ch_names]
    if missing:
        raise PreprocessingError(
            f"requested channels not present in recording: {missing}"
        )
    raw.pick(list(subset))
    return raw


def apply_notch(
    raw: mne.io.BaseRaw, config: PreprocessingConfig
) -> mne.io.BaseRaw:
    """Optionally remove power-line interference.

    Off by default.  With an 8-30 Hz passband the 60 Hz line component is already
    ~60 dB down after the band-pass, so notching first would be a no-op on the
    output while adding a second filter's ringing to the signal.  The step is kept
    configurable because a broadband variant of this pipeline (e.g. 1-100 Hz for
    ERP-style analysis) would genuinely need it.
    """
    if not config.notch_enabled:
        return raw

    nyquist = raw.info["sfreq"] / 2.0
    freqs = [f for f in config.notch_freqs if f < nyquist]
    dropped = [f for f in config.notch_freqs if f >= nyquist]
    if dropped:
        log.warning(
            "notch frequencies %s are at or above Nyquist (%.1f Hz) and were skipped",
            dropped,
            nyquist,
        )
    if not freqs:
        return raw

    raw.notch_filter(freqs=freqs, method="fir", phase=config.filter_phase, verbose="ERROR")
    return raw


def apply_bandpass(
    raw: mne.io.BaseRaw, config: PreprocessingConfig
) -> mne.io.BaseRaw:
    """Band-pass to the sensorimotor rhythms.

    The default 8-30 Hz window spans mu (8-13 Hz) and beta (13-30 Hz).  Motor
    imagery produces event-related desynchronisation in exactly these bands over
    the contralateral sensorimotor cortex; that is the effect the whole project
    decodes.  The band-pass simultaneously suppresses drift and sweat artifacts
    (<1 Hz), ocular activity (mostly <4 Hz) and EMG (mostly >40 Hz) without any
    subject-specific artifact rejection, which is why aggressive artifact removal
    is unnecessary at this stage.

    A zero-phase FIR is used so that event-locked activity is not shifted in time:
    a causal (minimum-phase) filter would delay the ERD onset by a band-dependent
    amount and misalign it with the epoch window.
    """
    sfreq = raw.info["sfreq"]
    if config.h_freq >= sfreq / 2.0:
        raise PreprocessingError(
            f"h_freq={config.h_freq} Hz is at or above Nyquist ({sfreq / 2.0} Hz)"
        )
    if config.l_freq <= 0 or config.l_freq >= config.h_freq:
        raise PreprocessingError(
            f"invalid passband: l_freq={config.l_freq}, h_freq={config.h_freq}"
        )

    raw.filter(
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        method=config.filter_method,
        fir_design=config.fir_design,
        phase=config.filter_phase,
        filter_length=config.filter_length,
        verbose="ERROR",
    )
    return raw


def apply_common_average_reference(
    raw: mne.io.BaseRaw, config: PreprocessingConfig
) -> mne.io.BaseRaw:
    """Re-reference to the common average.

    EEGMMIDB is recorded against a single physical reference electrode, so every
    channel carries that electrode's activity with the opposite sign.  For a
    left-vs-right decoder this is actively harmful: the contrast between C3 and C4
    is the signal, and a shared reference component is a common-mode term sitting
    on top of both.  CAR removes it and is the standard choice for a 64-channel
    montage with good scalp coverage.

    Applied directly (``projection=False``) rather than as a projector so the
    stored epochs are the referenced data, with nothing left to apply downstream.
    """
    if not config.common_average_reference:
        return raw

    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
    return raw


def preprocess_raw(
    raw: mne.io.BaseRaw, config: PreprocessingConfig
) -> mne.io.BaseRaw:
    """Run the full continuous-signal chain in its fixed order."""
    raw = select_channels(raw, config.channel_subset)
    raw = apply_notch(raw, config)
    raw = apply_bandpass(raw, config)
    raw = apply_common_average_reference(raw, config)
    return raw


def normalize_epochs(data: np.ndarray, method: str) -> np.ndarray:
    """Normalise epoched data.

    Parameters
    ----------
    data
        Array of shape ``(n_epochs, n_channels, n_times)``.
    method
        ``"zscore_per_channel"``
            z-score each channel of each epoch independently.  This is the
            standard input scaling for EEGNet and it removes the per-session
            amplitude differences (impedance, skull thickness) that otherwise
            dominate cross-subject variance.
        ``"zscore_per_epoch"``
            z-score using statistics pooled over all channels of an epoch,
            preserving relative amplitude differences *between* channels -- which
            is what CSP-style spatial filters rely on.
        ``"none"``
            leave the data in volts.

    Notes
    -----
    All supported methods compute their statistics **within a single epoch**.  No
    statistic is ever pooled across epochs, runs or subjects, so normalisation
    cannot leak information between a future train and test split.  This is why it
    is safe to apply here rather than inside a cross-validation fold.
    """
    if method == "none":
        return data
    if data.size == 0:
        return data

    if method == "zscore_per_channel":
        axis = (2,)
    elif method == "zscore_per_epoch":
        axis = (1, 2)
    else:
        raise PreprocessingError(f"unknown normalization method {method!r}")

    mean = data.mean(axis=axis, keepdims=True)
    std = data.std(axis=axis, keepdims=True)
    # Guard against a flat channel producing inf/nan.
    std = np.where(std < 1e-20, 1.0, std)
    return (data - mean) / std


def describe_filter_settings(config: PreprocessingConfig) -> List[str]:
    """Human-readable summary of the filter chain, for reports and figures."""
    lines = [
        f"band-pass {config.l_freq}-{config.h_freq} Hz "
        f"({config.filter_method.upper()}, {config.fir_design}, phase={config.filter_phase})",
    ]
    if config.notch_enabled:
        lines.append(f"notch {list(config.notch_freqs)} Hz")
    else:
        lines.append("notch disabled (redundant inside the band-pass stopband)")
    lines.append(
        "common average reference"
        if config.common_average_reference
        else "original reference retained"
    )
    lines.append(
        f"epoch window [{config.epoch_tmin}, {config.epoch_tmax}) s "
        f"= {config.expected_epoch_samples} samples"
    )
    lines.append(
        f"baseline correction: {config.baseline if config.baseline else 'disabled'}"
    )
    lines.append(f"normalization at load time: {config.normalization}")
    return lines
