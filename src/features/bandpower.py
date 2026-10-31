"""Spectral band power for Phase 2 feature extraction.

Everything here is computed **within a single epoch**.  No statistic is ever
pooled across epochs, runs or subjects, which is what makes it safe to run the
whole extraction once, before any cross-validation split, without leaking
information across a fold boundary.  :mod:`tests.test_leakage` proves this
structurally: extracting features for a subset of epochs returns exactly the
corresponding rows of the full extraction.

Units and normalisation
-----------------------
Band power is computed from the stored arrays **in volts, un-normalised**.  That
matters more than it looks.  The pipeline's default load-time normalisation is
``zscore_per_channel``, which forces every channel of every epoch to unit
variance; measured on S001 that collapses the C3/C4 lateralisation index from
-0.122 to -0.008 and drives the between-channel spread of total power to ~1e-13.
Any feature that compares one channel's power to another's -- which is most of
Phase 2 -- must therefore be built on un-normalised data, and
:func:`~src.features.extract.extract_features` loads with ``normalize=False``.

Cross-subject scale is handled where it belongs instead: by the ratio features
(which are scale-free by construction) and by the fold-local ``StandardScaler``
inside the cross-validation pipeline.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.signal import welch

#: Frequency bands, in Hz.  ``mu`` and ``beta`` partition the 8-30 Hz passband
#: that :mod:`src.preprocessing.filters` already applied, so their relative
#: powers sum to 1 by construction -- see :func:`relative_band_power`.
BANDS: Mapping[str, Tuple[float, float]] = {
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

#: The passband the preprocessing produced.  Used as the denominator for
#: relative power: outside it the signal is filter roll-off, not measurement.
PASSBAND_HZ: Tuple[float, float] = (8.0, 30.0)

#: Welch segment length in samples.  256 at 160 Hz gives 1.6 s segments and a
#: 0.625 Hz bin width, so the 5 Hz mu band spans ~8 bins.  This is the value the
#: Phase 1 spectral figures and the Phase 1.6 QC both use; keeping it identical
#: means Phase 2 numbers are directly comparable with theirs.
WELCH_NPERSEG: int = 256

#: Floor applied before taking a logarithm, in V²/Hz.  Real band powers here are
#: ~1e-12, so this only ever guards against an exact zero (a flat channel), never
#: against ordinary small values.
LOG_FLOOR: float = 1e-30


def welch_psd(
    X: np.ndarray, sfreq: float, nperseg: int = WELCH_NPERSEG
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD of every epoch and channel.

    Parameters
    ----------
    X
        ``(n_epochs, n_channels, n_times)`` in volts.

    Returns
    -------
    freqs
        ``(n_freqs,)`` in Hz.
    psd
        ``(n_epochs, n_channels, n_freqs)`` in V²/Hz.
    """
    if X.ndim != 3:
        raise ValueError(f"expected (n_epochs, n_channels, n_times), got {X.shape}")
    freqs, psd = welch(X, fs=sfreq, nperseg=min(nperseg, X.shape[2]), axis=-1)
    return freqs, psd


def band_mask(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    """Boolean mask of the bins inside ``band``.

    The interval is half-open at the top, ``[lo, hi)``, so adjacent bands such as
    mu ``[8, 13)`` and beta ``[13, 30)`` partition the passband without sharing a
    bin.  A closed interval would double-count the 13 Hz bin and make relative
    powers sum to slightly more than one.
    """
    lo, hi = band
    return (freqs >= lo) & (freqs < hi)


def band_power(psd: np.ndarray, freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    """Mean PSD across the bins of ``band``.

    Returns ``(n_epochs, n_channels)`` in V²/Hz.  The mean rather than the sum is
    used so the value does not scale with band width, which keeps mu and beta
    numerically comparable despite beta being three times wider.
    """
    mask = band_mask(freqs, band)
    if not mask.any():
        raise ValueError(f"no PSD bins fall inside band {band}")
    return psd[..., mask].mean(axis=-1)


def absolute_band_powers(
    psd: np.ndarray, freqs: np.ndarray, bands: Mapping[str, Tuple[float, float]] = BANDS
) -> Dict[str, np.ndarray]:
    """``{band_name: (n_epochs, n_channels)}`` mean PSD per band, in V²/Hz."""
    return {name: band_power(psd, freqs, band) for name, band in bands.items()}


def total_passband_power(psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Mean PSD across the whole 8-30 Hz passband, ``(n_epochs, n_channels)``."""
    return band_power(psd, freqs, PASSBAND_HZ)


def relative_band_power(
    band: np.ndarray, total: np.ndarray, n_band_bins: int, n_total_bins: int
) -> np.ndarray:
    """Fraction of passband power falling in a band.

    Both inputs are *mean* PSD, so each is re-weighted back to a sum before the
    ratio is taken; otherwise the "fraction" would not be a fraction.

    The weights are **bin counts, not nominal bandwidths**.  With a 0.625 Hz bin
    width the mu band ``[8, 13)`` holds 8 bins and beta ``[13, 30)`` holds 27, so
    their nominal widths (5 and 17 Hz) are not exactly proportional to the number
    of bins each actually contains.  Re-weighting by nominal width leaves a ~1%
    residual and the two relative powers then fail to sum to one; re-weighting by
    bin count is exact, because the bins genuinely partition the passband.

    Because mu and beta partition ``[8, 30)``, the two relative powers sum to 1
    for every epoch and channel, so they carry a single degree of freedom between
    them.  Both are kept because the Phase 2 brief asks for both, and every model
    used is regularised, so the exact collinearity is harmless -- but it is a
    property of the design, not an accident, and ``tests/test_features.py``
    asserts it so it cannot drift into being one.
    """
    denominator = total * float(n_total_bins)
    return np.divide(
        band * float(n_band_bins),
        denominator,
        out=np.zeros_like(band),
        where=denominator > 0,
    )


def n_bins(freqs: np.ndarray, band: Tuple[float, float]) -> int:
    """How many PSD bins fall inside ``band``."""
    return int(band_mask(freqs, band).sum())


def log_power(power: np.ndarray) -> np.ndarray:
    """Natural log of band power, floored so an exact zero cannot produce -inf.

    Band power is strongly right-skewed across epochs; the log makes the
    distribution closer to symmetric, which is what linear models assume. It also
    turns the between-subject amplitude differences documented in Phase 1.7
    (2.33x in median epoch amplitude) into an additive offset rather than a
    multiplicative one -- which a per-fold ``StandardScaler`` can then centre.
    """
    return np.log(np.maximum(power, LOG_FLOOR))


def band_width(band: Tuple[float, float]) -> float:
    return float(band[1] - band[0])
