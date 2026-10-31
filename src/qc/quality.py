"""Phase 1.6 -- diagnostic quality control for real EEG epochs.

This module **measures**.  It does not clean, reject, re-threshold or re-filter
anything.  Every function here reads the epochs that
``python -m src.cli.build_dataset`` already wrote and returns a description of
them; nothing it does can change what a later phase is trained on.

Why it exists
-------------
Phase 1.5 found that the inherited 150 µV peak-to-peak flag fires on ~78% of real
EEGMMIDB epochs and on 0% of the synthetic fixtures it was calibrated against.
That is a statement about the *threshold*, not yet a statement about the data.
Phase 1.6 characterises the real amplitude distribution so that a Phase 2
decision about the threshold can be made from evidence.

Definitions used throughout
---------------------------
peak-to-peak (per channel)
    ``max(x) - min(x)`` over the 640 samples of one epoch, for one channel.
peak-to-peak (per epoch)
    the **maximum over channels** of the per-channel peak-to-peak.  This is the
    conventional definition and the one MNE's ``reject`` uses; it is also exactly
    the quantity ``epoching.extract_epochs`` already stores as ``peak_to_peak_v``.
absolute amplitude (per epoch)
    ``max |x|`` over all channels and all samples.
flagged
    per-epoch peak-to-peak > 150 µV.  **The threshold is inherited from Phase 1
    and is not changed here** (:data:`PreprocessingConfig.reject_log_threshold_v`).
artifact channel count
    how many of the 64 channels individually exceed 150 µV peak-to-peak in that
    epoch.  An epoch is flagged exactly when this count is >= 1, so the count says
    whether the flag came from one electrode or from the whole montage -- which is
    the difference between a bad channel and a genuinely high-amplitude epoch.

Amplitudes are reported in **microvolts**.  The stored arrays are in volts; the
conversion is the only unit change applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch

#: Volts -> microvolts.
V_TO_UV: float = 1e6

#: Mu band, in Hz.  Matches the band used by ``viz.figures.lateralization_index``
#: so Phase 1.6 numbers are directly comparable with the Phase 1.5 report.
MU_BAND_HZ: Tuple[float, float] = (8.0, 13.0)

#: Welch segment length, in samples.  Also matches ``viz.figures``.
WELCH_NPERSEG: int = 256


# --------------------------------------------------------------------------------------
# Amplitude measurement
# --------------------------------------------------------------------------------------


def channel_peak_to_peak_uv(X: np.ndarray) -> np.ndarray:
    """Per-epoch, per-channel peak-to-peak amplitude in µV.

    Parameters
    ----------
    X
        ``(n_epochs, n_channels, n_times)`` in volts.

    Returns
    -------
    ``(n_epochs, n_channels)`` in µV.
    """
    return (X.max(axis=2) - X.min(axis=2)) * V_TO_UV


def channel_peak_abs_uv(X: np.ndarray) -> np.ndarray:
    """Per-epoch, per-channel maximum absolute amplitude in µV."""
    return np.abs(X).max(axis=2) * V_TO_UV


@dataclass(frozen=True)
class AmplitudeMeasures:
    """Every amplitude quantity Phase 1.6 reports, for one set of epochs."""

    #: ``(n_epochs, n_channels)`` peak-to-peak, µV.
    per_channel_p2p: np.ndarray
    #: ``(n_epochs, n_channels)`` max |x|, µV.
    per_channel_abs: np.ndarray
    #: ``(n_epochs,)`` peak-to-peak maximised over channels, µV.
    epoch_p2p_max: np.ndarray
    #: ``(n_epochs,)`` max |x| over channels and time, µV.
    epoch_abs_max: np.ndarray
    #: ``(n_epochs,)`` how many channels individually exceed the threshold.
    artifact_channel_count: np.ndarray
    #: ``(n_epochs,)`` boolean flag, epoch peak-to-peak > threshold.
    artifact_flag: np.ndarray
    #: The inherited threshold, in µV.  Recorded so no reader has to assume it.
    threshold_uv: float
    channels: Tuple[str, ...]

    @property
    def n_epochs(self) -> int:
        return int(self.epoch_p2p_max.shape[0])


def measure_amplitudes(
    X: np.ndarray, channels: Sequence[str], threshold_v: float
) -> AmplitudeMeasures:
    """Compute every amplitude quantity for ``X`` (volts) at the given threshold.

    ``threshold_v`` is passed in rather than read from the config so that the
    caller has to name it explicitly; Phase 1.6 always passes the inherited
    ``PreprocessingConfig.reject_log_threshold_v`` unchanged.
    """
    if X.ndim != 3:
        raise ValueError(f"expected (n_epochs, n_channels, n_times), got {X.shape}")
    if X.shape[1] != len(channels):
        raise ValueError(
            f"{X.shape[1]} channels in the array but {len(channels)} channel names"
        )

    threshold_uv = float(threshold_v) * V_TO_UV
    per_channel_p2p = channel_peak_to_peak_uv(X)
    per_channel_abs = channel_peak_abs_uv(X)
    epoch_p2p_max = per_channel_p2p.max(axis=1)
    over = per_channel_p2p > threshold_uv

    return AmplitudeMeasures(
        per_channel_p2p=per_channel_p2p,
        per_channel_abs=per_channel_abs,
        epoch_p2p_max=epoch_p2p_max,
        epoch_abs_max=per_channel_abs.max(axis=1),
        artifact_channel_count=over.sum(axis=1).astype(int),
        # Identical by construction to ``over.any(axis=1)``; written against the
        # epoch-level maximum so it mirrors the pipeline's own flag exactly.
        artifact_flag=epoch_p2p_max > threshold_uv,
        threshold_uv=threshold_uv,
        channels=tuple(channels),
    )


# --------------------------------------------------------------------------------------
# 1 -- epoch-level QC table
# --------------------------------------------------------------------------------------

#: Columns of ``data/metadata/epoch_qc.csv``, in order.  Amplitudes are in µV.
EPOCH_QC_COLUMNS: Tuple[str, ...] = (
    "epoch_id",
    "subject",
    "run",
    "class",
    "onset",
    "peak_to_peak_max",
    "peak_abs_max",
    "artifact_flag",
    "artifact_channel_count",
)


def epoch_qc_table(
    metadata: pd.DataFrame, measures: AmplitudeMeasures
) -> pd.DataFrame:
    """One row per epoch, in :data:`EPOCH_QC_COLUMNS` order.

    ``onset`` is the event onset in seconds from the start of the recording, i.e.
    the ``onset_time`` the pipeline already recorded.
    """
    if len(metadata) != measures.n_epochs:
        raise ValueError(
            f"{len(metadata)} metadata rows but {measures.n_epochs} epochs measured"
        )

    return pd.DataFrame(
        {
            "epoch_id": metadata["epoch_uid"].to_numpy(),
            "subject": metadata["subject_id"].to_numpy(),
            "run": metadata["run_id"].to_numpy(),
            "class": metadata["class_label"].to_numpy(),
            "onset": metadata["onset_time"].to_numpy(),
            "peak_to_peak_max": measures.epoch_p2p_max,
            "peak_abs_max": measures.epoch_abs_max,
            "artifact_flag": measures.artifact_flag,
            "artifact_channel_count": measures.artifact_channel_count,
        },
        columns=list(EPOCH_QC_COLUMNS),
    )


# --------------------------------------------------------------------------------------
# 2 -- distribution statistics
# --------------------------------------------------------------------------------------

#: Statistics reported for each amplitude measure, in report order.
DISTRIBUTION_STATS: Tuple[str, ...] = (
    "min",
    "max",
    "mean",
    "median",
    "std",
    "p95",
    "p99",
)


def describe(values: np.ndarray) -> Dict[str, float]:
    """The seven Phase 1.6 distribution statistics for a 1-D sample.

    ``std`` is the **sample** standard deviation (``ddof=1``): these 23 epochs are
    a sample of the subject's data, not the population.  Percentiles use numpy's
    default linear interpolation.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return {name: float("nan") for name in DISTRIBUTION_STATS}
    return {
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
    }


def distribution_table(measures: AmplitudeMeasures) -> pd.DataFrame:
    """Distribution statistics for peak-to-peak and absolute amplitude (µV)."""
    rows = {
        "peak_to_peak_uv": describe(measures.epoch_p2p_max),
        "peak_abs_uv": describe(measures.epoch_abs_max),
    }
    return pd.DataFrame(rows).T[list(DISTRIBUTION_STATS)]


# --------------------------------------------------------------------------------------
# 3 -- channel-level artifact frequency
# --------------------------------------------------------------------------------------

CHANNEL_QC_COLUMNS: Tuple[str, ...] = (
    "channel",
    "flagged_epochs",
    "pct_epochs_flagged",
    "median_peak_to_peak",
    "max_peak_to_peak",
)


def channel_qc_table(measures: AmplitudeMeasures) -> pd.DataFrame:
    """Per-channel flag frequency, sorted descending.

    A channel is counted for an epoch when *that channel's own* peak-to-peak
    exceeds the threshold -- not merely when the epoch as a whole is flagged.
    The two extra amplitude columns are what a Phase 2 threshold decision will
    actually need, so they are carried alongside the counts.
    """
    over = measures.per_channel_p2p > measures.threshold_uv
    counts = over.sum(axis=0).astype(int)
    n_epochs = measures.n_epochs

    table = pd.DataFrame(
        {
            "channel": list(measures.channels),
            "flagged_epochs": counts,
            "pct_epochs_flagged": 100.0 * counts / n_epochs,
            "median_peak_to_peak": np.median(measures.per_channel_p2p, axis=0),
            "max_peak_to_peak": measures.per_channel_p2p.max(axis=0),
        },
        columns=list(CHANNEL_QC_COLUMNS),
    )
    return table.sort_values(
        ["flagged_epochs", "median_peak_to_peak"], ascending=False, kind="mergesort"
    ).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 4 / 7 -- class-level artifact distribution and its association with class
# --------------------------------------------------------------------------------------


def class_qc_table(epoch_qc: pd.DataFrame, classes: Sequence[str]) -> pd.DataFrame:
    """Epoch count, flagged count and flagged percentage per class.

    ``classes`` fixes the row order and guarantees a class with zero epochs still
    appears, so a missing class can never be mistaken for a clean one.
    """
    rows = []
    for label in classes:
        subset = epoch_qc[epoch_qc["class"] == label]
        n = len(subset)
        flagged = int(subset["artifact_flag"].sum())
        rows.append(
            {
                "class": label,
                "n_epochs": n,
                "n_flagged": flagged,
                "pct_flagged": (100.0 * flagged / n) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ClassAssociation:
    """Whether the artifact flag is associated with the class label."""

    chi2: float
    cramers_v: float
    p_permutation: float
    n_permutations: int
    max_rate_spread_pp: float

    def verdict(self, spread_threshold_pp: float = 25.0, alpha: float = 0.05) -> str:
        """A plain-language reading of the association.

        Two conditions have to agree before contamination is called class-linked:
        a permutation p-value below ``alpha`` *and* a spread between the best and
        worst class rate above ``spread_threshold_pp`` percentage points.  With 23
        epochs neither alone is worth much -- the p-value is underpowered and the
        spread is unstable -- so requiring both is the conservative reading.
        """
        if self.p_permutation < alpha and self.max_rate_spread_pp > spread_threshold_pp:
            return "class-associated"
        if self.max_rate_spread_pp > spread_threshold_pp:
            return "spread is large but not statistically supported at this n"
        return "broadly distributed"


def class_association(
    epoch_qc: pd.DataFrame, *, n_permutations: int = 20000, seed: int = 20260815
) -> ClassAssociation:
    """Test whether flag rate depends on class, by label permutation.

    A chi-square *test* is not trustworthy here: with 23 epochs across three
    classes several expected cell counts fall below 5, which is exactly where the
    asymptotic approximation fails.  The chi-square *statistic* is still a
    perfectly good measure of table-wide deviation, so it is used as the test
    statistic and its null distribution is obtained by shuffling the class labels.
    That makes the p-value exact up to Monte-Carlo error and free of any
    small-count assumption.
    """
    labels = epoch_qc["class"].to_numpy()
    flags = epoch_qc["artifact_flag"].to_numpy().astype(bool)
    classes = np.unique(labels)

    def statistic(lab: np.ndarray) -> float:
        observed = np.array(
            [
                [int(((lab == c) & flags).sum()), int(((lab == c) & ~flags).sum())]
                for c in classes
            ],
            dtype=float,
        )
        total = observed.sum()
        row = observed.sum(axis=1, keepdims=True)
        col = observed.sum(axis=0, keepdims=True)
        expected = row @ col / total
        # Cells with zero expectation contribute nothing and would divide by zero.
        mask = expected > 0
        return float(
            (((observed - expected) ** 2)[mask] / expected[mask]).sum()
        )

    observed_stat = statistic(labels)

    rng = np.random.default_rng(seed)
    permuted = labels.copy()
    hits = 0
    for _ in range(n_permutations):
        rng.shuffle(permuted)
        if statistic(permuted) >= observed_stat - 1e-12:
            hits += 1
    # +1 in numerator and denominator: the observed labelling is itself one of the
    # possible permutations, which keeps the p-value from ever being exactly 0.
    p_value = (hits + 1) / (n_permutations + 1)

    n = len(epoch_qc)
    # Cramer's V for an r x 2 table: min(r-1, 2-1) = 1 when there are >= 2 classes.
    denom = n * max(min(len(classes) - 1, 1), 1)
    cramers_v = float(np.sqrt(observed_stat / denom)) if denom else float("nan")

    rates = [
        100.0 * flags[labels == c].mean() if (labels == c).any() else np.nan
        for c in classes
    ]
    spread = float(np.nanmax(rates) - np.nanmin(rates))

    return ClassAssociation(
        chi2=observed_stat,
        cramers_v=cramers_v,
        p_permutation=float(p_value),
        n_permutations=int(n_permutations),
        max_rate_spread_pp=spread,
    )


# --------------------------------------------------------------------------------------
# 6 -- mu power and lateralisation, flagged vs unflagged
# --------------------------------------------------------------------------------------


def mu_power_per_epoch(
    X: np.ndarray, channels: Sequence[str], channel: str, sfreq: float
) -> np.ndarray:
    """Per-epoch mu-band (8-13 Hz) power at one channel, in V²/Hz.

    Welch PSD with the same 256-sample segment length used by the Phase 1
    spectral figures, then the mean of the PSD over the mu bins.  The input must
    be in volts and **un-normalised**: z-scored data has no meaningful PSD.
    """
    index = list(channels).index(channel)
    freqs, psd = welch(X[:, index, :], fs=sfreq, nperseg=WELCH_NPERSEG, axis=-1)
    lo, hi = MU_BAND_HZ
    band = (freqs >= lo) & (freqs <= hi)
    return psd[:, band].mean(axis=1)


def lateralization_index(c3_power: np.ndarray, c4_power: np.ndarray) -> float:
    """``log(mean C4 mu / mean C3 mu)`` -- the Phase 1.5 group-level definition.

    Averaging power across epochs *before* taking the ratio is what
    ``viz.figures.lateralization_index`` does, so the numbers here can be compared
    with the Phase 1.5 report directly.  Note this is not the mean of the
    per-epoch indices; :func:`per_epoch_lateralization` gives that instead.
    """
    c3, c4 = float(np.mean(c3_power)), float(np.mean(c4_power))
    if c3 <= 0 or c4 <= 0:
        return float("nan")
    return float(np.log(c4 / c3))


def per_epoch_lateralization(
    c3_power: np.ndarray, c4_power: np.ndarray
) -> np.ndarray:
    """``log(C4 / C3)`` computed within each epoch.

    Reported alongside the group-level index because it is the quantity whose
    spread can be compared between the flagged and unflagged sets -- a single
    group-level number has no dispersion to compare.
    """
    return np.log(np.asarray(c4_power, dtype=float) / np.asarray(c3_power, dtype=float))


#: Scalp regions for the 64-channel EEGMMIDB montage.  A flat count of flagged
#: channels says how *much* an epoch exceeded the threshold; it says nothing about
#: *where*, and where is what distinguishes an ocular artifact from occipital
#: alpha from a genuine sensorimotor event.  Grouping is by electrode position
#: only -- it encodes no claim about what any region's activity means.
CHANNEL_REGIONS: Dict[str, Tuple[str, ...]] = {
    "frontopolar / anterior-frontal": (
        "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    ),
    "frontal": ("F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FT8"),
    "sensorimotor (fronto-central / central)": (
        "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    ),
    "temporal": ("T7", "T8", "T9", "T10", "TP7", "TP8"),
    "centro-parietal / parietal": (
        "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    ),
    "parieto-occipital / occipital": (
        "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
    ),
}


def region_of(channel: str) -> str:
    """Which :data:`CHANNEL_REGIONS` entry a channel belongs to."""
    for region, members in CHANNEL_REGIONS.items():
        if channel in members:
            return region
    return "unassigned"


def region_qc_table(measures: AmplitudeMeasures) -> pd.DataFrame:
    """Flag frequency aggregated by scalp region.

    ``flag_events`` counts channel-by-epoch threshold crossings, so a region with
    many electrodes can accumulate more of them; ``pct_of_all_flag_events``
    normalises that against the total, and ``pct_channel_epochs_flagged``
    normalises against the region's own size, which is the rate that is
    comparable between regions.
    """
    over = measures.per_channel_p2p > measures.threshold_uv
    regions = [region_of(c) for c in measures.channels]
    total_events = int(over.sum())

    rows = []
    for region in list(CHANNEL_REGIONS) + (["unassigned"] if "unassigned" in regions else []):
        cols = [i for i, r in enumerate(regions) if r == region]
        if not cols:
            continue
        events = int(over[:, cols].sum())
        cell_count = len(cols) * measures.n_epochs
        rows.append(
            {
                "region": region,
                "n_channels": len(cols),
                "flag_events": events,
                "pct_of_all_flag_events": 100.0 * events / total_events if total_events else 0.0,
                "pct_channel_epochs_flagged": 100.0 * events / cell_count,
                "median_peak_to_peak": float(np.median(measures.per_channel_p2p[:, cols])),
                "max_peak_to_peak": float(measures.per_channel_p2p[:, cols].max()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("pct_channel_epochs_flagged", ascending=False)
        .reset_index(drop=True)
    )


def dominant_region_per_epoch(
    epoch_qc: pd.DataFrame, measures: AmplitudeMeasures
) -> pd.DataFrame:
    """For each flagged epoch, which region contributed the most crossings.

    This is what turns "18 epochs are flagged" into something interpretable: an
    epoch whose crossings are all frontopolar and one whose crossings are all
    occipital are not the same finding, even though the flag treats them
    identically.
    """
    over = measures.per_channel_p2p > measures.threshold_uv
    regions = np.array([region_of(c) for c in measures.channels])

    rows = []
    for position in range(measures.n_epochs):
        if not over[position].any():
            continue
        hit_regions = regions[over[position]]
        names, counts = np.unique(hit_regions, return_counts=True)
        order = np.argsort(-counts)
        worst_channel = measures.channels[int(np.argmax(measures.per_channel_p2p[position]))]
        rows.append(
            {
                "epoch_id": epoch_qc.iloc[position]["epoch_id"],
                "class": epoch_qc.iloc[position]["class"],
                "peak_to_peak_max": float(measures.epoch_p2p_max[position]),
                "n_flagged_channels": int(over[position].sum()),
                "dominant_region": str(names[order[0]]),
                "dominant_region_channels": int(counts[order[0]]),
                "largest_channel": worst_channel,
            }
        )
    return pd.DataFrame(rows)


def pairwise_class_contrasts(epoch_qc: pd.DataFrame) -> pd.DataFrame:
    """Flag rate for each pair of classes, plus task-vs-rest, with Fisher's exact.

    The single 3-class test answers "does the flag rate depend on class at all".
    That is not the question Phase 2 needs answered.  A decoder is trained on
    *contrasts*, so what matters is whether the flag rate differs across the
    specific contrast it will learn -- left vs right for the binary decoder, task
    vs rest for the three-state engine.  A confound in one is not a confound in
    the other, and the omnibus test cannot tell them apart.

    Fisher's exact test is used rather than chi-square because every one of these
    2x2 tables has cells in the single digits.
    """
    from scipy.stats import fisher_exact

    labels = epoch_qc["class"].to_numpy()
    flags = epoch_qc["artifact_flag"].to_numpy().astype(bool)

    def counts(mask: np.ndarray) -> Tuple[int, int]:
        return int(flags[mask].sum()), int(mask.sum())

    present = [c for c in pd.unique(labels)]
    contrasts: list[Tuple[str, np.ndarray, str, np.ndarray]] = []
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            contrasts.append((a, labels == a, b, labels == b))
    if "REST" in present:
        contrasts.append(
            ("TASK (left+right)", labels != "REST", "REST", labels == "REST")
        )

    rows = []
    for name_a, mask_a, name_b, mask_b in contrasts:
        flagged_a, n_a = counts(mask_a)
        flagged_b, n_b = counts(mask_b)
        table = [[flagged_a, n_a - flagged_a], [flagged_b, n_b - flagged_b]]
        _, p_value = fisher_exact(table)
        rate_a = 100.0 * flagged_a / n_a if n_a else float("nan")
        rate_b = 100.0 * flagged_b / n_b if n_b else float("nan")
        rows.append(
            {
                "contrast": f"{name_a} vs {name_b}",
                "n_a": n_a,
                "flagged_a": flagged_a,
                "rate_a_pct": rate_a,
                "n_b": n_b,
                "flagged_b": flagged_b,
                "rate_b_pct": rate_b,
                "spread_pp": abs(rate_a - rate_b),
                "fisher_p": float(p_value),
            }
        )
    return pd.DataFrame(rows).sort_values("spread_pp", ascending=False).reset_index(drop=True)


def permutation_difference(
    values: np.ndarray,
    group: np.ndarray,
    *,
    n_permutations: int = 20000,
    seed: int = 20260815,
) -> Tuple[float, float]:
    """Two-sided permutation test on a difference of means.

    Returns ``(observed_difference, p_value)`` where the difference is
    ``mean(values[group]) - mean(values[~group])``.  Used for the flagged vs
    unflagged lateralisation comparison, where one group has five members and no
    parametric test would be defensible.
    """
    values = np.asarray(values, dtype=float)
    group = np.asarray(group, dtype=bool)
    if group.all() or (~group).all():
        return float("nan"), float("nan")

    observed = float(values[group].mean() - values[~group].mean())
    rng = np.random.default_rng(seed)
    shuffled = group.copy()
    hits = 0
    for _ in range(n_permutations):
        rng.shuffle(shuffled)
        if abs(values[shuffled].mean() - values[~shuffled].mean()) >= abs(observed) - 1e-15:
            hits += 1
    return observed, float((hits + 1) / (n_permutations + 1))


def amplitude_signal_correlation(
    epoch_p2p: np.ndarray, quantities: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Spearman correlation between epoch amplitude and each named quantity.

    The point of this is to ask whether the flag is tracking *noise* or *signal*.
    If epoch peak-to-peak correlates strongly with sensorimotor mu power, then a
    threshold that removes high-amplitude epochs also removes high-mu epochs --
    which is to say it would delete the feature the decoder needs.  Spearman
    rather than Pearson: n is 23 and neither quantity is guaranteed normal.
    """
    from scipy.stats import spearmanr

    rows = []
    for name, values in quantities.items():
        rho, p_value = spearmanr(epoch_p2p, values)
        rows.append({"quantity": name, "spearman_rho": float(rho), "p_value": float(p_value)})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class LateralizationSummary:
    """C3/C4 mu power and lateralisation for one subset of epochs."""

    name: str
    n_epochs: int
    c3_mu_power: float           # V^2/Hz, mean across epochs
    c4_mu_power: float           # V^2/Hz, mean across epochs
    lateralization: float        # log(mean C4 / mean C3)
    per_epoch_mean: float        # mean of the per-epoch log ratios
    per_epoch_std: float         # sample sd of the per-epoch log ratios

    def as_row(self) -> Dict[str, object]:
        return {
            "subset": self.name,
            "n_epochs": self.n_epochs,
            "c3_mu_power": self.c3_mu_power,
            "c4_mu_power": self.c4_mu_power,
            "lateralization_index": self.lateralization,
            "per_epoch_li_mean": self.per_epoch_mean,
            "per_epoch_li_std": self.per_epoch_std,
        }


def summarize_lateralization(
    name: str, c3_power: np.ndarray, c4_power: np.ndarray
) -> LateralizationSummary:
    """Build a :class:`LateralizationSummary` from per-epoch mu powers."""
    n = int(len(c3_power))
    per_epoch = per_epoch_lateralization(c3_power, c4_power) if n else np.array([])
    return LateralizationSummary(
        name=name,
        n_epochs=n,
        c3_mu_power=float(np.mean(c3_power)) if n else float("nan"),
        c4_mu_power=float(np.mean(c4_power)) if n else float("nan"),
        lateralization=lateralization_index(c3_power, c4_power) if n else float("nan"),
        per_epoch_mean=float(np.mean(per_epoch)) if n else float("nan"),
        per_epoch_std=float(np.std(per_epoch, ddof=1)) if n > 1 else float("nan"),
    )


# --------------------------------------------------------------------------------------
# 5 -- picking the representative epochs
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Representative:
    """One epoch chosen to illustrate a point on the amplitude distribution."""

    key: str            # "clean" | "median" | "high"
    banner: str         # the label the figure must carry
    position: int       # row index into the epoch arrays
    epoch_id: str
    class_label: str
    peak_to_peak_uv: float
    artifact_flag: bool
    artifact_channel_count: int


#: The three figures Phase 1.6 requires, and the wording each must carry.
#: "POTENTIAL ARTIFACT" is deliberate: amplitude alone does not establish that a
#: signal is artefactual, and the report must not claim more than was measured.
REPRESENTATIVE_BANNERS: Dict[str, str] = {
    "clean": "CLEAN",
    "median": "MEDIAN",
    "high": "HIGH-AMPLITUDE / POTENTIAL ARTIFACT",
}


def pick_representatives(
    epoch_qc: pd.DataFrame, measures: AmplitudeMeasures
) -> Dict[str, Representative]:
    """Cleanest, median-quality and most contaminated epoch, by peak-to-peak.

    The median epoch is the one holding the middle rank, not a synthetic average,
    so all three figures show data that actually exists.  With an even number of
    epochs the lower of the two middle ranks is taken.
    """
    order = np.argsort(measures.epoch_p2p_max, kind="mergesort")
    picks = {
        "clean": int(order[0]),
        "median": int(order[(len(order) - 1) // 2]),
        "high": int(order[-1]),
    }

    out: Dict[str, Representative] = {}
    for key, position in picks.items():
        row = epoch_qc.iloc[position]
        out[key] = Representative(
            key=key,
            banner=REPRESENTATIVE_BANNERS[key],
            position=position,
            epoch_id=str(row["epoch_id"]),
            class_label=str(row["class"]),
            peak_to_peak_uv=float(row["peak_to_peak_max"]),
            artifact_flag=bool(row["artifact_flag"]),
            artifact_channel_count=int(row["artifact_channel_count"]),
        )
    return out
