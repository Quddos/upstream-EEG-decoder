"""Phase 5 -- per-subject signal diagnostics.

Five phases have raised the mean cross-subject accuracy and none has moved the
15-22% of subjects who sit at or below chance. This module stops asking which
model and asks whether the signal exists in those subjects at all.

Everything here is **descriptive**. No model is trained, nothing is fitted, and no
subject's diagnostic depends on any other subject. Each quantity is a summary
statistic of one subject's own epochs, so every diagnostic could have been computed
before any decoder existed -- which is what makes comparing them against the
pre-existing LOSOCV outcome a fair test rather than a circular one.

Pre-registration
----------------
The four diagnostic families and their formulas were written here **before** any
association with LOSOCV accuracy was examined, and the below-chance group was
locked from the frozen Phase 4B result first (``phase5_outcome_definition.json``).
Anything added afterwards is labelled exploratory in the output.

An honesty note on independence
-------------------------------
These diagnostics are *representation-independent* in the sense that they involve
no learned transform, no spatial filter and no classifier. They are **not**
independent of the *measurement*: they are band powers at sensorimotor electrodes,
which is the same physical quantity the frozen features summarise. A diagnostic
that shares the decoder's blind spot cannot reveal that blind spot. That limit is
stated in the report rather than papered over, and it is why a null result here is
weaker evidence than a positive one.

Terminology
-----------
The epoch window is ``[0, 4)`` from cue onset and nothing before the cue is stored,
so there is **no pre-cue baseline** and classical event-related desynchronisation
cannot be computed. What *is* available is a contrast between task epochs and the
REST epochs of the same subject, and it is named ``task_vs_rest_mu_change``
throughout for that reason. It is a related but different quantity and must not be
reported as ERD.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch

from src.data.physionet_spec import LEFT_HAND_IMAGERY, REST, RIGHT_HAND_IMAGERY

#: Mu band, matching every previous phase.
MU_BAND: Tuple[float, float] = (8.0, 13.0)

#: Flanking band used as the SNR denominator. The preprocessing passband is
#: 8-30 Hz, so nothing below 8 Hz survives and the only usable flank is above the
#: beta transition. 20-30 Hz is chosen because it is well clear of the 13 Hz mu
#: edge while staying inside the passband. This constraint follows from the frozen
#: filter and is reported as a limitation.
FLANK_BAND: Tuple[float, float] = (20.0, 30.0)

#: Welch segment length, identical to every previous phase.
WELCH_NPERSEG: int = 256

LEFT_HEMISPHERE = "C3"
RIGHT_HEMISPHERE = "C4"


def band_mean(psd: np.ndarray, freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        raise ValueError(f"no PSD bins inside {band}")
    return psd[..., mask].mean(axis=-1)


def epoch_psd(X: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD per epoch and channel, from un-normalised volts."""
    freqs, psd = welch(X, fs=sfreq, nperseg=min(WELCH_NPERSEG, X.shape[2]), axis=-1)
    return freqs, psd


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference, ``(mean(a) - mean(b)) / pooled sd``."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def fisher_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """``(mean_a - mean_b)^2 / (var_a + var_b)`` -- univariate class separability.

    A descriptive statistic, not a fitted discriminant: no covariance is inverted
    and no weight is learned, so this cannot be a model in disguise.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    denominator = a.var(ddof=1) + b.var(ddof=1)
    if denominator <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) ** 2 / denominator)


def subject_diagnostics(
    X: np.ndarray,
    metadata: pd.DataFrame,
    channels: Sequence[str],
    sfreq: float,
) -> pd.DataFrame:
    """One row of diagnostics per subject.

    ``X`` is ``(n_epochs, n_channels, n_times)`` in **volts, un-normalised**. The
    per-channel z-score applied at load time by default would equalise channel
    power and destroy every quantity below.
    """
    channels = list(channels)
    c3 = channels.index(LEFT_HEMISPHERE)
    c4 = channels.index(RIGHT_HEMISPHERE)

    freqs, psd = epoch_psd(X, sfreq)
    mu = band_mean(psd, freqs, MU_BAND)
    flank = band_mean(psd, freqs, FLANK_BAND)

    log_mu_c3 = np.log(np.maximum(mu[:, c3], 1e-30))
    log_mu_c4 = np.log(np.maximum(mu[:, c4], 1e-30))
    lateralisation = log_mu_c4 - log_mu_c3
    log_mu_all = np.log(np.maximum(mu, 1e-30))

    labels = metadata["class_label"].to_numpy()
    subjects = metadata["subject_id"].to_numpy()
    runs = metadata["run"].to_numpy()

    rows: List[Dict[str, object]] = []
    for subject in sorted(pd.unique(subjects)):
        s = subjects == subject
        is_left = s & (labels == LEFT_HAND_IMAGERY)
        is_right = s & (labels == RIGHT_HAND_IMAGERY)
        is_rest = s & (labels == REST)
        is_task = is_left | is_right

        row: Dict[str, object] = {"subject_id": subject, "n_epochs": int(s.sum())}

        # --- 1. mu-band SNR --------------------------------------------------
        # Mu power relative to the 20-30 Hz flank. A subject with a genuine mu
        # rhythm shows a peak here; one without has a flat spectrum and a ratio
        # near 1.
        for name, index in ((LEFT_HEMISPHERE, c3), (RIGHT_HEMISPHERE, c4)):
            row[f"mu_snr_{name}"] = float(np.mean(mu[s, index] / flank[s, index]))
            for cls, mask in (("left", is_left), ("right", is_right), ("rest", is_rest)):
                row[f"mu_snr_{name}_{cls}"] = (
                    float(np.mean(mu[mask, index] / flank[mask, index]))
                    if mask.any() else float("nan")
                )
        row["mu_snr_mean"] = float(
            (row[f"mu_snr_{LEFT_HEMISPHERE}"] + row[f"mu_snr_{RIGHT_HEMISPHERE}"]) / 2
        )

        # --- 2. task-vs-rest mu-power change ---------------------------------
        # NOT event-related desynchronisation: this epoching has no pre-cue
        # baseline. Negative means task epochs carry less mu power than that
        # subject's own rest epochs, the direction motor imagery predicts.
        for name, logmu in ((LEFT_HEMISPHERE, log_mu_c3), (RIGHT_HEMISPHERE, log_mu_c4)):
            if is_rest.any() and is_task.any():
                row[f"task_vs_rest_mu_change_{name}"] = float(
                    logmu[is_task].mean() - logmu[is_rest].mean()
                )
                row[f"task_vs_rest_d_{name}"] = cohens_d(logmu[is_task], logmu[is_rest])
            else:
                row[f"task_vs_rest_mu_change_{name}"] = float("nan")
                row[f"task_vs_rest_d_{name}"] = float("nan")
        row["task_vs_rest_mu_change_mean"] = float(
            np.nanmean([
                row[f"task_vs_rest_mu_change_{LEFT_HEMISPHERE}"],
                row[f"task_vs_rest_mu_change_{RIGHT_HEMISPHERE}"],
            ])
        )
        # Contralateral pattern: LEFT-hand imagery should suppress mu at C4.
        row["contra_change_left_at_C4"] = (
            float(log_mu_c4[is_left].mean() - log_mu_c4[is_rest].mean())
            if is_left.any() and is_rest.any() else float("nan")
        )
        row["contra_change_right_at_C3"] = (
            float(log_mu_c3[is_right].mean() - log_mu_c3[is_rest].mean())
            if is_right.any() and is_rest.any() else float("nan")
        )

        # --- 3. lateralisation: magnitude vs consistency ----------------------
        # Magnitude is how far apart the classes sit; consistency is whether they
        # sit apart the same way every time. A subject can show a large mean
        # separation driven by one run and no stable pattern, and only the second
        # is usable by a decoder trained on other people.
        left_lat = lateralisation[is_left]
        right_lat = lateralisation[is_right]
        row["lat_mean_left"] = float(left_lat.mean()) if len(left_lat) else float("nan")
        row["lat_mean_right"] = float(right_lat.mean()) if len(right_lat) else float("nan")
        row["lat_separation"] = row["lat_mean_right"] - row["lat_mean_left"]
        row["lat_separation_abs"] = abs(row["lat_separation"])
        row["lat_d"] = cohens_d(right_lat, left_lat)
        row["lat_pooled_sd"] = float(np.std(lateralisation[is_task], ddof=1))

        signs = []
        for run in sorted(pd.unique(runs[s])):
            in_run = s & (runs == run)
            left_run = lateralisation[in_run & (labels == LEFT_HAND_IMAGERY)]
            right_run = lateralisation[in_run & (labels == RIGHT_HAND_IMAGERY)]
            if len(left_run) and len(right_run):
                signs.append(np.sign(right_run.mean() - left_run.mean()))
        row["lat_n_runs"] = len(signs)
        row["lat_runs_positive"] = int(sum(1 for x in signs if x > 0))
        row["lat_sign_consistency"] = (
            float(max(sum(1 for x in signs if x > 0), sum(1 for x in signs if x < 0)) / len(signs))
            if signs else float("nan")
        )
        row["lat_expected_direction"] = bool(row["lat_separation"] > 0)

        # Epoch-level ordering relative to the subject's own midpoint. A threshold
        # count, not a fitted model.
        if len(left_lat) and len(right_lat):
            midpoint = (left_lat.mean() + right_lat.mean()) / 2
            correct = np.concatenate([left_lat < midpoint, right_lat >= midpoint])
            row["lat_epoch_agreement"] = float(correct.mean())
        else:
            row["lat_epoch_agreement"] = float("nan")

        # --- 4. within-subject class separability -----------------------------
        row["fisher_lateralisation"] = fisher_ratio(right_lat, left_lat)
        row["fisher_mu_C3"] = fisher_ratio(log_mu_c3[is_right], log_mu_c3[is_left])
        row["fisher_mu_C4"] = fisher_ratio(log_mu_c4[is_right], log_mu_c4[is_left])

        per_channel = np.array([
            fisher_ratio(log_mu_all[is_right, i], log_mu_all[is_left, i])
            for i in range(log_mu_all.shape[1])
        ])
        row["fisher_roi_mean"] = float(np.nanmean(per_channel))
        row["fisher_roi_max"] = float(np.nanmax(per_channel))
        row["fisher_roi_argmax_channel"] = channels[int(np.nanargmax(per_channel))]

        rows.append(row)

    return pd.DataFrame(rows)


#: Diagnostics registered in advance, grouped by the brief's four families.
#: Anything not listed here is exploratory and is labelled as such in the report.
PREREGISTERED: Dict[str, Tuple[str, ...]] = {
    "1_mu_snr": ("mu_snr_C3", "mu_snr_C4", "mu_snr_mean"),
    "2_task_vs_rest_mu_change": (
        "task_vs_rest_mu_change_C3",
        "task_vs_rest_mu_change_C4",
        "task_vs_rest_mu_change_mean",
    ),
    "3_lateralisation": (
        "lat_separation", "lat_d", "lat_sign_consistency", "lat_epoch_agreement",
    ),
    "4_separability": ("fisher_lateralisation", "fisher_roi_mean", "fisher_roi_max"),
}

#: Added for interpretation only; reported separately and never used to support a
#: conclusion on its own.
EXPLORATORY: Tuple[str, ...] = (
    "lat_separation_abs", "lat_pooled_sd", "contra_change_left_at_C4",
    "contra_change_right_at_C3", "fisher_mu_C3", "fisher_mu_C4",
    "task_vs_rest_d_C3", "task_vs_rest_d_C4",
)
