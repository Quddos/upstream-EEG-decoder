"""Phase 5B -- run-level lateralisation diagnostics.

Phase 5 established that the persistent below-chance subgroup lacks consistent
sensorimotor mu-power lateralisation. It could not say whether that is a stable
property of the person or a property of the session, because every diagnostic was
computed by pooling a subject's three runs.

This module recomputes the *same* Phase 5 lateralisation quantities **within each
run separately**, so R04, R08 and R12 become three repeated measurements of the
same subject. Nothing about the definition changes; only the window over which it
is computed.

Reuse, not modification
-----------------------
Every primitive is imported unchanged from the frozen
:mod:`src.features.diagnostics` -- the mu band, the PSD settings, ``cohens_d`` and
``fisher_ratio``. This module adds a grouping level and nothing else, so a
run-level number and the Phase 5 subject-level number are the same statistic at
two granularities.

The statistical cost of doing this
----------------------------------
A run contains 15 task epochs, roughly 7 LEFT and 8 RIGHT. A Cohen's d estimated
from 7-versus-8 samples is extremely noisy: its sampling standard error is close to
0.5, which is the same size as the "usable lateralisation" threshold itself. Run
level is therefore the right *unit* for asking about session dependence and a poor
*estimator* of any individual session's true effect.

That is not a reason to avoid the analysis -- it is a reason to interpret it
against a null. :func:`usable_rate_null` shuffles labels within each run and
recomputes the criterion, which gives the rate at which a run passes by chance.
Without that reference, "one of three runs was usable" cannot be distinguished
from noise.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.physionet_spec import LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY
from src.features.diagnostics import (
    LEFT_HEMISPHERE,
    MU_BAND,
    RIGHT_HEMISPHERE,
    band_mean,
    cohens_d,
    epoch_psd,
    fisher_ratio,
)

#: The usable-lateralisation criterion, carried over unchanged from Phase 5:
#: a Cohen's d above 0.5 in magnitude AND a separation in the contralateral
#: direction (RIGHT-hand imagery yielding a higher log(C4/C3) than LEFT).
#: It was fixed before Phase 5B and is not re-tuned here.
USABLE_D_THRESHOLD: float = 0.5


def lateralisation_per_epoch(
    X: np.ndarray, channels: Sequence[str], sfreq: float
) -> np.ndarray:
    """``log(C4 mu / C3 mu)`` per epoch -- the frozen Phase 2/5 definition."""
    channels = list(channels)
    freqs, psd = epoch_psd(X, sfreq)
    mu = band_mean(psd, freqs, MU_BAND)
    c3 = np.log(np.maximum(mu[:, channels.index(LEFT_HEMISPHERE)], 1e-30))
    c4 = np.log(np.maximum(mu[:, channels.index(RIGHT_HEMISPHERE)], 1e-30))
    return c4 - c3


def run_level_diagnostics(
    X: np.ndarray,
    metadata: pd.DataFrame,
    channels: Sequence[str],
    sfreq: float,
) -> pd.DataFrame:
    """One row per (subject, run).

    ``X`` must be un-normalised volts; the load-time per-channel z-score would
    equalise C3 and C4 power and erase the quantity being measured.
    """
    lateralisation = lateralisation_per_epoch(X, channels, sfreq)
    labels = metadata["class_label"].to_numpy()
    subjects = metadata["subject_id"].to_numpy()
    runs = metadata["run"].to_numpy()

    rows: List[Dict[str, object]] = []
    for subject in sorted(pd.unique(subjects)):
        for run in sorted(pd.unique(runs[subjects == subject])):
            in_run = (subjects == subject) & (runs == run)
            left = lateralisation[in_run & (labels == LEFT_HAND_IMAGERY)]
            right = lateralisation[in_run & (labels == RIGHT_HAND_IMAGERY)]

            separation = float(right.mean() - left.mean()) if len(left) and len(right) else np.nan
            d = cohens_d(right, left)
            usable = bool(np.isfinite(d) and abs(d) > USABLE_D_THRESHOLD and separation > 0)

            rows.append(
                {
                    "subject_id": subject,
                    "run": int(run),
                    "run_id": f"R{int(run):02d}",
                    "n_left": int(len(left)),
                    "n_right": int(len(right)),
                    "lat_mean_left": float(left.mean()) if len(left) else np.nan,
                    "lat_mean_right": float(right.mean()) if len(right) else np.nan,
                    "lat_separation": separation,
                    "lat_d": d,
                    "lat_sign": int(np.sign(separation)) if np.isfinite(separation) else 0,
                    "expected_direction": bool(separation > 0) if np.isfinite(separation) else False,
                    "fisher_lateralisation": fisher_ratio(right, left),
                    "usable_lateralisation": usable,
                }
            )
    return pd.DataFrame(rows)


def summarise_subject(runs: pd.DataFrame) -> pd.DataFrame:
    """Collapse run-level rows to one row per subject, preserving the repeats."""
    rows: List[Dict[str, object]] = []
    for subject, group in runs.groupby("subject_id"):
        d_values = group["lat_d"].to_numpy(float)
        signs = group["lat_sign"].to_numpy(int)
        rows.append(
            {
                "subject_id": subject,
                "n_runs": int(len(group)),
                "n_usable_runs": int(group["usable_lateralisation"].sum()),
                "usable_fraction": float(group["usable_lateralisation"].mean()),
                "n_expected_direction": int(group["expected_direction"].sum()),
                "sign_consistency": float(
                    max((signs > 0).sum(), (signs < 0).sum()) / len(signs)
                ),
                "lat_d_mean": float(np.nanmean(d_values)),
                "lat_d_sd": float(np.nanstd(d_values, ddof=1)) if len(d_values) > 1 else np.nan,
                "lat_d_min": float(np.nanmin(d_values)),
                "lat_d_max": float(np.nanmax(d_values)),
                "lat_d_range": float(np.nanmax(d_values) - np.nanmin(d_values)),
                "all_runs_usable": bool(group["usable_lateralisation"].all()),
                "no_run_usable": bool((~group["usable_lateralisation"]).all()),
                "mixed_across_runs": bool(
                    group["usable_lateralisation"].any()
                    and not group["usable_lateralisation"].all()
                ),
            }
        )
    return pd.DataFrame(rows)


def usable_rate_null(
    X: np.ndarray,
    metadata: pd.DataFrame,
    channels: Sequence[str],
    sfreq: float,
    *,
    n_permutations: int = 2000,
    seed: int = 20260818,
) -> Dict[str, float]:
    """How often a run passes the usable criterion when labels are shuffled.

    Labels are permuted **within each run**, preserving the run structure and each
    run's class proportions, and the criterion is re-evaluated. This gives the
    false-positive rate of "this run had usable lateralisation" at the real sample
    size, which is the reference any observed rate has to be read against.
    """
    lateralisation = lateralisation_per_epoch(X, channels, sfreq)
    labels = metadata["class_label"].to_numpy()
    subjects = metadata["subject_id"].to_numpy()
    runs = metadata["run"].to_numpy()

    blocks: List[Tuple[np.ndarray, np.ndarray]] = []
    for subject in pd.unique(subjects):
        for run in pd.unique(runs[subjects == subject]):
            in_run = (subjects == subject) & (runs == run)
            task = in_run & np.isin(labels, [LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY])
            blocks.append((lateralisation[task], labels[task] == RIGHT_HAND_IMAGERY))

    rng = np.random.default_rng(seed)
    rates = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        passed = 0
        for values, is_right in blocks:
            shuffled = rng.permutation(is_right)
            right, left = values[shuffled], values[~shuffled]
            separation = right.mean() - left.mean()
            d = cohens_d(right, left)
            if np.isfinite(d) and abs(d) > USABLE_D_THRESHOLD and separation > 0:
                passed += 1
        rates[i] = passed / len(blocks)

    return {
        "n_permutations": int(n_permutations),
        "n_runs": int(len(blocks)),
        "null_mean_usable_rate": float(rates.mean()),
        "null_sd": float(rates.std(ddof=1)),
        "null_p95": float(np.percentile(rates, 95)),
    }
