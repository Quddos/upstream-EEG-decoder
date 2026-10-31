"""Phase 1.6 -- run the real-EEG quality-control diagnostic.

    python -m src.cli.qc_report --subject 1 --run 4

Reads the epochs the Phase 1 pipeline already wrote, measures them, and emits

    data/metadata/epoch_qc.csv
    data/metadata/channel_qc.csv
    reports/figures/08_qc_epoch_clean.png
    reports/figures/09_qc_epoch_median.png
    reports/figures/10_qc_epoch_high_amplitude.png
    reports/figures/11_qc_amplitude_distribution.png

plus a full text summary on stdout, which is the source the QC report is written
from.  ``--json`` additionally dumps every computed number so the report can be
regenerated without re-deriving anything by hand.

This entry point is **diagnostic only**.  It does not rebuild the dataset, does
not touch any preprocessing parameter, does not change the 150 µV threshold and
does not drop a single epoch -- it only reads and describes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.cli._common import setup_logging
from src.config import DatasetConfig
from src.data.physionet_spec import PHASE1_CLASSES
from src.io.store import subject_archive_path
from src.qc import figures as qc_figures
from src.qc.quality import (
    amplitude_signal_correlation,
    class_association,
    class_qc_table,
    channel_qc_table,
    distribution_table,
    dominant_region_per_epoch,
    epoch_qc_table,
    measure_amplitudes,
    mu_power_per_epoch,
    pairwise_class_contrasts,
    per_epoch_lateralization,
    permutation_difference,
    pick_representatives,
    region_qc_table,
    summarize_lateralization,
)

EPOCH_QC_FILENAME = "epoch_qc.csv"
CHANNEL_QC_FILENAME = "channel_qc.csv"


def load_run(
    config: DatasetConfig, subject: int, run: int, data_root: Optional[Path]
):
    """Load one recording's epochs, un-normalised, straight from the archive.

    Deliberately not :func:`store.load_dataset`: that applies the per-epoch
    z-score, which destroys the amplitude information this whole phase is about.
    The stored array is in volts and is exactly what the pipeline produced.
    """
    paths = config.resolved_paths
    index = pd.read_csv(paths.metadata / "epoch_index.csv")

    rows = index[(index["subject"] == subject) & (index["run"] == run)]
    if rows.empty:
        raise SystemExit(
            f"no epochs for subject {subject} run {run} in {paths.metadata}; "
            f"build the dataset first"
        )

    subject_id = str(rows["subject_id"].iloc[0])
    processed_dir = (
        Path(data_root).resolve() / "data" / "processed" if data_root else paths.processed
    )
    archive = subject_archive_path(processed_dir, subject_id)
    if not archive.exists():
        raise SystemExit(
            f"{archive} is missing. The processed epochs are gitignored; pass "
            f"--data-root to point at the tree that holds data/processed."
        )

    with np.load(archive, allow_pickle=False) as bundle:
        X_all = bundle["X"].astype(np.float64)
        run_all = bundle["run"]
        channels = tuple(str(c) for c in bundle["channels"])
        times = bundle["times"]

    # The archive holds every run of the subject, concatenated in the order the
    # index lists them, so a positional mask on `run` selects the right block.
    mask = run_all == run
    if int(mask.sum()) != len(rows):
        raise SystemExit(
            f"{archive} holds {int(mask.sum())} epochs for run {run} but the index "
            f"lists {len(rows)} -- the dataset is inconsistent, rebuild it"
        )

    return X_all[mask], rows.reset_index(drop=True), channels, times


def _fmt_table(frame: pd.DataFrame, floatfmt: str = "%.2f") -> str:
    return frame.to_string(index=False, float_format=lambda v: floatfmt % v)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1.6 real-EEG quality control (diagnostic only)"
    )
    parser.add_argument("--subject", type=int, default=1, help="subject number")
    parser.add_argument("--run", type=int, default=4, help="run number")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "directory holding data/processed, if it is not under the project "
            "root (the processed archives are gitignored, so a worktree may not "
            "have them)"
        ),
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=20000,
        help="label permutations for the class-association test",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="also write every number to this file"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    config = DatasetConfig()
    threshold_v = config.preprocessing.reject_log_threshold_v  # inherited, unchanged
    paths = config.resolved_paths
    paths.mkdirs()

    X, index_rows, channels, times = load_run(
        config, args.subject, args.run, args.data_root
    )
    subject_id = str(index_rows["subject_id"].iloc[0])
    run_id = str(index_rows["run_id"].iloc[0])
    sfreq = float(index_rows["sampling_rate"].iloc[0])

    measures = measure_amplitudes(X, channels, threshold_v)

    # --- 1. epoch-level QC table ---------------------------------------------
    epoch_qc = epoch_qc_table(index_rows, measures)
    epoch_qc_path = paths.metadata / EPOCH_QC_FILENAME
    epoch_qc.to_csv(epoch_qc_path, index=False)

    # --- 2. distribution statistics ------------------------------------------
    distribution = distribution_table(measures)

    # --- 3. channel-level artifact frequency ---------------------------------
    channel_qc = channel_qc_table(measures)
    channel_qc_path = paths.metadata / CHANNEL_QC_FILENAME
    channel_qc.to_csv(channel_qc_path, index=False)
    region_qc = region_qc_table(measures)
    epoch_regions = dominant_region_per_epoch(epoch_qc, measures)

    # --- 4. class-level artifact distribution --------------------------------
    class_qc = class_qc_table(epoch_qc, PHASE1_CLASSES)

    # --- 7. is the flag associated with class? -------------------------------
    association = class_association(
        epoch_qc, n_permutations=args.permutations, seed=config.seed
    )
    contrasts = pairwise_class_contrasts(epoch_qc)

    # --- 5. representative figures -------------------------------------------
    representatives = pick_representatives(epoch_qc, measures)
    figure_paths: Dict[str, Path] = {}
    for key, rep in representatives.items():
        figure_paths[key] = qc_figures.plot_representative_epoch(
            X,
            times,
            channels,
            measures,
            rep,
            paths.figures,
            qc_figures.REPRESENTATIVE_FILENAMES[key],
            subject=subject_id,
            run=run_id,
        )
    figure_paths["distribution"] = qc_figures.plot_amplitude_distribution(
        measures, epoch_qc, paths.figures
    )

    # --- 6. flagged vs unflagged lateralisation ------------------------------
    c3 = mu_power_per_epoch(X, channels, "C3", sfreq)
    c4 = mu_power_per_epoch(X, channels, "C4", sfreq)
    flagged = measures.artifact_flag

    subsets = [
        summarize_lateralization("all", c3, c4),
        summarize_lateralization("flagged", c3[flagged], c4[flagged]),
        summarize_lateralization("unflagged", c3[~flagged], c4[~flagged]),
    ]
    lateralization = pd.DataFrame([s.as_row() for s in subsets])

    # Per class as well: the left-vs-right *difference* is the only lateralisation
    # quantity Phase 1.5 established as meaningful, so a flagged/unflagged
    # comparison that ignores class could be confounded by class composition.
    # The reference scale for "is the flagged/unflagged shift large?": the
    # left-vs-right separation is the effect the decoder has to detect, so a shift
    # in the lateralisation index only means something relative to it.
    class_lateralization = pd.DataFrame(
        [
            summarize_lateralization(
                label,
                c3[(epoch_qc["class"] == label).to_numpy()],
                c4[(epoch_qc["class"] == label).to_numpy()],
            ).as_row()
            for label in PHASE1_CLASSES
        ]
    ).rename(columns={"subset": "class"})
    by_label = dict(
        zip(class_lateralization["class"], class_lateralization["lateralization_index"])
    )
    left_right_separation = float(
        by_label.get("RIGHT_HAND_IMAGERY", np.nan) - by_label.get("LEFT_HAND_IMAGERY", np.nan)
    )

    per_class_rows: List[Dict[str, object]] = []
    for label in PHASE1_CLASSES:
        in_class = (epoch_qc["class"] == label).to_numpy()
        for subset_name, subset_mask in (
            ("flagged", in_class & flagged),
            ("unflagged", in_class & ~flagged),
        ):
            summary = summarize_lateralization(
                subset_name, c3[subset_mask], c4[subset_mask]
            )
            row = summary.as_row()
            row["class"] = label
            per_class_rows.append(row)
    lateralization_by_class = pd.DataFrame(per_class_rows)[
        ["class", "subset", "n_epochs", "c3_mu_power", "c4_mu_power", "lateralization_index"]
    ]

    # Does artifact status move the lateralisation measure by more than chance?
    per_epoch_li = per_epoch_lateralization(c3, c4)
    li_difference, li_p = permutation_difference(
        per_epoch_li, flagged, n_permutations=args.permutations, seed=config.seed
    )
    # The same comparison inside REST, the only class with epochs on both sides of
    # the flag.  Without this the flagged/unflagged split is confounded with class:
    # the unflagged set contains no LEFT_HAND_IMAGERY epochs at all.
    rest_mask = (epoch_qc["class"] == "REST").to_numpy()
    rest_li_difference, rest_li_p = permutation_difference(
        per_epoch_li[rest_mask],
        flagged[rest_mask],
        n_permutations=args.permutations,
        seed=config.seed,
    )

    # Is the flag tracking noise, or is it tracking the sensorimotor signal itself?
    correlations = amplitude_signal_correlation(
        measures.epoch_p2p_max,
        {
            "C3 mu power": c3,
            "C4 mu power": c4,
            "per-epoch lateralization": per_epoch_li,
            "artifact_channel_count": measures.artifact_channel_count.astype(float),
        },
    )

    # The sensorimotor channels the decoder actually depends on, pulled out of the
    # 64-row channel table so the report does not have to state their rates from
    # a rank position.
    sensorimotor = channel_qc[channel_qc["channel"].isin(["C3", "C4", "Cz", "C1", "C2"])]

    # --- report ---------------------------------------------------------------
    n = len(epoch_qc)
    n_flagged = int(flagged.sum())
    out: List[str] = []
    out.append("=" * 78)
    out.append(f"PHASE 1.6 QUALITY CONTROL -- {subject_id} {run_id}")
    out.append("=" * 78)
    out.append(
        f"epochs {n}   channels {len(channels)}   {sfreq:.0f} Hz   "
        f"threshold {measures.threshold_uv:.0f} µV peak-to-peak (inherited, unchanged)"
    )
    out.append(f"flagged {n_flagged}/{n} ({100.0 * n_flagged / n:.1f}%)")
    out.append("")
    out.append("--- 1. epoch-level QC table " + "-" * 50)
    out.append(_fmt_table(epoch_qc))
    out.append("")
    out.append("--- 2. distribution statistics (µV) " + "-" * 42)
    out.append(_fmt_table(distribution.reset_index(names="measure")))
    out.append("")
    out.append("--- 3. channel-level artifact frequency (top 15 of 64) " + "-" * 23)
    out.append(_fmt_table(channel_qc.head(15)))
    out.append(
        f"    channels never flagged: "
        f"{int((channel_qc['flagged_epochs'] == 0).sum())} of {len(channel_qc)}"
    )
    out.append("    sensorimotor channels (what the decoder uses):")
    out.append(_fmt_table(sensorimotor))
    out.append("")
    out.append("    by scalp region -- where the crossings actually are:")
    out.append(_fmt_table(region_qc))
    out.append("")
    out.append("    dominant region per flagged epoch:")
    out.append(_fmt_table(epoch_regions))
    out.append("")
    out.append("--- 4. class-level artifact distribution " + "-" * 37)
    out.append(_fmt_table(class_qc))
    out.append("")
    out.append("--- 5. representative epochs " + "-" * 49)
    for key in ("clean", "median", "high"):
        rep = representatives[key]
        out.append(
            f"  {rep.banner:<36} {rep.epoch_id}  {rep.class_label:<20} "
            f"p-p {rep.peak_to_peak_uv:8.1f} µV  "
            f"{rep.artifact_channel_count:>2}/64 channels over threshold"
        )
        out.append(f"      -> {figure_paths[key]}")
    out.append(f"      -> {figure_paths['distribution']}")
    out.append("")
    out.append("--- 6. mu power and lateralisation, flagged vs unflagged " + "-" * 21)
    out.append(_fmt_table(lateralization, floatfmt="%.6g"))
    out.append("")
    out.append("    per class:")
    out.append(_fmt_table(lateralization_by_class, floatfmt="%.6g"))
    out.append("")
    out.append("    by class (all epochs) -- the scale to judge any shift against:")
    out.append(
        _fmt_table(
            class_lateralization[
                ["class", "n_epochs", "c3_mu_power", "c4_mu_power", "lateralization_index"]
            ],
            floatfmt="%.6g",
        )
    )
    out.append(
        f"      left-vs-right separation (RIGHT - LEFT): {left_right_separation:+.4f}"
    )
    out.append("")
    out.append(
        f"    per-epoch log(C4/C3), flagged minus unflagged:\n"
        f"      all epochs  : {li_difference:+.4f}  "
        f"(permutation p = {li_p:.4f})\n"
        f"      REST only   : {rest_li_difference:+.4f}  "
        f"(permutation p = {rest_li_p:.4f})   <- the only class-matched comparison"
    )
    out.append("")
    out.append("    amplitude vs signal (Spearman, n=23):")
    out.append(_fmt_table(correlations, floatfmt="%.4f"))
    out.append("")
    out.append("--- 7. is the artifact flag associated with class? " + "-" * 27)
    out.append(
        f"    chi-square statistic  : {association.chi2:.4f}\n"
        f"    Cramer's V            : {association.cramers_v:.4f}\n"
        f"    permutation p-value   : {association.p_permutation:.4f} "
        f"({association.n_permutations} label shuffles)\n"
        f"    max class-rate spread : {association.max_rate_spread_pp:.1f} "
        f"percentage points\n"
        f"    reading               : {association.verdict()}"
    )
    out.append("")
    out.append("    contrast by contrast (Fisher exact):")
    out.append(_fmt_table(contrasts, floatfmt="%.4f"))
    out.append("")
    out.append("--- artifacts written " + "-" * 56)
    out.append(f"    {epoch_qc_path}")
    out.append(f"    {channel_qc_path}")
    out.append("")
    out.append(
        "No epoch was dropped, no preprocessing parameter was changed, no threshold\n"
        "was re-tuned and no model was trained.  Phase 1.6 is diagnostic only."
    )
    report = "\n".join(out)
    print(report)

    if args.json:
        payload = {
            "subject_id": subject_id,
            "run_id": run_id,
            "sampling_rate": sfreq,
            "n_epochs": n,
            "n_channels": len(channels),
            "threshold_uv": measures.threshold_uv,
            "n_flagged": n_flagged,
            "pct_flagged": 100.0 * n_flagged / n,
            "distribution": distribution.to_dict(orient="index"),
            "class_qc": class_qc.to_dict(orient="records"),
            "channel_qc": channel_qc.to_dict(orient="records"),
            "region_qc": region_qc.to_dict(orient="records"),
            "epoch_dominant_region": epoch_regions.to_dict(orient="records"),
            "epoch_qc": epoch_qc.to_dict(orient="records"),
            "lateralization": lateralization.to_dict(orient="records"),
            "lateralization_by_class": lateralization_by_class.to_dict(orient="records"),
            "lateralization_by_class_all_epochs": class_lateralization.to_dict(
                orient="records"
            ),
            "left_right_separation": left_right_separation,
            "lateralization_permutation": {
                "all_epochs_difference": li_difference,
                "all_epochs_p": li_p,
                "rest_only_difference": rest_li_difference,
                "rest_only_p": rest_li_p,
            },
            "amplitude_signal_correlation": correlations.to_dict(orient="records"),
            "class_contrasts": contrasts.to_dict(orient="records"),
            "sensorimotor_channels": sensorimotor.to_dict(orient="records"),
            "class_association": {
                "chi2": association.chi2,
                "cramers_v": association.cramers_v,
                "p_permutation": association.p_permutation,
                "n_permutations": association.n_permutations,
                "max_rate_spread_pp": association.max_rate_spread_pp,
                "verdict": association.verdict(),
            },
            "representatives": {
                key: {
                    "epoch_id": rep.epoch_id,
                    "banner": rep.banner,
                    "class": rep.class_label,
                    "peak_to_peak_uv": rep.peak_to_peak_uv,
                    "artifact_channel_count": rep.artifact_channel_count,
                    "figure": str(figure_paths[key]),
                }
                for key, rep in representatives.items()
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
