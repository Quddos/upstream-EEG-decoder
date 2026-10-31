"""Phase 5 -- per-subject signal audit: statistics, figures and interpretation.

    python -m src.cli.phase5_audit

Computes the pre-registered diagnostics, compares the below-chance and
above-chance groups defined **before** any diagnostic existed, and reports the
evidence bearing on two interpretations:

A. **signal absence** -- the failing subjects lack usable sensorimotor
   lateralisation;
B. **pipeline loss** -- they have measurable, consistent signal that the
   representation and decoder fail to exploit.

No classifier is trained anywhere in this module. The only model output used is the
frozen Phase 4B LOSOCV balanced accuracy, read from disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.cli._common import setup_logging
from src.config import DatasetConfig
from src.data.physionet_spec import SENSORIMOTOR_ROI, roi_indices
from src.features.diagnostics import EXPLORATORY, PREREGISTERED, subject_diagnostics
from src.io.store import load_dataset

BELOW = "below_chance"
ABOVE = "at_or_above_chance"
GROUP_COLORS = {BELOW: "#c0392b", ABOVE: "#2b6cb0"}


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def compare_groups(frame: pd.DataFrame, column: str) -> Dict[str, float]:
    """Below-chance vs above-chance on one diagnostic.

    Mann-Whitney rather than a t-test: eight subjects in the smaller group, and
    several diagnostics (Fisher ratios, SNR) are strongly right-skewed by
    construction. Cliff's delta accompanies it because a p-value at n=8 vs 32 says
    little about how separated the groups are.
    """
    a = frame.loc[frame["group"] == BELOW, column].to_numpy(float)
    b = frame.loc[frame["group"] == ABOVE, column].to_numpy(float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return {"n_below": len(a), "n_above": len(b)}
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Cliff's delta: P(above > below) - P(above < below).
    greater = sum((y > x) for x in a for y in b)
    less = sum((y < x) for x in a for y in b)
    delta = (greater - less) / (len(a) * len(b))
    return {
        "n_below": int(len(a)), "n_above": int(len(b)),
        "median_below": float(np.median(a)), "median_above": float(np.median(b)),
        "mean_below": float(a.mean()), "mean_above": float(b.mean()),
        "mannwhitney_u": float(u.statistic), "p": float(u.pvalue),
        "cliffs_delta": float(delta),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5 per-subject signal audit")
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    config = DatasetConfig()
    phase_dir = args.dir or (config.paths.root / "data" / "metadata" / "phase5")
    out_dir = args.out_dir or (config.paths.root / "reports" / "figures")
    phase_dir.mkdir(parents=True, exist_ok=True)

    # --- outcome labels, locked before any diagnostic --------------------------
    labels_path = phase_dir / "phase5_outcome_labels.csv"
    outcome = pd.read_csv(labels_path)
    definition = json.loads((phase_dir / "phase5_outcome_definition.json").read_text())

    # --- diagnostics ----------------------------------------------------------
    diagnostics_path = phase_dir / "phase5_subject_diagnostics.csv"
    if diagnostics_path.exists():
        diagnostics = pd.read_csv(diagnostics_path)
    else:
        dataset = load_dataset(config, normalize=False)
        sfreq = float(dataset.metadata["sampling_rate"].iloc[0])
        picks = list(roi_indices(dataset.channels, SENSORIMOTOR_ROI))
        diagnostics = subject_diagnostics(
            dataset.X[:, picks, :].astype(np.float64),
            dataset.metadata,
            [dataset.channels[i] for i in picks],
            sfreq,
        )
        diagnostics.to_csv(diagnostics_path, index=False)

    frame = diagnostics.merge(outcome, on="subject_id", how="inner")
    frame.to_csv(phase_dir / "phase5_subject_audit.csv", index=False)

    out: List[str] = ["=" * 78, "PHASE 5 -- PER-SUBJECT SIGNAL AUDIT", "=" * 78, ""]
    out.append(f"subjects            : {len(frame)}")
    out.append(f"below chance (<0.50): {int((frame['group'] == BELOW).sum())}  "
               f"{definition['below_subjects']}")
    out.append(f"at/above chance     : {int((frame['group'] == ABOVE).sum())}")
    out.append(f"outcome source      : {definition['source']}")
    out.append("  (locked before any diagnostic was computed)")
    out.append("")

    payload: Dict[str, object] = {
        "outcome_definition": definition,
        "n_subjects": int(len(frame)),
    }

    # --- pre-registered comparisons -------------------------------------------
    out.append("--- pre-registered diagnostics: below vs at/above chance " + "-" * 21)
    comparisons: Dict[str, Dict[str, float]] = {}
    for family, columns in PREREGISTERED.items():
        out.append(f"  [{family}]")
        for column in columns:
            if column not in frame.columns:
                continue
            result = compare_groups(frame, column)
            comparisons[column] = result
            if "p" not in result:
                continue
            out.append(
                f"    {column:32s} below={result['median_below']:+8.3f} "
                f"above={result['median_above']:+8.3f} "
                f"delta={result['cliffs_delta']:+.3f} p={result['p']:.4f}"
            )
        out.append("")
    payload["preregistered_comparisons"] = comparisons

    # --- correlations with LOSOCV accuracy ------------------------------------
    out.append("--- correlation with LOSOCV balanced accuracy (exploratory) " + "-" * 18)
    correlations: Dict[str, Dict[str, float]] = {}
    accuracy = frame["losocv_balanced_accuracy"].to_numpy(float)
    for family, columns in PREREGISTERED.items():
        for column in columns:
            if column not in frame.columns:
                continue
            values = frame[column].to_numpy(float)
            mask = np.isfinite(values) & np.isfinite(accuracy)
            if mask.sum() < 5:
                continue
            r, rp = stats.pearsonr(values[mask], accuracy[mask])
            rho, sp = stats.spearmanr(values[mask], accuracy[mask])
            correlations[column] = {
                "pearson_r": float(r), "pearson_p": float(rp),
                "spearman_rho": float(rho), "spearman_p": float(sp),
                "n": int(mask.sum()),
            }
            out.append(f"  {column:32s} r={r:+.3f} (p={rp:.4f})  rho={rho:+.3f} (p={sp:.4f})")
    payload["correlations"] = correlations
    out.append("")
    out.append("  NOTE: these are associations across 40 subjects, not causal evidence,")
    out.append("  and they were computed after the group comparison above.")
    out.append("")

    # --- exploratory ----------------------------------------------------------
    out.append("--- exploratory diagnostics (labelled, not used to support a claim) " + "-" * 9)
    exploratory: Dict[str, Dict[str, float]] = {}
    for column in EXPLORATORY:
        if column not in frame.columns:
            continue
        result = compare_groups(frame, column)
        exploratory[column] = result
        if "p" in result:
            out.append(f"  {column:32s} below={result['median_below']:+8.3f} "
                       f"above={result['median_above']:+8.3f} "
                       f"delta={result['cliffs_delta']:+.3f} p={result['p']:.4f}")
    payload["exploratory_comparisons"] = exploratory
    out.append("")

    # --- the A vs B discriminating evidence -----------------------------------
    # Signal absence predicts the failing subjects lack lateralisation; pipeline
    # loss predicts they have it and it is not being used. The two are separated
    # by whether lateralisation-family diagnostics differ while mu-rhythm presence
    # does not.
    out.append("--- evidence bearing on A (signal absence) vs B (pipeline loss) " + "-" * 14)
    lateralisation_keys = ("lat_separation", "lat_d", "fisher_lateralisation",
                           "lat_epoch_agreement")
    presence_keys = ("mu_snr_mean", "mu_snr_C3", "mu_snr_C4",
                     "task_vs_rest_mu_change_mean")
    verdict: Dict[str, object] = {}
    for name, keys in (("lateralisation_family", lateralisation_keys),
                       ("signal_presence_family", presence_keys)):
        significant = [k for k in keys if comparisons.get(k, {}).get("p", 1.0) < 0.05]
        deltas = [abs(comparisons[k]["cliffs_delta"]) for k in keys if k in comparisons
                  and "cliffs_delta" in comparisons[k]]
        verdict[name] = {
            "n_diagnostics": len(keys),
            "n_significant_p05": len(significant),
            "significant": significant,
            "mean_abs_cliffs_delta": float(np.mean(deltas)) if deltas else float("nan"),
        }
        out.append(f"  {name:24s} {len(significant)}/{len(keys)} significant at p<0.05, "
                   f"mean |delta| = {verdict[name]['mean_abs_cliffs_delta']:.3f}")
    payload["family_verdict"] = verdict

    # How many below-chance subjects show a usable lateralised pattern at all?
    below = frame[frame["group"] == BELOW]
    above = frame[frame["group"] == ABOVE]
    usable_below = int(((below["lat_d"].abs() > 0.5) & (below["lat_separation"] > 0)).sum())
    usable_above = int(((above["lat_d"].abs() > 0.5) & (above["lat_separation"] > 0)).sum())
    payload["usable_lateralisation"] = {
        "criterion": "|lat_d| > 0.5 AND separation in the expected direction",
        "below_chance": {"n": usable_below, "of": int(len(below))},
        "at_or_above_chance": {"n": usable_above, "of": int(len(above))},
    }
    out.append(f"  subjects with a usable lateralised pattern (|d|>0.5, correct sign):")
    out.append(f"    below chance : {usable_below}/{len(below)}")
    out.append(f"    at/above     : {usable_above}/{len(above)}")
    out.append("")

    # --- figures ---------------------------------------------------------------
    written: List[Path] = []
    written.append(_plot_distributions(frame, out_dir))
    written.append(_plot_scatter(frame, out_dir))
    written.append(_plot_per_subject(frame, out_dir))
    written.append(_plot_lateralisation_map(frame, out_dir))
    written.append(_plot_effect_sizes(comparisons, out_dir))

    out.append("--- figures " + "-" * 65)
    for path in written:
        out.append(f"  {path.name}")

    print("\n".join(out))
    (phase_dir / "phase5_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nsummary : {phase_dir / 'phase5_summary.json'}")
    return 0


def _plot_distributions(frame: pd.DataFrame, out_dir: Path) -> Path:
    """53 -- each pre-registered diagnostic, by group."""
    columns = [c for fam in PREREGISTERED.values() for c in fam if c in frame.columns]
    ncols = 4
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, column in zip(axes, columns):
        data, names, colours = [], [], []
        for group in (BELOW, ABOVE):
            values = frame.loc[frame["group"] == group, column].to_numpy(float)
            data.append(values[np.isfinite(values)])
            names.append("below" if group == BELOW else "at/above")
            colours.append(GROUP_COLORS[group])
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops=dict(color="#111111"))
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.5)
        for i, values in enumerate(data, start=1):
            jitter = (np.random.default_rng(i).random(len(values)) - 0.5) * 0.22
            ax.scatter(np.full(len(values), i) + jitter, values, s=20,
                       color="#333333", alpha=0.65, zorder=3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(names, fontsize=9)
        ax.set_title(column, loc="left", fontsize=9)
        _despine(ax)

    for ax in axes[len(columns):]:
        ax.set_visible(False)
    fig.suptitle("Figure 53 — pre-registered diagnostics by outcome group",
                 fontsize=13, y=1.0, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "53_diagnostics_by_group")


def _plot_scatter(frame: pd.DataFrame, out_dir: Path) -> Path:
    """54 -- diagnostics against LOSOCV accuracy."""
    columns = ["mu_snr_mean", "task_vs_rest_mu_change_mean", "lat_d",
               "lat_separation", "lat_epoch_agreement", "fisher_lateralisation"]
    columns = [c for c in columns if c in frame.columns]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for ax, column in zip(axes, columns):
        for group in (ABOVE, BELOW):
            sub = frame[frame["group"] == group]
            ax.scatter(sub[column], sub["losocv_balanced_accuracy"], s=60,
                       color=GROUP_COLORS[group], edgecolor="white", linewidth=0.7,
                       label="below chance" if group == BELOW else "at/above chance",
                       zorder=3)
        values = frame[column].to_numpy(float)
        accuracy = frame["losocv_balanced_accuracy"].to_numpy(float)
        mask = np.isfinite(values)
        if mask.sum() > 2:
            r, p = stats.pearsonr(values[mask], accuracy[mask])
            slope, intercept = np.polyfit(values[mask], accuracy[mask], 1)
            xs = np.linspace(values[mask].min(), values[mask].max(), 40)
            ax.plot(xs, slope * xs + intercept, color="#c05621", linewidth=1.8)
            ax.set_title(f"{column}\nr = {r:+.3f} (p = {p:.4f})", loc="left", fontsize=10)
        ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1.4)
        ax.set_ylabel("LOSOCV balanced accuracy")
        ax.set_xlabel(column, fontsize=9)
        _despine(ax)
    axes[0].legend(fontsize=8, frameon=False)

    for ax in axes[len(columns):]:
        ax.set_visible(False)
    fig.suptitle("Figure 54 — diagnostics vs cross-subject performance (n=40)",
                 fontsize=13, y=1.0, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "54_diagnostics_vs_losocv")


def _plot_per_subject(frame: pd.DataFrame, out_dir: Path) -> Path:
    """55 -- every subject: accuracy beside the two decisive diagnostics."""
    ordered = frame.sort_values("losocv_balanced_accuracy")
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    x = np.arange(len(ordered))
    colours = [GROUP_COLORS[g] for g in ordered["group"]]

    axes[0].bar(x, ordered["losocv_balanced_accuracy"], color=colours, alpha=0.9)
    axes[0].axhline(0.5, color="#333333", linestyle="--", linewidth=1.5)
    axes[0].set_ylabel("LOSOCV\nbalanced accuracy")
    axes[0].set_title("(a) outcome — red is the below-chance group", loc="left", fontsize=11)

    axes[1].bar(x, ordered["lat_d"], color=colours, alpha=0.9)
    axes[1].axhline(0.0, color="#333333", linewidth=1.2)
    axes[1].axhline(0.5, color="#2f855a", linestyle=":", linewidth=1.5)
    axes[1].set_ylabel("lateralisation\nCohen's d")
    axes[1].set_title("(b) lateralisation effect size (dotted line: d = 0.5)",
                      loc="left", fontsize=11)

    axes[2].bar(x, ordered["mu_snr_mean"], color=colours, alpha=0.9)
    axes[2].set_ylabel("mu-band SNR\n(mu / 20-30 Hz)")
    axes[2].set_title("(c) mu rhythm presence — note it does NOT track the outcome",
                      loc="left", fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(ordered["subject_id"], rotation=60, ha="right", fontsize=7)

    for ax in axes:
        _despine(ax)
    fig.suptitle("Figure 55 — per subject: outcome, lateralisation, mu presence",
                 fontsize=13, y=1.0, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "55_per_subject_diagnostics")


def _plot_lateralisation_map(frame: pd.DataFrame, out_dir: Path) -> Path:
    """56 -- magnitude against consistency, the distinction the brief asks for."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    for group in (ABOVE, BELOW):
        sub = frame[frame["group"] == group]
        axes[0].scatter(sub["lat_separation"], sub["lat_epoch_agreement"], s=70,
                        color=GROUP_COLORS[group], edgecolor="white", linewidth=0.7,
                        label="below chance" if group == BELOW else "at/above chance",
                        zorder=3)
    for _, row in frame.iterrows():
        axes[0].annotate(row["subject_id"], (row["lat_separation"], row["lat_epoch_agreement"]),
                         textcoords="offset points", xytext=(6, 3), fontsize=6.5,
                         color="#555555")
    axes[0].axvline(0.0, color="#333333", linewidth=1.2)
    axes[0].axhline(0.5, color="#333333", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("magnitude: lateralisation separation  log(C4/C3), RIGHT − LEFT")
    axes[0].set_ylabel("consistency: epoch-level agreement")
    axes[0].set_title("(a) magnitude vs consistency", loc="left", fontsize=11)
    axes[0].legend(fontsize=9, frameon=False)

    # Run-level sign agreement, which separates "weak but stable" from "unstable".
    counts = (
        frame.groupby(["group", "lat_sign_consistency"]).size().unstack(fill_value=0)
    )
    bottom = np.zeros(len(counts.columns))
    for group in counts.index:
        axes[1].bar(np.arange(len(counts.columns)) + (0.0 if group == BELOW else 0.35),
                    counts.loc[group].to_numpy(), 0.33,
                    color=GROUP_COLORS[group], alpha=0.9,
                    label="below chance" if group == BELOW else "at/above chance")
    axes[1].set_xticks(np.arange(len(counts.columns)) + 0.17)
    axes[1].set_xticklabels([f"{c:.2f}" for c in counts.columns])
    axes[1].set_xlabel("run-level sign consistency (fraction of runs agreeing)")
    axes[1].set_ylabel("subjects")
    axes[1].set_title("(b) is the lateralisation sign stable across runs?",
                      loc="left", fontsize=11)
    axes[1].legend(fontsize=9, frameon=False)

    for ax in axes:
        _despine(ax)
    fig.suptitle("Figure 56 — lateralisation: magnitude distinguished from consistency",
                 fontsize=13, y=1.02, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "56_lateralisation_magnitude_vs_consistency")


def _plot_effect_sizes(comparisons: Dict[str, Dict[str, float]], out_dir: Path) -> Path:
    """57 -- which diagnostics actually separate the groups."""
    items = [(k, v) for k, v in comparisons.items() if "cliffs_delta" in v]
    items.sort(key=lambda kv: abs(kv[1]["cliffs_delta"]))
    names = [k for k, _ in items]
    deltas = [v["cliffs_delta"] for _, v in items]
    ps = [v["p"] for _, v in items]
    colours = ["#2f855a" if p < 0.05 else "#a0aec0" for p in ps]

    fig, ax = plt.subplots(figsize=(11, 0.45 * len(names) + 2.4))
    ax.barh(range(len(names)), deltas, color=colours, alpha=0.9)
    for i, (d, p) in enumerate(zip(deltas, ps)):
        ax.text(d + (0.02 if d >= 0 else -0.02), i, f"p={p:.3f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=8)
    ax.axvline(0.0, color="#333333", linewidth=1.2)
    for threshold in (-0.474, 0.474):
        ax.axvline(threshold, color="#c05621", linestyle=":", linewidth=1.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Cliff's delta  (at/above chance − below chance)")
    ax.set_xlim(-1.15, 1.15)
    ax.set_title(
        "Figure 57 — group separation per diagnostic\n"
        "green = p < 0.05; dotted lines mark |delta| = 0.474, the conventional "
        "'large effect' boundary",
        loc="left", fontsize=11)
    _despine(ax)
    fig.tight_layout()
    return _save(fig, out_dir, "57_diagnostic_effect_sizes")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
