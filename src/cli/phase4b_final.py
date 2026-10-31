"""Phase 4B-final -- all-64 analysis, ROI-21 comparison, figures 47-52, decision.

    python -m src.cli.phase4b_final

Reads the all-64 results produced by :mod:`src.cli.phase4b_all64` and the frozen
ROI-21 results from Phase 4B, and answers the one question Phase 4B left open: does
giving CSP and Riemannian the full montage change Decision C?

The ROI-21 artifacts are opened read-only and never rewritten. All-64 figures use a
new number range (47-52) so nothing from Phases 1-4B is overwritten.

The pre-registered ROI-21 result remains primary. All-64 is a secondary condition,
and the report says so wherever a number from it is quoted.
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

PRIMARY_TASK = "left_vs_right"
PRIMARY_CLASSIFIER = "logistic_regression"
REPS = ("frozen", "csp", "riemann")
LABELS = {"frozen": "frozen features", "csp": "CSP", "riemann": "Riemannian TS"}
COLORS = {"frozen": "#4a6fa5", "csp": "#b7791f", "riemann": "#2f855a"}
SET_COLORS = {"roi21": "#4299e1", "all64": "#2c5282"}
CHANCE_COLOR = "#c0392b"


def chance(task: str) -> float:
    return 0.5 if task == "left_vs_right" else 1.0 / 3.0


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def scores(frame: pd.DataFrame, task: str, classifier: str, rep: str,
           protocol: str = "losocv", metric: str = "balanced_accuracy") -> pd.Series:
    rows = frame[
        (frame["protocol"] == protocol) & (frame["task"] == task)
        & (frame["classifier"] == classifier) & (frame["representation"] == rep)
    ]
    if rows.empty:
        return pd.Series(dtype=float)
    if protocol == "within_subject":
        rows = rows.assign(s=rows["held_out_subject"].str.split("/").str[0])
        return rows.groupby("s")[metric].mean().sort_index()
    return rows.set_index("held_out_subject")[metric].sort_index()


def one_sample(values: np.ndarray, null: float) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    half = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return {
        "n": int(len(values)), "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
        "ci_low": float(values.mean() - half), "ci_high": float(values.mean() + half),
        "min": float(values.min()), "max": float(values.max()),
        "above_chance": int((values > null).sum()), "below_chance": int((values < null).sum()),
        "pct_below_chance": float(100.0 * (values < null).mean()),
        "t_p": float(stats.ttest_1samp(values, null).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(values - null).pvalue),
    }


def paired(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(b, float) - np.asarray(a, float)
    half = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
    return {
        "n": int(len(diff)), "mean_difference": float(diff.mean()),
        "median_difference": float(np.median(diff)),
        "ci_low": float(diff.mean() - half), "ci_high": float(diff.mean() + half),
        "improved": int((diff > 0).sum()), "worsened": int((diff < 0).sum()),
        "unchanged": int((diff == 0).sum()),
        "t_p": float(stats.ttest_rel(b, a).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(b, a).pvalue),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4B-final all-64 analysis")
    parser.add_argument("--all64-dir", type=Path, default=None)
    parser.add_argument("--roi21-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    config = DatasetConfig()
    root = config.paths.root
    all64_dir = args.all64_dir or (root / "data" / "metadata" / "phase4b_all64")
    roi21_dir = args.roi21_dir or (root / "data" / "metadata" / "phase4b")
    out_dir = args.out_dir or (root / "reports" / "figures")

    a64 = pd.read_csv(all64_dir / "phase4b_all64_results.csv")
    roi = pd.read_csv(roi21_dir / "phase4b_results.csv")
    roi = roi[roi["channel_set"] == "roi21"]

    payload: Dict[str, object] = {
        "primary_condition": "roi21 (pre-registered, Phase 4B; unchanged)",
        "secondary_condition": "all64 (this run)",
    }
    out: List[str] = ["=" * 78, "PHASE 4B-FINAL -- ALL-64-CHANNEL SECONDARY COMPARISON",
                      "=" * 78, ""]

    tasks = [t for t in ("left_vs_right", "left_vs_right_vs_rest")
             if t in set(a64["task"])]
    protocols = sorted(set(a64["protocol"]))
    out.append(f"all-64 rows: {len(a64)}   tasks: {tasks}   protocols: {protocols}")
    if "error" in a64.columns and a64["error"].notna().any():
        out.append(f"  errors: {int(a64['error'].notna().sum())} fold(s)")
    out.append("")

    # --- endpoints -----------------------------------------------------------
    endpoints: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        null = chance(task)
        out.append(f"--- {task} (chance {null:.4f}) " + "-" * 42)
        for classifier in ("logistic_regression", "linear_svm"):
            for rep in REPS:
                v = scores(a64, task, classifier, rep)
                if v.empty or len(v) < 3:
                    continue
                summary = one_sample(v.to_numpy(), null)
                rows = a64[(a64["protocol"] == "losocv") & (a64["task"] == task)
                           & (a64["classifier"] == classifier)
                           & (a64["representation"] == rep)]
                summary["macro_f1"] = float(rows["macro_f1"].mean())
                for column in rows.columns:
                    if column.startswith("recall_"):
                        summary[column] = float(rows[column].mean())
                endpoints[f"all64/{task}/{classifier}/{rep}"] = summary
                out.append(
                    f"  {classifier:20s} {rep:8s} mean={summary['mean']:.4f} "
                    f"CI=[{summary['ci_low']:.4f},{summary['ci_high']:.4f}] "
                    f"above={summary['above_chance']}/{summary['n']} "
                    f"below={summary['below_chance']} SD={summary['sd']:.3f} "
                    f"t p={summary['t_p']:.2e} F1={summary['macro_f1']:.4f}"
                )
        out.append("")
    payload["all64_endpoints"] = endpoints

    # --- paired vs the frozen baseline --------------------------------------
    out.append("--- paired vs frozen baseline " + "-" * 48)
    comparisons: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        for classifier in ("logistic_regression", "linear_svm"):
            for baseline_name, baseline_frame, baseline_set in (
                ("frozen_all64", a64, "all64"),
                ("frozen_roi21_primary", roi, "roi21"),
            ):
                base = scores(baseline_frame, task, classifier, "frozen")
                if base.empty:
                    continue
                for rep in ("csp", "riemann"):
                    other = scores(a64, task, classifier, rep)
                    subjects = sorted(set(base.index) & set(other.index))
                    if len(subjects) < 3:
                        continue
                    result = paired(base[subjects].to_numpy(), other[subjects].to_numpy())
                    key = f"{task}/{classifier}/all64_{rep}_vs_{baseline_name}"
                    comparisons[key] = result
                    out.append(
                        f"  {task:20s} {classifier:20s} all64 {rep:8s} vs {baseline_name:20s} "
                        f"mean={result['mean_difference']:+.4f} "
                        f"CI=[{result['ci_low']:+.4f},{result['ci_high']:+.4f}] "
                        f"+{result['improved']}/-{result['worsened']} "
                        f"t p={result['t_p']:.4f} W p={result['wilcoxon_p']:.4f}"
                    )
    payload["paired_comparisons"] = comparisons
    out.append("")

    # --- ROI-21 vs all-64, same representation ------------------------------
    out.append("--- montage effect: all64 vs roi21, same representation " + "-" * 22)
    montage: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        for rep in REPS:
            a = scores(roi, task, PRIMARY_CLASSIFIER, rep)
            b = scores(a64, task, PRIMARY_CLASSIFIER, rep)
            subjects = sorted(set(a.index) & set(b.index))
            if len(subjects) < 3:
                continue
            result = paired(a[subjects].to_numpy(), b[subjects].to_numpy())
            montage[f"{task}/{rep}"] = result
            out.append(
                f"  {task:22s} {rep:8s} roi21={a[subjects].mean():.4f} "
                f"all64={b[subjects].mean():.4f} diff={result['mean_difference']:+.4f} "
                f"+{result['improved']}/-{result['worsened']} "
                f"t p={result['t_p']:.4f} W p={result['wilcoxon_p']:.4f}"
            )
    payload["montage_effect"] = montage
    out.append("")

    # --- heterogeneity -------------------------------------------------------
    out.append("--- heterogeneity (binary, logistic regression) " + "-" * 30)
    heterogeneity: Dict[str, Dict[str, float]] = {}
    for label, frame in (("roi21", roi), ("all64", a64)):
        for rep in REPS:
            v = scores(frame, PRIMARY_TASK, PRIMARY_CLASSIFIER, rep)
            if v.empty or len(v) < 3:
                continue
            summary = one_sample(v.to_numpy(), 0.5)
            heterogeneity[f"{label}/{rep}"] = summary
            out.append(
                f"  {label:6s} {rep:8s} min={summary['min']:.3f} max={summary['max']:.3f} "
                f"mean={summary['mean']:.3f} SD={summary['sd']:.3f} "
                f"below={summary['below_chance']}/{summary['n']} "
                f"({summary['pct_below_chance']:.0f}%)"
            )
    payload["heterogeneity"] = heterogeneity
    out.append("")

    # --- within vs across ----------------------------------------------------
    out.append("--- within vs across (all-64) " + "-" * 48)
    within_across: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        for rep in REPS:
            across = scores(a64, task, PRIMARY_CLASSIFIER, rep)
            inside = scores(a64, task, PRIMARY_CLASSIFIER, rep, protocol="within_subject")
            if inside.empty or across.empty:
                continue
            subjects = sorted(set(across.index) & set(inside.index))
            if len(subjects) < 3:
                continue
            A = across[subjects].to_numpy(); W = inside[subjects].to_numpy()
            within_across[f"{task}/{rep}"] = {
                "across_mean": float(A.mean()), "within_mean": float(W.mean()),
                "gap": float(W.mean() - A.mean()), "n": len(subjects),
                "within_higher": int((W > A).sum()),
                "t_p": float(stats.ttest_rel(W, A).pvalue),
            }
            r = within_across[f"{task}/{rep}"]
            out.append(f"  {task:22s} {rep:8s} across={r['across_mean']:.4f} "
                       f"within={r['within_mean']:.4f} gap={r['gap']:+.4f} p={r['t_p']:.4f}")
    payload["within_vs_across_all64"] = within_across
    out.append("")

    # --- figures -------------------------------------------------------------
    written: List[Path] = []
    binary = {r: scores(a64, PRIMARY_TASK, PRIMARY_CLASSIFIER, r) for r in REPS}
    binary = {r: v for r, v in binary.items() if not v.empty and len(v) >= 3}
    roi_binary = {r: scores(roi, PRIMARY_TASK, PRIMARY_CLASSIFIER, r) for r in REPS}

    if binary:
        written.append(_plot_means(binary, roi_binary, out_dir))
        if "frozen" in binary:
            written.append(_plot_paired(binary, roi_binary, out_dir))
        written.append(_plot_per_subject(binary, out_dir))
        written.append(_plot_below_chance(binary, roi_binary, out_dir))
    if within_across:
        written.append(_plot_within_across(a64, out_dir, tasks))
    written.append(_plot_summary(roi_binary, binary, out_dir))

    out.append("--- figures " + "-" * 65)
    for path in written:
        out.append(f"  {path.name}")

    print("\n".join(out))
    (all64_dir / "phase4b_all64_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nsummary : {all64_dir / 'phase4b_all64_summary.json'}")
    return 0


def _plot_means(binary, roi_binary, out_dir: Path) -> Path:
    """47 -- all-64 means beside the ROI-21 primary."""
    order = [r for r in REPS if r in binary]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = np.arange(len(order))
    width = 0.38
    for offset, (frame, label, colour) in zip(
        (-width / 2, width / 2),
        ((roi_binary, "ROI-21 (primary)", SET_COLORS["roi21"]),
         (binary, "all-64 (secondary)", SET_COLORS["all64"])),
    ):
        means, cis = [], []
        for rep in order:
            v = frame.get(rep)
            if v is None or len(v) < 3:
                means.append(np.nan); cis.append(0); continue
            arr = v.to_numpy()
            means.append(arr.mean())
            cis.append(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))
        ax.bar(x + offset, means, width, yerr=cis, capsize=5, label=label,
               color=colour, alpha=0.9)
        for xi, m, c in zip(x + offset, means, cis):
            if np.isfinite(m):
                ax.text(xi, m + c + 0.012, f"{m:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color=CHANCE_COLOR, linestyle="--", linewidth=1.8, label="chance")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[r] for r in order])
    ax.set_ylabel("LOSOCV balanced accuracy"); ax.set_ylim(0, 1)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("Figure 47 — all-64 vs ROI-21 (LEFT vs RIGHT, logistic regression)",
                 loc="left", fontsize=11)
    _despine(ax); fig.tight_layout()
    return _save(fig, out_dir, "47_all64_vs_roi21_means")


def _plot_paired(binary, roi_binary, out_dir: Path) -> Path:
    """48 -- paired differences against both frozen baselines."""
    subjects = sorted(binary["frozen"].index)
    pairs = []
    for rep in ("csp", "riemann"):
        if rep in binary:
            pairs.append((f"all64 {LABELS[rep]}\n− all64 frozen",
                          binary[rep][subjects].to_numpy() - binary["frozen"][subjects].to_numpy()))
            if not roi_binary["frozen"].empty:
                common = sorted(set(subjects) & set(roi_binary["frozen"].index))
                pairs.append((f"all64 {LABELS[rep]}\n− ROI-21 frozen (primary)",
                              binary[rep][common].to_numpy() - roi_binary["frozen"][common].to_numpy()))
    fig, ax = plt.subplots(figsize=(11, 6.0))
    data = [d for _, d in pairs]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color="#111111"))
    for patch in bp["boxes"]:
        patch.set_facecolor("#4a6fa5"); patch.set_alpha(0.5)
    for i, d in enumerate(data, start=1):
        jitter = (np.random.default_rng(i).random(len(d)) - 0.5) * 0.2
        ax.scatter(np.full(len(d), i) + jitter, d, s=18, color="#333333", alpha=0.6, zorder=3)
        ax.text(i, max(d) + 0.02, f"mean {d.mean():+.4f}\n+{int((d>0).sum())}/-{int((d<0).sum())}",
                ha="center", fontsize=8)
    ax.axhline(0.0, color="#333333", linewidth=1.3)
    ax.set_xticks(range(1, len(pairs) + 1))
    ax.set_xticklabels([n for n, _ in pairs], fontsize=8)
    ax.set_ylabel("paired difference in balanced accuracy")
    ax.set_title("Figure 48 — paired subject-level differences (all-64)", loc="left", fontsize=11)
    _despine(ax); fig.tight_layout()
    return _save(fig, out_dir, "48_all64_paired_differences")


def _plot_per_subject(binary, out_dir: Path) -> Path:
    """49 -- every subject under each all-64 representation."""
    order = [r for r in REPS if r in binary]
    base = binary[order[0]]
    rank = np.argsort(base.to_numpy())
    subjects = list(base.index)
    fig, ax = plt.subplots(figsize=(15, 6.0))
    x = np.arange(len(rank)); width = 0.8 / len(order)
    for i, rep in enumerate(order):
        ax.bar(x + (i - (len(order) - 1) / 2) * width, binary[rep].to_numpy()[rank], width,
               label=LABELS[rep], color=COLORS[rep], alpha=0.9)
    ax.axhline(0.5, color=CHANCE_COLOR, linestyle="--", linewidth=1.8, label="chance")
    ax.set_xticks(x); ax.set_xticklabels([subjects[i] for i in rank], rotation=60,
                                          ha="right", fontsize=7)
    ax.set_ylabel("LOSOCV balanced accuracy"); ax.set_ylim(0, 1)
    ax.legend(fontsize=9, ncol=4, frameon=False)
    ax.set_title("Figure 49 — per-subject accuracy, all-64 (LEFT vs RIGHT)",
                 loc="left", fontsize=11)
    _despine(ax); fig.tight_layout()
    return _save(fig, out_dir, "49_all64_per_subject")


def _plot_below_chance(binary, roi_binary, out_dir: Path) -> Path:
    """50 -- the deciding endpoint, both montages."""
    order = [r for r in REPS if r in binary]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(order)); width = 0.38
    for offset, (frame, label, colour) in zip(
        (-width / 2, width / 2),
        ((roi_binary, "ROI-21", SET_COLORS["roi21"]),
         (binary, "all-64", SET_COLORS["all64"])),
    ):
        pct, counts = [], []
        for rep in order:
            v = frame.get(rep)
            if v is None or len(v) < 3:
                pct.append(np.nan); counts.append(0); continue
            arr = v.to_numpy()
            pct.append(100.0 * (arr < 0.5).mean()); counts.append(int((arr < 0.5).sum()))
        ax.bar(x + offset, pct, width, label=label, color=colour, alpha=0.9)
        for xi, p, c, rep in zip(x + offset, pct, counts, order):
            if np.isfinite(p):
                n = len(frame[rep])
                ax.text(xi, p + 0.6, f"{c}/{n}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[r] for r in order])
    ax.set_ylabel("% of subjects below chance")
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("Figure 50 — subjects the representation fails, both montages",
                 loc="left", fontsize=11)
    _despine(ax); fig.tight_layout()
    return _save(fig, out_dir, "50_all64_below_chance")


def _plot_within_across(a64: pd.DataFrame, out_dir: Path, tasks: Sequence[str]) -> Path:
    """51 -- within vs across at all-64."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.4 * len(tasks), 5.4))
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        order, a_vals, w_vals = [], [], []
        for rep in REPS:
            across = scores(a64, task, PRIMARY_CLASSIFIER, rep)
            inside = scores(a64, task, PRIMARY_CLASSIFIER, rep, protocol="within_subject")
            if across.empty or inside.empty:
                continue
            order.append(rep); a_vals.append(across.mean()); w_vals.append(inside.mean())
        if not order:
            continue
        x = np.arange(len(order)); width = 0.38
        ax.bar(x - width / 2, a_vals, width, label="across (LOSOCV)", color="#4a6fa5", alpha=0.9)
        ax.bar(x + width / 2, w_vals, width, label="within (leave-one-run-out)",
               color="#68a691", alpha=0.9)
        for i, (a, w) in enumerate(zip(a_vals, w_vals)):
            ax.annotate(f"{w - a:+.3f}", (i, max(a, w) + 0.03), ha="center", fontsize=9,
                        color="#2f855a")
        ax.axhline(chance(task), color=CHANCE_COLOR, linestyle="--", linewidth=1.6)
        ax.set_xticks(x); ax.set_xticklabels([LABELS[r] for r in order], fontsize=9)
        ax.set_ylabel("balanced accuracy"); ax.set_ylim(0, 1)
        ax.set_title(task, loc="left", fontsize=10); ax.legend(fontsize=8, frameon=False)
        _despine(ax)
    fig.suptitle("Figure 51 — within vs across subject, all-64", fontsize=13, y=1.02,
                 x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "51_all64_within_vs_across")


def _plot_summary(roi_binary, binary, out_dir: Path) -> Path:
    """52 -- every representation and montage on one axis."""
    entries = []
    for rep in REPS:
        for label, frame in (("ROI-21", roi_binary), ("all-64", binary)):
            v = frame.get(rep)
            if v is None or len(v) < 3:
                continue
            arr = v.to_numpy()
            entries.append((f"{LABELS[rep]}\n{label}", arr.mean(),
                            1.96 * arr.std(ddof=1) / np.sqrt(len(arr)),
                            SET_COLORS["roi21"] if label == "ROI-21" else SET_COLORS["all64"]))
    fig, ax = plt.subplots(figsize=(12, 6.0))
    x = np.arange(len(entries))
    ax.bar(x, [e[1] for e in entries], 0.62, yerr=[e[2] for e in entries], capsize=5,
           color=[e[3] for e in entries], alpha=0.9)
    for xi, e in zip(x, entries):
        ax.text(xi, e[1] + e[2] + 0.012, f"{e[1]:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color=CHANCE_COLOR, linestyle="--", linewidth=1.8, label="chance")
    ax.set_xticks(x); ax.set_xticklabels([e[0] for e in entries], fontsize=8)
    ax.set_ylabel("LOSOCV balanced accuracy"); ax.set_ylim(0, 1)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("Figure 52 — every representation and montage (LEFT vs RIGHT)",
                 loc="left", fontsize=11)
    _despine(ax); fig.tight_layout()
    return _save(fig, out_dir, "52_all64_summary")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
