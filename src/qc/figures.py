"""Phase 1.6 -- representative-epoch figures for visual inspection.

Three figures, one per point on the amplitude distribution: the cleanest epoch,
the median-quality epoch and the most contaminated one.  Each shows all 64
channels, then C3 and C4 on their own, because the sensorimotor pair is what the
decoder will actually use and a montage-wide butterfly plot hides what happens
there.

The banner wording matters.  The highest-amplitude epoch is labelled
**HIGH-AMPLITUDE / POTENTIAL ARTIFACT**, never "artifact": a large excursion in an
8-30 Hz band-passed epoch is consistent with an artifact *and* with a genuine
high-amplitude neural or muscular event, and nothing measured in Phase 1.6
separates the two.  Calling it an artifact would assert a conclusion the data does
not support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: figures are written, never shown
import matplotlib.pyplot as plt
import numpy as np

from src.qc.quality import V_TO_UV, AmplitudeMeasures, Representative

#: Colour used for anything that marks the inherited 150 µV threshold.
THRESHOLD_COLOR = "#c0392b"

#: Banner background per representative, from calm to alarming.
BANNER_COLORS: Dict[str, str] = {
    "clean": "#1a7f5a",
    "median": "#8a6d1f",
    "high": THRESHOLD_COLOR,
}


def _channel_index(channels: Sequence[str], name: str) -> int:
    try:
        return list(channels).index(name)
    except ValueError as exc:  # pragma: no cover - montage is fixed at 64 channels
        raise ValueError(f"channel {name!r} not in montage {list(channels)}") from exc


def _plot_single_channel(
    ax: plt.Axes,
    times: np.ndarray,
    trace_uv: np.ndarray,
    label: str,
    threshold_uv: float,
) -> None:
    """One channel, with its own peak-to-peak annotated."""
    p2p = float(trace_uv.max() - trace_uv.min())
    ax.plot(times, trace_uv, linewidth=0.9, color="#12395b")
    ax.axhline(0.0, linewidth=0.6, color="#999999", zorder=0)

    # Mark the excursion itself rather than a symmetric +/- band: the threshold is
    # on peak-to-peak, so a horizontal pair of lines at +/-75 uV would imply a
    # criterion that is not the one being applied.  The marker sits outside the
    # data area (clip_on=False) so it never covers the trace.
    x_marker = times[-1] + 0.012 * (times[-1] - times[0])
    ax.annotate(
        "",
        xy=(x_marker, trace_uv.max()),
        xytext=(x_marker, trace_uv.min()),
        arrowprops=dict(
            arrowstyle="<->", color=THRESHOLD_COLOR, linewidth=1.1, clip_on=False
        ),
        annotation_clip=False,
    )
    over = p2p > threshold_uv
    ax.text(
        0.995,
        0.94,
        f"peak-to-peak {p2p:.0f} µV — "
        + ("above" if over else "below")
        + f" the {threshold_uv:.0f} µV flag",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=THRESHOLD_COLOR if over else "#3d6b4f",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85),
    )

    ax.set_ylabel(f"{label}\n(µV)", fontsize=10)
    ax.set_xlim(times[0], times[-1])
    ax.margins(y=0.15)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_representative_epoch(
    X: np.ndarray,
    times: np.ndarray,
    channels: Sequence[str],
    measures: AmplitudeMeasures,
    rep: Representative,
    out_dir: Path,
    filename: str,
    *,
    subject: str,
    run: str,
) -> Path:
    """Render one representative epoch: all channels, then C3, then C4.

    ``X`` is the full ``(n_epochs, n_channels, n_times)`` volt array; the epoch is
    selected by ``rep.position`` so the figure can never drift out of step with
    the QC table.
    """
    epoch_uv = X[rep.position] * V_TO_UV
    p2p_per_channel = measures.per_channel_p2p[rep.position]
    threshold_uv = measures.threshold_uv

    fig = plt.figure(figsize=(12.5, 9.0))
    grid = fig.add_gridspec(
        3, 1, height_ratios=[2.4, 1.0, 1.0], hspace=0.32, top=0.86, bottom=0.07
    )

    # --- all channels ---------------------------------------------------------
    ax_all = fig.add_subplot(grid[0])
    over_mask = p2p_per_channel > threshold_uv
    for i in range(epoch_uv.shape[0]):
        ax_all.plot(
            times,
            epoch_uv[i],
            linewidth=0.55 if over_mask[i] else 0.4,
            color=THRESHOLD_COLOR if over_mask[i] else "#8899aa",
            alpha=0.85 if over_mask[i] else 0.5,
            zorder=3 if over_mask[i] else 2,
        )
    ax_all.set_xlim(times[0], times[-1])
    ax_all.set_ylabel("all 64 channels (µV)", fontsize=10)
    ax_all.set_title(
        f"{rep.artifact_channel_count} of {len(channels)} channels exceed "
        f"{threshold_uv:.0f} µV peak-to-peak (shown in red)",
        loc="left",
        fontsize=10,
        color="#333333",
    )
    for side in ("top", "right"):
        ax_all.spines[side].set_visible(False)

    # --- C3 and C4 ------------------------------------------------------------
    for row, name in ((1, "C3"), (2, "C4")):
        ax = fig.add_subplot(grid[row])
        _plot_single_channel(
            ax, times, epoch_uv[_channel_index(channels, name)], name, threshold_uv
        )
        if row == 2:
            ax.set_xlabel("time from cue onset (s)", fontsize=10)

    # --- banner ---------------------------------------------------------------
    fig.text(
        0.5,
        0.955,
        rep.banner,
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#ffffff",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor=BANNER_COLORS[rep.key],
            edgecolor="none",
        ),
    )
    fig.text(
        0.5,
        0.895,
        f"{subject} {run}  ·  {rep.epoch_id}  ·  {rep.class_label}  ·  "
        f"epoch peak-to-peak {rep.peak_to_peak_uv:.1f} µV  ·  "
        f"flag={'YES' if rep.artifact_flag else 'no'} "
        f"(inherited {threshold_uv:.0f} µV threshold, diagnostic only)",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


#: Output filename per representative.  Numbering continues from the Phase 1
#: figures, which end at 07.
REPRESENTATIVE_FILENAMES: Dict[str, str] = {
    "clean": "08_qc_epoch_clean.png",
    "median": "09_qc_epoch_median.png",
    "high": "10_qc_epoch_high_amplitude.png",
}


def plot_amplitude_distribution(
    measures: AmplitudeMeasures,
    epoch_qc,
    out_dir: Path,
    filename: str = "11_qc_amplitude_distribution.png",
) -> Path:
    """Where the 23 epochs sit relative to the inherited threshold.

    Not required by the Phase 1.6 brief, but the whole question is "is 150 µV in
    the right place for this data", and that is a question about a distribution.
    A table of percentiles answers it less directly than one picture.
    """
    p2p = measures.epoch_p2p_max
    threshold = measures.threshold_uv

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    axes[0].hist(p2p, bins=12, color="#4a6fa5", edgecolor="white")
    axes[0].axvline(
        threshold,
        color=THRESHOLD_COLOR,
        linewidth=1.8,
        label=f"inherited flag: {threshold:.0f} µV",
    )
    axes[0].set_xlabel("epoch peak-to-peak (µV)")
    axes[0].set_ylabel("epochs")
    axes[0].set_title("Amplitude distribution", loc="left", fontsize=11)
    axes[0].legend(fontsize=9)

    classes = sorted(epoch_qc["class"].unique())
    palette = {"LEFT_HAND_IMAGERY": "#c05621", "RIGHT_HAND_IMAGERY": "#2b6cb0", "REST": "#4a5568"}
    for label in classes:
        subset = epoch_qc[epoch_qc["class"] == label]
        axes[1].scatter(
            subset["onset"],
            subset["peak_to_peak_max"],
            s=46,
            label=label,
            color=palette.get(label, "#666666"),
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    axes[1].axhline(threshold, color=THRESHOLD_COLOR, linewidth=1.8, zorder=2)
    axes[1].set_xlabel("cue onset within the run (s)")
    axes[1].set_ylabel("epoch peak-to-peak (µV)")
    axes[1].set_title("Amplitude over the course of the run", loc="left", fontsize=11)
    axes[1].legend(fontsize=8, frameon=False)

    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path
