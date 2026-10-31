"""Diagnostic visualisations for the Phase 1 dataset.

Seven figures are produced, each answering a specific question about whether the
pipeline did what it claims:

    01_raw_eeg_sample          Does the continuous signal look like EEG at all?
    02_channel_montage         Did channel names resolve to real electrode positions?
    03_event_timeline          Are events alternating rest/task with the right labels?
    04_epoch_left_hand         Does a left-hand imagery epoch look sane?
    05_epoch_right_hand        Does a right-hand imagery epoch look sane?
    06_power_spectral_density  Is the spectrum confined to the passband?
    07_psd_left_vs_right       *The scientific check*: is mu/beta power lateralised
                               contralaterally to the imagined hand?

Figure 07 is the one that matters.  Motor imagery of one hand desynchronises the
sensorimotor rhythm over the **opposite** hemisphere, so left-hand imagery must
show less mu power at C4 than at C3, and right-hand imagery the reverse.  If the
run-dependent T1/T2 mapping were wrong, this asymmetry would invert -- which makes
the figure a direct test of the labelling, not just decoration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never displayed interactively

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch

from src.config import DatasetConfig
from src.data.loader import annotations_table, load_raw
from src.data.physionet_spec import (
    CONTRALATERAL_HAND_CHANNELS,
    DATASET_VERSION,
    LEFT_HAND_IMAGERY,
    REST,
    RIGHT_HAND_IMAGERY,
    RunSpec,
)
from src.io.store import EEGDataset, load_dataset
from src.preprocessing.filters import describe_filter_settings, preprocess_raw

log = logging.getLogger(__name__)

DPI = 140
CLASS_COLORS: Dict[str, str] = {
    LEFT_HAND_IMAGERY: "#2c7fb8",
    RIGHT_HAND_IMAGERY: "#d95f02",
    REST: "#7f7f7f",
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def dataset_label(config: DatasetConfig) -> str:
    """Provenance label prefixed to every figure title.

    Real figures are titled with the dataset's actual identity; synthetic ones say
    so plainly.  A reader must never have to guess which they are looking at.
    """
    if config.source == "synthetic":
        return "Synthetic fixture (not real EEG)"
    return f"PhysioNet EEGMMIDB v{DATASET_VERSION}"


def _synthetic_banner(fig: plt.Figure, config: DatasetConfig) -> None:
    """Stamp figures built from simulated data so they can never be misread."""
    if config.source != "synthetic":
        return
    # Bottom-right, so it never collides with a centred suptitle.
    fig.text(
        0.995,
        0.005,
        "SYNTHETIC FIXTURE DATA - NOT REAL EEG",
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#b30000",
        bbox={"facecolor": "#ffe5e5", "edgecolor": "#b30000", "boxstyle": "round,pad=0.3"},
    )


# --------------------------------------------------------------------------------------
# 1-3: continuous-signal figures
# --------------------------------------------------------------------------------------


def plot_raw_sample(
    spec: RunSpec,
    config: DatasetConfig,
    out_dir: Path,
    duration_s: float = 10.0,
    n_channels: int = 8,
) -> Path:
    """Figure 01 -- raw vs band-passed traces for a handful of channels."""
    raw_unfiltered = load_raw(spec, config)
    raw_filtered = preprocess_raw(raw_unfiltered.copy(), config.preprocessing)

    picks = [ch for ch in ("FC3", "C3", "CP3", "Cz", "FC4", "C4", "CP4", "Pz") if ch in raw_filtered.ch_names]
    picks = picks[:n_channels]
    stop = int(duration_s * raw_filtered.info["sfreq"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, raw, title in (
        (axes[0], raw_unfiltered, "Raw EEG (unfiltered)"),
        (axes[1], raw_filtered, "After preprocessing"),
    ):
        data = raw.copy().pick(picks).get_data(stop=stop) * 1e6  # volts -> microvolts
        times = np.arange(data.shape[1]) / raw.info["sfreq"]
        offset = np.nanmax(np.abs(data)) * 1.6 or 1.0
        for i, ch in enumerate(picks):
            ax.plot(times, data[i] + i * offset, linewidth=0.7, color="#333333")
        ax.set_yticks([i * offset for i in range(len(picks))])
        ax.set_yticklabels(picks)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("channel")
        ax.grid(alpha=0.2, axis="x")

    axes[1].set_xlabel("time (s)")
    subtitle = " | ".join(describe_filter_settings(config.preprocessing)[:3])
    fig.suptitle(
        f"{dataset_label(config)} - {spec.subject_id} {spec.run_id}\n"
        f"first {duration_s:.0f} s | {subtitle}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _synthetic_banner(fig, config)
    return _save(fig, out_dir, "01_raw_eeg_sample")


def plot_montage(spec: RunSpec, config: DatasetConfig, out_dir: Path) -> Path:
    """Figure 02 -- electrode layout, confirming names resolved to real positions."""
    raw = load_raw(spec, config)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    raw.plot_sensors(kind="topomap", show_names=True, axes=axes[0], show=False)
    axes[0].set_title(f"{len(raw.ch_names)} EEG electrodes (10-10 system)", fontsize=11)

    # Highlight the hand-knob electrodes the left/right contrast depends on.
    positions = raw.get_montage().get_positions()["ch_pos"]
    xs = np.array([positions[ch][0] for ch in raw.ch_names])
    ys = np.array([positions[ch][1] for ch in raw.ch_names])
    axes[1].scatter(xs, ys, s=22, color="#cccccc", zorder=2)
    for label, channel in CONTRALATERAL_HAND_CHANNELS.items():
        if channel not in raw.ch_names:
            continue
        x, y = positions[channel][0], positions[channel][1]
        axes[1].scatter([x], [y], s=180, color=CLASS_COLORS[label], zorder=3)
        axes[1].annotate(
            f"{channel}\n({label.split('_')[0].lower()} hand)",
            (x, y),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axes[1].set_aspect("equal")
    axes[1].set_title("Sensorimotor electrodes used for the\ncontralateral ERD check", fontsize=11)
    axes[1].axis("off")

    fig.suptitle(
        f"{dataset_label(config)} - {spec.subject_id} {spec.run_id}\nChannel montage",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _synthetic_banner(fig, config)
    return _save(fig, out_dir, "02_channel_montage")


def plot_event_timeline(spec: RunSpec, config: DatasetConfig, out_dir: Path) -> Path:
    """Figure 03 -- annotation timeline with run-resolved class labels."""
    raw = load_raw(spec, config, preload=False)
    table = annotations_table(spec, raw)

    fig, ax = plt.subplots(figsize=(13, 4))
    seen: set[str] = set()
    for _, row in table.iterrows():
        label = row["class_label"] or "UNMAPPED"
        color = CLASS_COLORS.get(label, "#000000")
        ax.axvspan(
            row["onset_time"],
            row["onset_time"] + row["duration"],
            color=color,
            alpha=0.55,
            label=label if label not in seen else None,
        )
        seen.add(label)
        ax.text(
            row["onset_time"] + row["duration"] / 2,
            1.06,
            row["event_code"],
            ha="center",
            fontsize=7,
        )

    ax.set_xlim(0, float(raw.n_times) / raw.info["sfreq"])
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.legend(loc="lower right", ncol=3, fontsize=9, framealpha=0.9)
    ax.set_title(
        f"{dataset_label(config)} - {spec.subject_id} {spec.run_id} "
        f"({spec.task_type.value})\n"
        f"run-specific mapping: "
        + ", ".join(f"{k}->{v}" for k, v in sorted(spec.event_map.items())),
        fontsize=11,
        loc="left",
    )
    fig.tight_layout()
    _synthetic_banner(fig, config)
    return _save(fig, out_dir, "03_event_timeline")


# --------------------------------------------------------------------------------------
# 4-5: single-epoch figures
# --------------------------------------------------------------------------------------


def _first_epoch_of(dataset: EEGDataset, label: str) -> Optional[int]:
    matches = np.flatnonzero((dataset.metadata["class_label"] == label).to_numpy())
    return int(matches[0]) if matches.size else None


def plot_example_epoch(
    dataset: EEGDataset, label: str, config: DatasetConfig, out_dir: Path, name: str
) -> Optional[Path]:
    """Figures 04 and 05 -- one epoch of a given class, with a mu-band topography."""
    index = _first_epoch_of(dataset, label)
    if index is None:
        log.warning("no epoch of class %s to plot", label)
        return None

    row = dataset.metadata.iloc[index]
    epoch = dataset.X[index] * 1e6  # microvolts
    times = dataset.times
    channels = list(dataset.channels)

    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 2, width_ratios=[2.1, 1.0], hspace=0.35, wspace=0.25)

    # Butterfly plot of all channels.
    ax0 = fig.add_subplot(grid[0, 0])
    ax0.plot(times, epoch.T, linewidth=0.5, color="#888888", alpha=0.6)
    ax0.set_title(f"All {len(channels)} channels", loc="left", fontsize=11)
    ax0.set_ylabel("amplitude (uV)")
    ax0.grid(alpha=0.25)

    # The two hand-knob channels, which is where the class information lives.
    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
    for channel, color in (("C3", "#1b7837"), ("C4", "#762a83")):
        if channel in channels:
            ax1.plot(
                times, epoch[channels.index(channel)], linewidth=1.2, color=color, label=channel
            )
    ax1.legend(fontsize=9)
    ax1.set_xlabel("time from event onset (s)")
    ax1.set_ylabel("amplitude (uV)")
    ax1.set_title("Hand-knob electrodes", loc="left", fontsize=11)
    ax1.grid(alpha=0.25)

    # Mu-band power topography for this epoch.
    ax2 = fig.add_subplot(grid[:, 1])
    freqs, psd = welch(dataset.X[index], fs=float(row["sampling_rate"]), nperseg=256, axis=-1)
    mu_band = (freqs >= 8) & (freqs <= 13)
    mu_power = psd[:, mu_band].mean(axis=1)

    info = mne.create_info(channels, sfreq=float(row["sampling_rate"]), ch_types="eeg")
    info.set_montage(config.preprocessing.montage, on_missing="ignore")
    mne.viz.plot_topomap(mu_power, info, axes=ax2, show=False, cmap="viridis", contours=4)
    ax2.set_title("mu-band (8-13 Hz) power", fontsize=11)

    fig.suptitle(
        f"{dataset_label(config)}\n{label} - {row['subject_id']} {row['run_id']} "
        f"(event {row['event_code']}, onset {row['onset_time']:.2f} s, "
        f"window [{row['epoch_start']:.2f}, {row['epoch_end']:.2f}) s)",
        fontsize=12,
    )
    _synthetic_banner(fig, config)
    return _save(fig, out_dir, name)


# --------------------------------------------------------------------------------------
# 6-7: spectral figures
# --------------------------------------------------------------------------------------


def _band_psd(
    dataset: EEGDataset, label: str, channel: str, nperseg: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean PSD across all epochs of ``label`` at ``channel``."""
    mask = (dataset.metadata["class_label"] == label).to_numpy()
    index = list(dataset.channels).index(channel)
    sfreq = float(dataset.metadata["sampling_rate"].iloc[0])
    freqs, psd = welch(dataset.X[mask, index, :], fs=sfreq, nperseg=nperseg, axis=-1)
    return freqs, psd.mean(axis=0)


def plot_psd(dataset: EEGDataset, config: DatasetConfig, out_dir: Path) -> Path:
    """Figure 06 -- grand-average PSD, confirming the passband was applied."""
    sfreq = float(dataset.metadata["sampling_rate"].iloc[0])
    freqs, psd = welch(dataset.X, fs=sfreq, nperseg=256, axis=-1)
    grand = psd.mean(axis=0)  # (n_channels, n_freqs)
    pre = config.preprocessing

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].semilogy(freqs, grand.T, linewidth=0.4, color="#999999", alpha=0.45)
    axes[0].semilogy(freqs, grand.mean(axis=0), linewidth=2.0, color="#000000", label="mean")
    axes[0].axvspan(pre.l_freq, pre.h_freq, color="#2c7fb8", alpha=0.12, label="passband")
    axes[0].set_xlim(0, sfreq / 2)
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("PSD (V$^2$/Hz)")
    axes[0].set_title("All channels, all epochs", loc="left", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25, which="both")

    for label in dataset.classes:
        mask = (dataset.metadata["class_label"] == label).to_numpy()
        if not mask.any():
            continue
        axes[1].semilogy(
            freqs,
            psd[mask].mean(axis=(0, 1)),
            linewidth=1.6,
            color=CLASS_COLORS.get(label, None),
            label=f"{label} (n={int(mask.sum())})",
        )
    axes[1].axvspan(8, 13, color="#2c7fb8", alpha=0.10)
    axes[1].axvspan(13, 30, color="#d95f02", alpha=0.08)
    axes[1].set_xlim(0, min(50, sfreq / 2))
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("PSD (V$^2$/Hz)")
    axes[1].set_title("Per class, averaged over channels", loc="left", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25, which="both")

    fig.suptitle(
        f"{dataset_label(config)} - power spectral density\n"
        f"{dataset.metadata['subject_id'].nunique()} subject(s), {len(dataset)} epochs",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _synthetic_banner(fig, config)
    return _save(fig, out_dir, "06_power_spectral_density")


def plot_left_vs_right_psd(
    dataset: EEGDataset, config: DatasetConfig, out_dir: Path
) -> Path:
    """Figure 07 -- the contralateral-ERD check at C3 and C4.

    Expectation, from sensorimotor physiology: imagining the LEFT hand suppresses
    mu/beta power over the RIGHT hemisphere (C4), and imagining the RIGHT hand
    suppresses it over the LEFT hemisphere (C3).  The lateralisation index
    ``log(C4 / C3)`` should therefore be negative for left-hand imagery and
    positive for right-hand imagery.
    """
    channels = list(dataset.channels)
    available = [ch for ch in ("C3", "C4") if ch in channels]
    if len(available) < 2:
        raise RuntimeError("C3 and C4 are required for the lateralisation figure")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    lo, hi = config.preprocessing.l_freq - 2, config.preprocessing.h_freq + 2
    in_band_values: List[float] = []

    for ax, channel in zip(axes[:2], ("C3", "C4")):
        for label in (LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY, REST):
            mask = (dataset.metadata["class_label"] == label).to_numpy()
            if not mask.any():
                continue
            freqs, mean_psd = _band_psd(dataset, label, channel)
            in_band_values.extend(mean_psd[(freqs >= lo) & (freqs <= hi)].tolist())
            ax.semilogy(
                freqs,
                mean_psd,
                linewidth=1.8,
                color=CLASS_COLORS[label],
                label=f"{label} (n={int(mask.sum())})",
            )
        ax.axvspan(8, 13, color="#000000", alpha=0.06)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD (V$^2$/Hz)")
        hemisphere = "left" if channel == "C3" else "right"
        ax.set_title(f"{channel} ({hemisphere} hemisphere)", loc="left", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, which="both")

    # Scale the log axis to the in-band data; the filter stopband floor is many
    # orders of magnitude below it and would otherwise flatten the whole plot.
    if in_band_values:
        finite = np.asarray([v for v in in_band_values if v > 0])
        if finite.size:
            for ax in axes[:2]:
                ax.set_ylim(finite.min() * 0.5, finite.max() * 2.0)

    # Lateralisation index per class.
    labels: List[str] = []
    values: List[float] = []
    for label in (LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY, REST):
        mask = (dataset.metadata["class_label"] == label).to_numpy()
        if not mask.any():
            continue
        freqs, psd_c3 = _band_psd(dataset, label, "C3")
        _, psd_c4 = _band_psd(dataset, label, "C4")
        mu = (freqs >= 8) & (freqs <= 13)
        labels.append(label.replace("_IMAGERY", "").replace("_", " ").title())
        values.append(float(np.log(psd_c4[mu].mean() / psd_c3[mu].mean())))

    colors = [
        CLASS_COLORS[c]
        for c in (LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY, REST)
        if (dataset.metadata["class_label"] == c).any()
    ]
    axes[2].bar(labels, values, color=colors, alpha=0.85)
    axes[2].axhline(0, color="#000000", linewidth=1.0)
    axes[2].set_ylabel(r"lateralisation index  $\log(C4_{\mu} / C3_{\mu})$")
    axes[2].set_title(
        "negative = right hemisphere suppressed\n(expected for LEFT hand imagery)",
        loc="left",
        fontsize=10,
    )
    axes[2].grid(alpha=0.25, axis="y")
    # Headroom so the bar value labels never collide with the axes title.
    if values:
        span = max(abs(min(values)), abs(max(values))) or 1.0
        axes[2].set_ylim(min(0.0, min(values)) - 0.25 * span,
                         max(0.0, max(values)) + 0.25 * span)
    for i, value in enumerate(values):
        axes[2].text(
            i,
            value + (0.03 if value >= 0 else -0.03),
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(
        f"{dataset_label(config)} - left vs right hand motor imagery\n"
        f"contralateral mu/beta desynchronisation "
        f"({dataset.metadata['subject_id'].nunique()} subject(s), {len(dataset)} epochs)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _synthetic_banner(fig, config)
    return _save(fig, out_dir, "07_psd_left_vs_right")


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def generate_all_figures(
    config: DatasetConfig, spec: Optional[RunSpec] = None
) -> List[Path]:
    """Produce every required figure, returning the paths written."""
    paths = config.resolved_paths
    out_dir = paths.figures
    if spec is None:
        spec = RunSpec(subject=config.subjects[0], run=config.runs[0])

    dataset = load_dataset(config, normalize=False)

    produced: List[Optional[Path]] = [
        plot_raw_sample(spec, config, out_dir),
        plot_montage(spec, config, out_dir),
        plot_event_timeline(spec, config, out_dir),
        plot_example_epoch(dataset, LEFT_HAND_IMAGERY, config, out_dir, "04_epoch_left_hand"),
        plot_example_epoch(dataset, RIGHT_HAND_IMAGERY, config, out_dir, "05_epoch_right_hand"),
        plot_psd(dataset, config, out_dir),
        plot_left_vs_right_psd(dataset, config, out_dir),
    ]
    return [p for p in produced if p is not None]


def lateralization_index(dataset: EEGDataset, label: str) -> float:
    """``log(mu power at C4 / mu power at C3)`` for one class.

    Exposed separately from the figure so tests can assert on the number.
    """
    freqs, psd_c3 = _band_psd(dataset, label, "C3")
    _, psd_c4 = _band_psd(dataset, label, "C4")
    mu = (freqs >= 8) & (freqs <= 13)
    return float(np.log(psd_c4[mu].mean() / psd_c3[mu].mean()))
