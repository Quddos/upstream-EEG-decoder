"""Phase 2A -- reproducible per-epoch feature extraction.

Produces one row per epoch, carrying the full identity of that epoch
(``epoch_uid``, ``subject_id``, ``run_id``, ``class_label``, ``event_code``,
``onset_time``, ``source``) alongside the numeric features, so every row stays
traceable back to :file:`data/metadata/epoch_index.csv`.

Feature families
----------------
Two families are kept deliberately separate, because they behave differently
across subjects and Phase 2's whole question is about cross-subject stability:

``absolute``
    ``log`` band power per channel.  Physically interpretable, but it inherits
    the between-subject amplitude differences Phase 1.7 measured (2.33x in median
    epoch amplitude, S002 126 uV -> S004 294 uV).  A model can therefore use these
    to recognise *who* is being recorded, not only what they imagined.
``relative``
    within-epoch ratios and contrasts -- relative band power, the C3/C4
    lateralisation index, and the spatial ERD contrasts below.  These are
    scale-free by construction, so a subject's overall amplitude cancels.

The ``feature_family`` mapping is exported so the Phase 2 analysis can fit models
on each family separately and report which one actually transfers.

A note on "contralateral ERD"
-----------------------------
Classic ERD is a power *decrease relative to a pre-cue baseline*.  This dataset
cannot supply one: the epoch window is ``[0, 4)`` from cue onset, deliberately,
because any pre-onset window would fall inside the previous trial and import its
class -- a documented label-leakage decision in
``docs/PREPROCESSING_DECISIONS.md``.  Nothing before the cue is stored.

Two alternatives were rejected. Referencing each epoch to that subject's REST
epochs would use the held-out subject's own labels at test time, which is
leakage for the three-class task.  Referencing to the recording's mean would be
transductive -- legitimate for some purposes, but it makes the feature depend on
which other epochs happen to be present, so a single epoch could not be featurised
in isolation, and the real-time controller in Phase 3 must be able to do exactly
that.

What is computed instead is the *spatial* signature of ERD, entirely within one
epoch: the log power at a hand-knob electrode minus the log mean power across the
sensorimotor ROI.  Contralateral ERD makes C3 (or C4) quieter than the strip
around it, so this contrast captures the same phenomenon without any baseline,
any other epoch, or any label.  It is a proxy, and the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import DatasetConfig
from src.data.physionet_spec import SENSORIMOTOR_ROI, roi_indices
from src.features.bandpower import (
    BANDS,
    PASSBAND_HZ,
    absolute_band_powers,
    log_power,
    n_bins,
    relative_band_power,
    total_passband_power,
    welch_psd,
)
from src.io.store import load_dataset

#: Identity columns copied verbatim from the epoch index onto every feature row.
#: These are never features; :mod:`tests.test_leakage` asserts they are excluded
#: from any model input.
IDENTITY_COLUMNS: Tuple[str, ...] = (
    "epoch_uid",
    "subject_id",
    "run_id",
    "subject",
    "run",
    "class_label",
    "event_code",
    "onset_time",
    "source",
)

#: The hand-knob pair the lateralisation and ERD features are built on.
LEFT_HEMISPHERE_HAND = "C3"
RIGHT_HEMISPHERE_HAND = "C4"

#: Named channel sets for the Phase 2D ablation.
CHANNEL_SETS: Mapping[str, Optional[Tuple[str, ...]]] = {
    "all64": None,                       # every channel in the montage
    "roi21": SENSORIMOTOR_ROI,           # the documented sensorimotor ROI
    "c3c4": (LEFT_HEMISPHERE_HAND, RIGHT_HEMISPHERE_HAND),
}


@dataclass(frozen=True)
class FeatureTable:
    """Per-epoch features with their identity columns and family labels."""

    frame: pd.DataFrame
    feature_columns: Tuple[str, ...]
    #: ``{column: "absolute" | "relative"}``
    family: Mapping[str, str]
    #: ``{column: channel or "derived"}``
    channel_of: Mapping[str, str]
    #: ``{column: every channel the column is computed from}``.  Derived features
    #: declare their inputs explicitly so a channel-set restriction can exclude
    #: them by construction rather than by matching on their name.
    requires: Mapping[str, Tuple[str, ...]]
    channels: Tuple[str, ...]
    sfreq: float

    def matrix(self, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        """Model input as ``(n_epochs, n_features)``, identity columns excluded."""
        cols = list(columns) if columns is not None else list(self.feature_columns)
        return self.frame[cols].to_numpy(dtype=np.float64)

    def columns_for(
        self,
        *,
        channel_set: Optional[str] = None,
        families: Optional[Sequence[str]] = None,
    ) -> Tuple[str, ...]:
        """Feature columns restricted to a channel set and/or family.

        A column survives only when *every* channel it was computed from is inside
        the requested set.  That is what makes the Phase 2D ablation honest: the
        ROI-referenced contrasts (``roi_mean_*``, ``erd_*``) each depend on all 21
        ROI electrodes, so they are dropped from the two-channel C3/C4 condition
        automatically.  Keeping them would let 21 channels of information into a
        condition labelled as using two, and the ablation would measure nothing.
        """
        cols = list(self.feature_columns)
        if channel_set is not None:
            members = CHANNEL_SETS[channel_set]
            allowed = set(self.channels) if members is None else set(members)
            cols = [c for c in cols if set(self.requires[c]) <= allowed]
        if families is not None:
            wanted = set(families)
            cols = [c for c in cols if self.family[c] in wanted]
        return tuple(cols)


def _channel_feature_frames(
    psd: np.ndarray,
    freqs: np.ndarray,
    channels: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, str], Dict[str, str], Dict[str, Tuple[str, ...]]]:
    """Per-channel absolute and relative band-power features."""
    columns: Dict[str, np.ndarray] = {}
    family: Dict[str, str] = {}
    channel_of: Dict[str, str] = {}
    requires: Dict[str, Tuple[str, ...]] = {}

    powers = absolute_band_powers(psd, freqs, BANDS)
    total = total_passband_power(psd, freqs)
    total_bins = n_bins(freqs, PASSBAND_HZ)

    for band_name, power in powers.items():
        logged = log_power(power)
        relative = relative_band_power(
            power, total, n_bins(freqs, BANDS[band_name]), total_bins
        )
        for i, channel in enumerate(channels):
            # (10) log-transformed spectral features; (1-4) C3/C4 mu and beta.
            name = f"{channel}_log_{band_name}"
            columns[name] = logged[:, i]
            family[name] = "absolute"
            channel_of[name] = channel
            requires[name] = (channel,)

            # (5-6) relative band power.
            name = f"{channel}_rel_{band_name}"
            columns[name] = relative[:, i]
            family[name] = "relative"
            channel_of[name] = channel
            requires[name] = (channel,)

    return columns, family, channel_of, requires


def _derived_features(
    psd: np.ndarray,
    freqs: np.ndarray,
    channels: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, str], Dict[str, str], Dict[str, Tuple[str, ...]]]:
    """Lateralisation and spatial-ERD contrasts built from C3, C4 and the ROI."""
    columns: Dict[str, np.ndarray] = {}
    family: Dict[str, str] = {}
    channel_of: Dict[str, str] = {}
    requires: Dict[str, Tuple[str, ...]] = {}
    pair = (LEFT_HEMISPHERE_HAND, RIGHT_HEMISPHERE_HAND)

    index = {name: i for i, name in enumerate(channels)}
    powers = absolute_band_powers(psd, freqs, BANDS)

    have_pair = LEFT_HEMISPHERE_HAND in index and RIGHT_HEMISPHERE_HAND in index
    roi_present = [c for c in SENSORIMOTOR_ROI if c in index]
    have_roi = len(roi_present) == len(SENSORIMOTOR_ROI)

    for band_name, power in powers.items():
        logged = log_power(power)

        if have_pair:
            c3 = logged[:, index[LEFT_HEMISPHERE_HAND]]
            c4 = logged[:, index[RIGHT_HEMISPHERE_HAND]]

            # (7) lateralisation index, identical in definition to
            # viz.figures.lateralization_index and the Phase 1.5/1.6 reports:
            # log(C4 / C3) == log C4 - log C3.
            name = f"lateralization_{band_name}"
            columns[name] = c4 - c3
            family[name] = "relative"
            channel_of[name] = "derived"
            requires[name] = pair

            # Inter-channel difference in the other direction, kept because the
            # absolute hemispheric balance is not the same quantity as its sign.
            name = f"c3_minus_c4_log_{band_name}"
            columns[name] = c3 - c4
            family[name] = "relative"
            channel_of[name] = "derived"
            requires[name] = pair

        if have_roi:
            picks = [index[c] for c in SENSORIMOTOR_ROI]
            roi_mean_log = logged[:, picks].mean(axis=1)

            # (9) ROI aggregate band power.
            name = f"roi_mean_log_{band_name}"
            columns[name] = roi_mean_log
            family[name] = "absolute"
            channel_of[name] = "derived"
            requires[name] = tuple(SENSORIMOTOR_ROI)

            name = f"roi_std_log_{band_name}"
            columns[name] = logged[:, picks].std(axis=1)
            family[name] = "relative"
            channel_of[name] = "derived"
            requires[name] = tuple(SENSORIMOTOR_ROI)

            # (8) contralateral ERD proxy: how quiet a hand-knob electrode is
            # relative to the sensorimotor strip in the same epoch.  Negative
            # means that electrode is desynchronised against its own surround.
            for hand in (LEFT_HEMISPHERE_HAND, RIGHT_HEMISPHERE_HAND):
                if hand not in index:
                    continue
                name = f"erd_{hand}_vs_roi_{band_name}"
                columns[name] = logged[:, index[hand]] - roi_mean_log
                family[name] = "relative"
                channel_of[name] = "derived"
                requires[name] = tuple(sorted({hand, *SENSORIMOTOR_ROI}))

    return columns, family, channel_of, requires


def build_feature_table(
    X: np.ndarray,
    metadata: pd.DataFrame,
    channels: Sequence[str],
    sfreq: float,
) -> FeatureTable:
    """Featurise ``X`` (volts) into one row per epoch.

    Every column is a function of a single epoch only, so this is invariant to
    which other epochs are present -- the property the leakage audit checks and
    the property Phase 3's real-time controller will need.
    """
    if X.shape[0] != len(metadata):
        raise ValueError(
            f"{X.shape[0]} epochs but {len(metadata)} metadata rows"
        )

    freqs, psd = welch_psd(X, sfreq)

    per_channel, family_a, channel_a, requires_a = _channel_feature_frames(
        psd, freqs, channels
    )
    derived, family_b, channel_b, requires_b = _derived_features(psd, freqs, channels)

    columns = {**per_channel, **derived}
    family = {**family_a, **family_b}
    channel_of = {**channel_a, **channel_b}
    requires = {**requires_a, **requires_b}

    ordered = sorted(columns)
    frame = pd.DataFrame({name: columns[name] for name in ordered})

    for column in IDENTITY_COLUMNS:
        if column in metadata.columns:
            frame.insert(0, column, metadata[column].to_numpy())

    return FeatureTable(
        frame=frame,
        feature_columns=tuple(ordered),
        family=family,
        channel_of=channel_of,
        requires=requires,
        channels=tuple(channels),
        sfreq=float(sfreq),
    )


def extract_features(
    config: DatasetConfig,
    *,
    subjects: Optional[Sequence[int]] = None,
    data_root: Optional[str] = None,
) -> FeatureTable:
    """Load the processed dataset and featurise it.

    ``normalize=False`` is not optional here.  The default load-time
    ``zscore_per_channel`` sets every channel of every epoch to unit variance,
    which erases the between-channel power contrasts that most of these features
    measure -- on S001 it collapses the lateralisation index from -0.122 to
    -0.008.  See :mod:`src.features.bandpower` for the measurement.
    """
    if data_root is not None:
        from dataclasses import replace
        from pathlib import Path

        config = replace(
            config, paths=replace(config.paths, root=Path(data_root).resolve())
        )

    dataset = load_dataset(config, subjects=subjects, normalize=False)
    sfreq = float(dataset.metadata["sampling_rate"].iloc[0])
    return build_feature_table(
        dataset.X.astype(np.float64), dataset.metadata, dataset.channels, sfreq
    )
