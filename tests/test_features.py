"""Phase 2A -- tests for spectral feature extraction.

Offline: every case is built from synthetic arrays with a known spectrum, so the
band-power maths is checked against an answer computed by hand rather than
against whatever the code happened to produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.bandpower import (
    BANDS,
    LOG_FLOOR,
    PASSBAND_HZ,
    band_mask,
    band_power,
    log_power,
    n_bins,
    relative_band_power,
    total_passband_power,
    welch_psd,
)
from src.features.extract import (
    CHANNEL_SETS,
    IDENTITY_COLUMNS,
    build_feature_table,
)

SFREQ = 160.0
N_TIMES = 640
#: The full 64-channel montage, in stored order.  The tests need all of it, not
#: just the ROI: with a 21-channel montage the "all64" and "roi21" channel sets
#: would be identical and the ablation selection logic would go untested.
CHANNELS = ("FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
            "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
            "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
            "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8",
            "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
            "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz")


def sinusoid_epochs(freq_hz: float, amplitude: float = 1e-5, n_epochs: int = 3):
    t = np.arange(N_TIMES) / SFREQ
    wave = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return np.tile(wave, (n_epochs, len(CHANNELS), 1))


def metadata_for(n_epochs: int, label: str = "REST") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epoch_uid": [f"E{i:07d}" for i in range(n_epochs)],
            "subject_id": ["S001"] * n_epochs,
            "run_id": ["R04"] * n_epochs,
            "subject": [1] * n_epochs,
            "run": [4] * n_epochs,
            "class_label": [label] * n_epochs,
            "event_code": ["T0"] * n_epochs,
            "onset_time": np.arange(n_epochs, dtype=float),
            "source": ["physionet"] * n_epochs,
        }
    )


class TestBands:
    def test_mu_and_beta_partition_the_passband(self):
        """No shared bin, no gap -- otherwise relative powers cannot sum to one."""
        freqs = np.linspace(0, 80, 129)
        mu = band_mask(freqs, BANDS["mu"])
        beta = band_mask(freqs, BANDS["beta"])
        passband = band_mask(freqs, PASSBAND_HZ)
        assert not (mu & beta).any()
        assert np.array_equal(mu | beta, passband)

    def test_band_intervals_are_half_open_at_the_top(self):
        freqs = np.array([7.9, 8.0, 12.9, 13.0, 29.9, 30.0])
        assert band_mask(freqs, BANDS["mu"]).tolist() == [False, True, True, False, False, False]
        assert band_mask(freqs, BANDS["beta"]).tolist() == [False, False, False, True, True, False]

    def test_an_empty_band_raises_rather_than_returning_zero(self):
        freqs = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="no PSD bins"):
            band_power(np.zeros((1, 1, 3)), freqs, (40.0, 50.0))


class TestBandPower:
    def test_a_mu_sinusoid_lands_in_the_mu_band(self):
        freqs, psd = welch_psd(sinusoid_epochs(10.0), SFREQ)
        mu = band_power(psd, freqs, BANDS["mu"])
        beta = band_power(psd, freqs, BANDS["beta"])
        assert (mu > 100 * beta).all()

    def test_a_beta_sinusoid_lands_in_the_beta_band(self):
        freqs, psd = welch_psd(sinusoid_epochs(22.0), SFREQ)
        mu = band_power(psd, freqs, BANDS["mu"])
        beta = band_power(psd, freqs, BANDS["beta"])
        assert (beta > 100 * mu).all()

    def test_power_scales_with_the_square_of_amplitude(self):
        _, psd_1 = welch_psd(sinusoid_epochs(10.0, amplitude=1e-5), SFREQ)
        freqs, psd_2 = welch_psd(sinusoid_epochs(10.0, amplitude=2e-5), SFREQ)
        p1 = band_power(psd_1, freqs, BANDS["mu"])
        p2 = band_power(psd_2, freqs, BANDS["mu"])
        assert np.allclose(p2 / p1, 4.0, rtol=1e-6)


class TestRelativePower:
    def test_relative_powers_sum_to_exactly_one(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1e-5, size=(6, len(CHANNELS), N_TIMES))
        freqs, psd = welch_psd(X, SFREQ)
        total = total_passband_power(psd, freqs)
        total_bins = n_bins(freqs, PASSBAND_HZ)

        rel = [
            relative_band_power(
                band_power(psd, freqs, BANDS[name]), total, n_bins(freqs, BANDS[name]), total_bins
            )
            for name in BANDS
        ]
        assert np.allclose(sum(rel), 1.0, atol=1e-12)

    def test_bin_counts_not_nominal_widths_are_used(self):
        """Nominal widths (5 and 17 Hz) do not match the bin counts (8 and 27)."""
        freqs, _ = welch_psd(sinusoid_epochs(10.0), SFREQ)
        assert n_bins(freqs, BANDS["mu"]) + n_bins(freqs, BANDS["beta"]) == n_bins(
            freqs, PASSBAND_HZ
        )
        # The mismatch this guards against: bins are not proportional to width.
        width_ratio = (BANDS["mu"][1] - BANDS["mu"][0]) / (PASSBAND_HZ[1] - PASSBAND_HZ[0])
        bin_ratio = n_bins(freqs, BANDS["mu"]) / n_bins(freqs, PASSBAND_HZ)
        assert not np.isclose(width_ratio, bin_ratio, atol=1e-4)

    def test_relative_power_is_scale_free(self):
        """The property that should let ratio features survive a subject change."""
        freqs, psd_small = welch_psd(sinusoid_epochs(10.0, amplitude=1e-6), SFREQ)
        _, psd_large = welch_psd(sinusoid_epochs(10.0, amplitude=1e-3), SFREQ)
        total_bins = n_bins(freqs, PASSBAND_HZ)
        mu_bins = n_bins(freqs, BANDS["mu"])

        rel_small = relative_band_power(
            band_power(psd_small, freqs, BANDS["mu"]),
            total_passband_power(psd_small, freqs), mu_bins, total_bins,
        )
        rel_large = relative_band_power(
            band_power(psd_large, freqs, BANDS["mu"]),
            total_passband_power(psd_large, freqs), mu_bins, total_bins,
        )
        assert np.allclose(rel_small, rel_large, rtol=1e-9)


class TestLogPower:
    def test_log_of_zero_is_floored_not_infinite(self):
        assert np.isfinite(log_power(np.array([0.0]))).all()
        assert log_power(np.array([0.0]))[0] == pytest.approx(np.log(LOG_FLOOR))

    def test_ordinary_values_are_untouched_by_the_floor(self):
        values = np.array([1e-12, 1e-11])
        assert np.allclose(log_power(values), np.log(values))

    def test_a_scale_change_becomes_an_additive_offset(self):
        """Why the log matters for the 2.33x between-subject amplitude spread."""
        values = np.array([1e-12, 4e-12, 9e-12])
        assert np.allclose(log_power(values * 100) - log_power(values), np.log(100))


class TestFeatureTable:
    def _table(self, n_epochs=5, label="REST"):
        rng = np.random.default_rng(7)
        X = rng.normal(0, 1e-5, size=(n_epochs, len(CHANNELS), N_TIMES))
        return build_feature_table(X, metadata_for(n_epochs, label), CHANNELS, SFREQ)

    def test_one_row_per_epoch(self):
        table = self._table(n_epochs=9)
        assert len(table.frame) == 9

    def test_identity_columns_are_carried_and_are_not_features(self):
        table = self._table()
        for column in IDENTITY_COLUMNS:
            assert column in table.frame.columns
            assert column not in table.feature_columns

    def test_features_are_finite(self):
        assert np.isfinite(self._table().matrix()).all()

    def test_extraction_is_deterministic(self):
        a, b = self._table(), self._table()
        assert np.allclose(a.matrix(), b.matrix())

    def test_lateralization_is_log_c4_minus_log_c3(self):
        table = self._table()
        expected = (
            table.frame["C4_log_mu"].to_numpy() - table.frame["C3_log_mu"].to_numpy()
        )
        assert np.allclose(table.frame["lateralization_mu"].to_numpy(), expected)

    def test_the_two_hemispheric_contrasts_are_opposite(self):
        table = self._table()
        assert np.allclose(
            table.frame["lateralization_mu"], -table.frame["c3_minus_c4_log_mu"]
        )

    def test_erd_contrast_is_relative_to_the_roi_mean(self):
        table = self._table()
        roi_mean = table.frame["roi_mean_log_mu"].to_numpy()
        expected = table.frame["C3_log_mu"].to_numpy() - roi_mean
        assert np.allclose(table.frame["erd_C3_vs_roi_mu"].to_numpy(), expected)

    def test_every_feature_has_a_family_and_a_channel(self):
        table = self._table()
        for column in table.feature_columns:
            assert table.family[column] in {"absolute", "relative"}
            assert table.channel_of[column]


class TestChannelSetSelection:
    def _table(self):
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1e-5, size=(4, len(CHANNELS), N_TIMES))
        return build_feature_table(X, metadata_for(4), CHANNELS, SFREQ)

    def test_roi_selection_is_a_subset_of_all(self):
        table = self._table()
        assert set(table.columns_for(channel_set="roi21")) <= set(
            table.columns_for(channel_set="all64")
        )

    def test_c3c4_selection_uses_only_c3_and_c4(self):
        table = self._table()
        for column in table.columns_for(channel_set="c3c4"):
            channel = table.channel_of[column]
            assert channel in {"C3", "C4", "derived"}

    def test_c3c4_excludes_roi_referenced_features(self):
        """Keeping them would smuggle 21 channels into a two-channel condition."""
        table = self._table()
        columns = table.columns_for(channel_set="c3c4")
        assert not [c for c in columns if c.startswith("erd_")]
        assert not [c for c in columns if c.startswith("roi_")]

    def test_family_filtering_splits_the_feature_set(self):
        table = self._table()
        absolute = set(table.columns_for(families=("absolute",)))
        relative = set(table.columns_for(families=("relative",)))
        assert absolute and relative
        assert not (absolute & relative)
        assert absolute | relative == set(table.feature_columns)

    def test_channel_sets_are_ordered_by_size(self):
        table = self._table()
        sizes = [len(table.columns_for(channel_set=cs)) for cs in ("c3c4", "roi21", "all64")]
        assert sizes[0] < sizes[1] < sizes[2]

    def test_all_named_channel_sets_resolve(self):
        table = self._table()
        for name in CHANNEL_SETS:
            assert len(table.columns_for(channel_set=name)) > 0
