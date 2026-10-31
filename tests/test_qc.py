"""Tests for the Phase 1.6 quality-control measurements.

These run entirely on constructed arrays -- no real recording and no fixture build
-- because every property under test is a property of the *measurement*, not of any
particular dataset.  The one thing that must never break is the guarantee that this
module only describes: none of it may change an epoch, a threshold or a count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PreprocessingConfig
from src.data.physionet_spec import PHASE1_CLASSES
from src.qc.quality import (
    CHANNEL_REGIONS,
    EPOCH_QC_COLUMNS,
    channel_qc_table,
    class_association,
    class_qc_table,
    describe,
    distribution_table,
    dominant_region_per_epoch,
    epoch_qc_table,
    lateralization_index,
    measure_amplitudes,
    mu_power_per_epoch,
    pairwise_class_contrasts,
    per_epoch_lateralization,
    permutation_difference,
    pick_representatives,
    region_of,
    region_qc_table,
)

THRESHOLD_V = PreprocessingConfig().reject_log_threshold_v  # 150 uV
CHANNELS = ("C3", "C4", "AF7", "O2")


def make_epochs(peak_to_peaks_uv):
    """Epochs whose per-channel peak-to-peak amplitudes are exactly as specified.

    ``peak_to_peaks_uv`` is ``(n_epochs, n_channels)``.  Each channel is built as a
    symmetric square wave, so peak-to-peak is the requested value and max|x| is
    exactly half of it -- which makes every expected number checkable by hand.
    """
    p2p = np.asarray(peak_to_peaks_uv, dtype=float)
    n_epochs, n_channels = p2p.shape
    X = np.zeros((n_epochs, n_channels, 8), dtype=float)
    X[:, :, ::2] = (p2p / 2.0)[:, :, None] * 1e-6
    X[:, :, 1::2] = (-p2p / 2.0)[:, :, None] * 1e-6
    return X


def make_metadata(classes, n_epochs=None):
    n = n_epochs if n_epochs is not None else len(classes)
    return pd.DataFrame(
        {
            "epoch_uid": [f"E{i:07d}" for i in range(n)],
            "subject_id": ["S001"] * n,
            "run_id": ["R04"] * n,
            "class_label": list(classes),
            "onset_time": np.arange(n, dtype=float) * 4.1,
        }
    )


class TestAmplitudeMeasurement:
    def test_peak_to_peak_is_the_max_over_channels(self):
        measures = measure_amplitudes(
            make_epochs([[10.0, 200.0, 50.0, 30.0]]), CHANNELS, THRESHOLD_V
        )
        assert measures.epoch_p2p_max[0] == pytest.approx(200.0)

    def test_absolute_amplitude_is_half_the_peak_to_peak_for_a_square_wave(self):
        measures = measure_amplitudes(
            make_epochs([[10.0, 200.0, 50.0, 30.0]]), CHANNELS, THRESHOLD_V
        )
        assert measures.epoch_abs_max[0] == pytest.approx(100.0)

    def test_threshold_is_reported_in_microvolts(self):
        measures = measure_amplitudes(make_epochs([[1.0] * 4]), CHANNELS, THRESHOLD_V)
        assert measures.threshold_uv == pytest.approx(150.0)

    def test_flag_is_exactly_one_or_more_channels_over_threshold(self):
        measures = measure_amplitudes(
            make_epochs(
                [
                    [10.0, 20.0, 30.0, 40.0],      # nothing over
                    [10.0, 20.0, 151.0, 40.0],     # one channel over
                    [200.0, 300.0, 400.0, 10.0],   # three over
                ]
            ),
            CHANNELS,
            THRESHOLD_V,
        )
        assert measures.artifact_flag.tolist() == [False, True, True]
        assert measures.artifact_channel_count.tolist() == [0, 1, 3]
        # The two must agree for every epoch -- this is the invariant the whole
        # report leans on when it reads a count of 0 as "not flagged".
        assert np.array_equal(
            measures.artifact_flag, measures.artifact_channel_count >= 1
        )

    def test_exactly_at_the_threshold_is_not_flagged(self):
        """The pipeline uses a strict `>`; the QC must not quietly become `>=`."""
        measures = measure_amplitudes(
            make_epochs([[150.0, 10.0, 10.0, 10.0]]), CHANNELS, THRESHOLD_V
        )
        assert not measures.artifact_flag[0]

    def test_rejects_a_channel_count_mismatch(self):
        with pytest.raises(ValueError, match="channel names"):
            measure_amplitudes(make_epochs([[1.0] * 4]), ("C3", "C4"), THRESHOLD_V)


class TestEpochTable:
    def test_columns_are_exactly_the_brief_s_columns_in_order(self):
        X = make_epochs([[10.0, 20.0, 30.0, 40.0], [200.0, 10.0, 10.0, 10.0]])
        measures = measure_amplitudes(X, CHANNELS, THRESHOLD_V)
        table = epoch_qc_table(make_metadata(["REST", "LEFT_HAND_IMAGERY"]), measures)
        assert tuple(table.columns) == EPOCH_QC_COLUMNS
        assert len(table) == 2

    def test_refuses_to_build_a_misaligned_table(self):
        measures = measure_amplitudes(make_epochs([[1.0] * 4] * 3), CHANNELS, THRESHOLD_V)
        with pytest.raises(ValueError, match="metadata rows"):
            epoch_qc_table(make_metadata(["REST"]), measures)


class TestDistribution:
    def test_statistics_match_hand_computed_values(self):
        stats = describe(np.array([1.0, 2.0, 3.0, 4.0]))
        assert stats["min"] == 1.0
        assert stats["max"] == 4.0
        assert stats["mean"] == pytest.approx(2.5)
        assert stats["median"] == pytest.approx(2.5)
        assert stats["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
        assert stats["p95"] == pytest.approx(np.percentile([1, 2, 3, 4], 95))

    def test_std_is_the_sample_not_population_deviation(self):
        values = np.array([1.0, 5.0])
        assert describe(values)["std"] == pytest.approx(np.std(values, ddof=1))
        assert describe(values)["std"] != pytest.approx(np.std(values))

    def test_single_epoch_has_no_defined_spread(self):
        assert np.isnan(describe(np.array([7.0]))["std"])

    def test_table_reports_both_measures(self):
        measures = measure_amplitudes(
            make_epochs([[10.0] * 4, [200.0] * 4]), CHANNELS, THRESHOLD_V
        )
        table = distribution_table(measures)
        assert set(table.index) == {"peak_to_peak_uv", "peak_abs_uv"}
        assert table.loc["peak_to_peak_uv", "max"] == pytest.approx(200.0)
        assert table.loc["peak_abs_uv", "max"] == pytest.approx(100.0)


class TestChannelTable:
    def test_counts_each_channel_on_its_own_amplitude(self):
        # AF7 is over threshold in both epochs; C3 in neither, even though both
        # epochs are flagged.  A channel must not inherit its epoch's flag.
        X = make_epochs([[10.0, 10.0, 200.0, 10.0], [10.0, 10.0, 300.0, 10.0]])
        table = channel_qc_table(measure_amplitudes(X, CHANNELS, THRESHOLD_V))
        by_channel = table.set_index("channel")
        assert by_channel.loc["AF7", "flagged_epochs"] == 2
        assert by_channel.loc["AF7", "pct_epochs_flagged"] == pytest.approx(100.0)
        assert by_channel.loc["C3", "flagged_epochs"] == 0

    def test_every_channel_appears_and_the_order_is_descending(self):
        X = make_epochs([[10.0, 200.0, 300.0, 10.0]])
        table = channel_qc_table(measure_amplitudes(X, CHANNELS, THRESHOLD_V))
        assert set(table["channel"]) == set(CHANNELS)
        assert table["flagged_epochs"].is_monotonic_decreasing


class TestClassTable:
    def test_rates_are_per_class(self):
        X = make_epochs(
            [
                [200.0, 10.0, 10.0, 10.0],  # LEFT, flagged
                [10.0, 10.0, 10.0, 10.0],   # LEFT, clean
                [10.0, 10.0, 10.0, 10.0],   # REST, clean
            ]
        )
        measures = measure_amplitudes(X, CHANNELS, THRESHOLD_V)
        epoch_qc = epoch_qc_table(
            make_metadata(["LEFT_HAND_IMAGERY", "LEFT_HAND_IMAGERY", "REST"]), measures
        )
        table = class_qc_table(epoch_qc, PHASE1_CLASSES).set_index("class")
        assert table.loc["LEFT_HAND_IMAGERY", "pct_flagged"] == pytest.approx(50.0)
        assert table.loc["REST", "pct_flagged"] == pytest.approx(0.0)

    def test_a_class_with_no_epochs_still_appears(self):
        """A missing class must never be silently read as a clean one."""
        X = make_epochs([[10.0, 10.0, 10.0, 10.0]])
        measures = measure_amplitudes(X, CHANNELS, THRESHOLD_V)
        epoch_qc = epoch_qc_table(make_metadata(["REST"]), measures)
        table = class_qc_table(epoch_qc, PHASE1_CLASSES).set_index("class")
        assert table.loc["LEFT_HAND_IMAGERY", "n_epochs"] == 0
        assert np.isnan(table.loc["LEFT_HAND_IMAGERY", "pct_flagged"])


class TestClassAssociation:
    def _epoch_qc(self, classes, flags):
        return pd.DataFrame({"class": list(classes), "artifact_flag": list(flags)})

    def test_identical_rates_across_classes_are_not_associated(self):
        frame = self._epoch_qc(
            ["LEFT_HAND_IMAGERY", "RIGHT_HAND_IMAGERY", "REST"] * 4,
            [True, True, True, False, False, False] * 2,
        )
        result = class_association(frame, n_permutations=2000)
        assert result.p_permutation > 0.05
        assert result.verdict() == "broadly distributed"

    def test_perfect_separation_is_detected(self):
        frame = self._epoch_qc(
            ["LEFT_HAND_IMAGERY"] * 10 + ["REST"] * 10, [True] * 10 + [False] * 10
        )
        result = class_association(frame, n_permutations=2000)
        assert result.p_permutation < 0.01
        assert result.max_rate_spread_pp == pytest.approx(100.0)
        assert result.verdict() == "class-associated"

    def test_p_value_can_never_be_zero(self):
        """The observed labelling is one of the permutations, so p >= 1/(n+1)."""
        frame = self._epoch_qc(
            ["LEFT_HAND_IMAGERY"] * 10 + ["REST"] * 10, [True] * 10 + [False] * 10
        )
        result = class_association(frame, n_permutations=100)
        assert result.p_permutation >= 1 / 101

    def test_is_reproducible_for_a_fixed_seed(self):
        frame = self._epoch_qc(
            ["LEFT_HAND_IMAGERY", "REST"] * 6, [True, False, True, True] * 3
        )
        a = class_association(frame, n_permutations=500, seed=7)
        b = class_association(frame, n_permutations=500, seed=7)
        assert a.p_permutation == b.p_permutation

    def test_pairwise_contrasts_cover_every_pair_plus_task_vs_rest(self):
        frame = self._epoch_qc(
            ["LEFT_HAND_IMAGERY", "RIGHT_HAND_IMAGERY", "REST"] * 3,
            [True, True, False] * 3,
        )
        table = pairwise_class_contrasts(frame)
        assert len(table) == 4  # 3 pairs + task-vs-rest
        assert any("TASK" in name for name in table["contrast"])
        assert table["spread_pp"].is_monotonic_decreasing


class TestRegions:
    def test_every_montage_channel_is_assigned(self):
        montage = [c for members in CHANNEL_REGIONS.values() for c in members]
        assert len(montage) == 64
        assert len(set(montage)) == 64
        assert all(region_of(c) != "unassigned" for c in montage)

    def test_c3_and_c4_are_sensorimotor(self):
        assert region_of("C3") == region_of("C4")
        assert "sensorimotor" in region_of("C3")

    def test_region_table_attributes_crossings_to_the_right_region(self):
        X = make_epochs([[10.0, 10.0, 200.0, 10.0]])  # only AF7 over
        table = region_qc_table(measure_amplitudes(X, CHANNELS, THRESHOLD_V))
        by_region = table.set_index("region")["flag_events"]
        assert by_region[region_of("AF7")] == 1
        assert by_region[region_of("C3")] == 0

    def test_dominant_region_lists_only_flagged_epochs(self):
        X = make_epochs([[10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 200.0, 10.0]])
        measures = measure_amplitudes(X, CHANNELS, THRESHOLD_V)
        epoch_qc = epoch_qc_table(make_metadata(["REST", "REST"]), measures)
        table = dominant_region_per_epoch(epoch_qc, measures)
        assert len(table) == 1
        assert table.iloc[0]["epoch_id"] == "E0000001"
        assert table.iloc[0]["largest_channel"] == "AF7"


class TestRepresentatives:
    def test_picks_the_true_extremes_and_the_middle_rank(self):
        X = make_epochs([[p, 10.0, 10.0, 10.0] for p in (300.0, 100.0, 200.0)])
        measures = measure_amplitudes(X, CHANNELS, THRESHOLD_V)
        epoch_qc = epoch_qc_table(make_metadata(["REST"] * 3), measures)
        picks = pick_representatives(epoch_qc, measures)
        assert picks["clean"].peak_to_peak_uv == pytest.approx(100.0)
        assert picks["median"].peak_to_peak_uv == pytest.approx(200.0)
        assert picks["high"].peak_to_peak_uv == pytest.approx(300.0)

    def test_the_high_amplitude_banner_does_not_assert_an_artifact(self):
        """Amplitude alone cannot establish an artifact; the label must not claim it."""
        X = make_epochs([[p, 10.0, 10.0, 10.0] for p in (300.0, 100.0, 200.0)])
        measures = measure_amplitudes(X, CHANNELS, THRESHOLD_V)
        epoch_qc = epoch_qc_table(make_metadata(["REST"] * 3), measures)
        banner = pick_representatives(epoch_qc, measures)["high"].banner
        assert "POTENTIAL ARTIFACT" in banner
        assert "HIGH-AMPLITUDE" in banner


class TestLateralization:
    def test_index_is_the_log_ratio_of_mean_powers(self):
        c3 = np.array([1.0, 3.0])
        c4 = np.array([2.0, 6.0])
        assert lateralization_index(c3, c4) == pytest.approx(np.log(4.0 / 2.0))

    def test_equal_power_gives_zero(self):
        values = np.array([2.0, 4.0, 6.0])
        assert lateralization_index(values, values) == pytest.approx(0.0)

    def test_group_index_is_not_the_mean_of_per_epoch_indices(self):
        """They differ whenever power varies across epochs; the report says which."""
        c3 = np.array([1.0, 10.0])
        c4 = np.array([2.0, 5.0])
        assert lateralization_index(c3, c4) != pytest.approx(
            float(np.mean(per_epoch_lateralization(c3, c4)))
        )

    def test_mu_power_finds_a_planted_ten_hertz_rhythm(self):
        sfreq, n_times = 160.0, 640
        t = np.arange(n_times) / sfreq
        X = np.zeros((2, len(CHANNELS), n_times))
        X[:, 0, :] = 1e-5 * np.sin(2 * np.pi * 10.0 * t)   # C3: in the mu band
        X[:, 1, :] = 1e-5 * np.sin(2 * np.pi * 25.0 * t)   # C4: beta, outside it
        c3 = mu_power_per_epoch(X, CHANNELS, "C3", sfreq)
        c4 = mu_power_per_epoch(X, CHANNELS, "C4", sfreq)
        assert (c3 > 100 * c4).all()


class TestPermutationDifference:
    def test_recovers_a_planted_difference(self):
        values = np.array([1.0] * 8 + [5.0] * 8)
        group = np.array([False] * 8 + [True] * 8)
        difference, p_value = permutation_difference(values, group, n_permutations=2000)
        assert difference == pytest.approx(4.0)
        assert p_value < 0.01

    def test_no_difference_is_not_significant(self):
        # Interleaved halves of 1..20: group means are 10 and 11, a gap far
        # smaller than the spread within either group.
        values = np.arange(1.0, 21.0)
        group = np.array([True, False] * 10)
        _, p_value = permutation_difference(values, group, n_permutations=2000)
        assert p_value > 0.2

    def test_the_smallest_reachable_p_reflects_the_number_of_arrangements(self):
        """With 3 vs 3 there are only 20 splits, so p can never go below 2/20."""
        values = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
        group = np.array([False, False, False, True, True, True])
        _, p_value = permutation_difference(values, group, n_permutations=5000)
        assert p_value == pytest.approx(0.1, abs=0.02)

    def test_a_one_sided_grouping_is_undefined_rather_than_wrong(self):
        values = np.array([1.0, 2.0, 3.0])
        difference, p_value = permutation_difference(values, np.array([True] * 3))
        assert np.isnan(difference) and np.isnan(p_value)


class TestDiagnosticOnly:
    """Phase 1.6 describes; it must not modify.  These are the guardrails."""

    def test_measuring_does_not_touch_the_epochs(self):
        X = make_epochs([[200.0, 10.0, 10.0, 10.0], [10.0] * 4])
        before = X.copy()
        measures = measure_amplitudes(X, CHANNELS, THRESHOLD_V)
        epoch_qc = epoch_qc_table(make_metadata(["REST", "LEFT_HAND_IMAGERY"]), measures)
        channel_qc_table(measures)
        class_qc_table(epoch_qc, PHASE1_CLASSES)
        pick_representatives(epoch_qc, measures)
        assert np.array_equal(X, before)

    def test_no_epoch_is_dropped_however_contaminated(self):
        X = make_epochs([[9999.0] * 4] * 5)
        measures = measure_amplitudes(X, CHANNELS, THRESHOLD_V)
        epoch_qc = epoch_qc_table(make_metadata(["REST"] * 5), measures)
        assert measures.artifact_flag.all()
        assert len(epoch_qc) == 5

    def test_the_inherited_threshold_is_still_150_microvolts(self):
        """Phase 1.6 is explicitly forbidden from re-tuning this."""
        assert PreprocessingConfig().reject_log_threshold_v == pytest.approx(150e-6)

    def test_nothing_is_rejected_at_build_time(self):
        assert PreprocessingConfig().reject_peak_to_peak_v is None
