"""Phase 6B -- real-data behaviour of the frozen encoder driving the twin.

Marked ``physionet``: these need the processed EEGMMIDB epochs and are deselected
by default. When the data is absent they **skip**, never pass silently -- the
project's standing rule.

The question these answer is the Phase 6B gate: can the validated twin operate
when its neuro input comes from the actual frozen cross-subject decoder, and does
it stay uncertainty-aware when that input is weak? They are not asking whether the
decoder is any good; Paper 1 already answered that and it is frozen.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.encoder import (
    PAPER1,
    PAPER1_BELOW_CHANCE,
    Paper1NeuroStateAdapter,
    encode_cohort,
    subject_summary,
)
from src.twin.interfaces import DecisionEngineView
from src.twin.serialization import read_states, write_states
from src.twin.states import LatentState
from src.twin.twin import DigitalTwin, TwinConfig

pytestmark = pytest.mark.physionet

#: Representative cases, fixed from the frozen Paper 1 outcome.
STRONG = "S007"        # best in cohort, 0.932
NEAR_CHANCE = "S037"   # exactly 0.500; sharp posteriors, chance agreement
BELOW = "S003"         # worst in cohort, 0.398


@pytest.fixture(scope="module")
def cohort():
    try:
        return encode_cohort()
    except FileNotFoundError as error:  # pragma: no cover - environment dependent
        pytest.skip(f"PHYSIONET DATA UNAVAILABLE: {error}")


@pytest.fixture(scope="module")
def adapter():
    return Paper1NeuroStateAdapter()


def _twin(cohort, adapter, subject: str) -> DigitalTwin:
    twin = DigitalTwin(TwinConfig(patient_id=subject))
    twin.run(adapter.observations(cohort.for_subject(subject)))
    return twin


# ===================================================== Paper 1 is re-executed


class TestFrozenPaper1IsReproduced:
    def test_the_re_execution_matches_the_published_per_subject_values(self, cohort):
        """The only evidence that this is the frozen pipeline and not a new one."""
        report = cohort.reproduction
        assert report["verified"] is True
        assert report["n_subjects"] == PAPER1["n_subjects"]
        assert report["max_abs_difference"] < 1e-9

    def test_the_cohort_mean_is_paper1s_headline(self, cohort):
        assert cohort.reproduction["reproduced_mean"] == pytest.approx(
            PAPER1["mean_balanced_accuracy"], abs=1e-9)

    def test_the_dataset_is_the_permitted_subset(self, cohort):
        assert len(cohort.subjects) == 40
        assert len(cohort.posteriors) == PAPER1["n_binary_epochs"]
        runs = {p.run_id for p in cohort.posteriors}
        assert runs == {"R04", "R08", "R12"}

    def test_every_fold_used_the_frozen_primary_configuration(self, cohort):
        for fold in cohort.folds.values():
            assert fold.model == "logistic_regression"
            assert fold.channel_set == "roi21"
            assert fold.representation == "frozen"
            assert fold.task == "left_vs_right"
            assert fold.n_features == PAPER1["n_features"]
            assert fold.calibrated is False


# ============================================== subject independence, for real


class TestSubjectIndependenceOnRealFolds:
    def test_no_fold_trained_on_its_own_held_out_subject(self, cohort):
        for subject, fold in cohort.folds.items():
            assert subject not in fold.training_subjects

    def test_every_fold_trained_on_all_other_subjects(self, cohort):
        cohort_ids = set(cohort.subjects)
        for subject, fold in cohort.folds.items():
            assert set(fold.training_subjects) == cohort_ids - {subject}
            assert len(fold.training_subjects) == 39

    def test_the_run_level_audit_passes(self, cohort):
        assert cohort.audit["leakage_detected"] is False
        assert cohort.audit["n_folds"] == 40

    def test_every_epoch_was_scored_by_its_own_hold_out_fold(self, cohort):
        for posterior in cohort.posteriors:
            assert posterior.fold.held_out_subject == posterior.subject_id
            assert f"holdout={posterior.subject_id}" in posterior.fold.encoder_id


# ================================================ the three required cases


class TestRepresentativeSubjects:
    def test_the_strong_subject_accumulates_high_channel_confidence(
            self, cohort, adapter):
        twin = _twin(cohort, adapter, STRONG)
        view = DecisionEngineView(twin)
        summary = subject_summary(adapter.records(cohort.for_subject(STRONG)))
        assert summary["agreement_with_cue"] > 0.8
        assert twin.state.latent.confidence_estimate.value > 0.6
        assert view.latent_is_usable("confidence_estimate")

    def test_the_near_chance_subject_does_not_accumulate_confidence(
            self, cohort, adapter):
        """S037: sharp posteriors, chance agreement -- confidently wrong.

        The case that matters most. Its mean posterior margin is the highest in
        the cohort, so anything keying on per-epoch confidence would call this
        subject excellent. The twin must not.
        """
        twin = _twin(cohort, adapter, NEAR_CHANCE)
        summary = subject_summary(adapter.records(cohort.for_subject(NEAR_CHANCE)))
        assert 0.45 < summary["agreement_with_cue"] < 0.60, "expected chance agreement"
        assert summary["mean_margin"] > 0.6, "expected sharply separated posteriors"
        assert twin.state.latent.confidence_estimate.value == pytest.approx(0.5, abs=0.08)

    def test_a_below_chance_subject_accumulates_low_confidence(self, cohort, adapter):
        twin = _twin(cohort, adapter, BELOW)
        assert BELOW in PAPER1_BELOW_CHANCE
        assert twin.state.latent.confidence_estimate.value < 0.45

    def test_channel_confidence_orders_the_three_cases_correctly(
            self, cohort, adapter):
        values = {s: _twin(cohort, adapter, s).state.latent.confidence_estimate.value
                  for s in (STRONG, NEAR_CHANCE, BELOW)}
        assert values[STRONG] > values[NEAR_CHANCE] > values[BELOW]


# ================================ weak evidence must not become confident state


class TestWeakEvidenceStaysWeak:
    """The Phase 6B design requirement, stated as tests."""

    def test_no_subject_gets_a_usable_motor_state_estimate(self, cohort, adapter):
        """Paper 1 has no motor data, so the twin must not invent a motor state."""
        for subject in (STRONG, NEAR_CHANCE, BELOW):
            twin = _twin(cohort, adapter, subject)
            view = DecisionEngineView(twin)
            assert not view.latent_is_usable("motor_state_estimate")
            assert twin.state.latent.motor_state_estimate.n_updates == 0

    def test_the_near_chance_subject_gets_no_usable_attention_estimate(
            self, cohort, adapter):
        """A confidently wrong channel must not produce a confident person-state."""
        twin = _twin(cohort, adapter, NEAR_CHANCE)
        view = DecisionEngineView(twin)
        readout = view.latent("attention_estimate")
        assert not readout.usable, (
            "a decoder at chance produced a usable attention estimate; weak EEG "
            "evidence has been converted into a confident statement about the person"
        )

    def test_no_below_chance_subject_gets_a_usable_attention_estimate(
            self, cohort, adapter):
        for subject in PAPER1_BELOW_CHANCE:
            twin = _twin(cohort, adapter, subject)
            view = DecisionEngineView(twin)
            assert not view.latent_is_usable("attention_estimate"), subject

    def test_uncertainty_is_wider_for_weak_channels(self, cohort, adapter):
        strong = _twin(cohort, adapter, STRONG).state.latent
        weak = _twin(cohort, adapter, NEAR_CHANCE).state.latent
        assert weak.attention_estimate.sd > strong.attention_estimate.sd

    def test_every_state_is_flagged_uncalibrated(self, cohort, adapter):
        records = adapter.records(cohort.for_subject(STRONG))
        assert all(r.neuro.calibrated is False for r in records)
        assert all("/uncal/" in r.neuro.encoder_id for r in records)


# ================================================ per-epoch confidence is weak


class TestPerEpochConfidenceIsNotReliability:
    def test_posterior_margin_does_not_predict_decoder_correctness(
            self, cohort, adapter):
        """The empirical reason the twin cannot rely on per-epoch confidence.

        If this ever becomes a strong correlation, the architecture could be
        simplified -- and the Phase 6B report's central caveat would need
        rewriting. It is asserted so that such a change cannot pass unnoticed.
        """
        margins, agreements = [], []
        for subject in cohort.subjects:
            summary = subject_summary(adapter.records(cohort.for_subject(subject)))
            margins.append(summary["mean_margin"])
            agreements.append(summary["agreement_with_cue"])
        r = float(np.corrcoef(margins, agreements)[0, 1])
        assert abs(r) < 0.35, (
            f"per-subject margin now correlates with correctness at r={r:.3f}; "
            "the report's reliability caveat needs revisiting"
        )

    def test_accumulated_channel_confidence_reflects_its_own_agreement_input(
            self, cohort, adapter):
        """Mechanism consistency check -- NOT independent validation.

        Cue agreement is a direct input to the channel-confidence update (weight
        0.45 of 1.00), so a high correlation here shows the filter faithfully
        accumulates the signal it is given, and nothing more. Removing the
        agreement term drops this correlation to r = +0.061 (n.s.); see
        ``docs/PHASE6B_SCIENTIFIC_AUDIT.md`` section 1.
        """
        trust, agreements = [], []
        for subject in cohort.subjects:
            twin = _twin(cohort, adapter, subject)
            trust.append(twin.state.latent.confidence_estimate.value)
            agreements.append(subject_summary(
                adapter.records(cohort.for_subject(subject)))["agreement_with_cue"])
        r = float(np.corrcoef(trust, agreements)[0, 1])
        assert r > 0.7, (
            f"the channel-confidence state no longer reflects its own agreement "
            f"input (r={r:.3f}); the accumulation mechanism has changed")


# ================================================================ mechanics


class TestRepeatedRunsAndReplay:
    def test_all_three_runs_of_a_subject_are_present_and_ordered(self, cohort, adapter):
        records = adapter.records(cohort.for_subject(STRONG))
        assert {r.run_id for r in records} == {"R04", "R08", "R12"}
        observations = adapter.observations(cohort.for_subject(STRONG))
        times = [o.t for o in observations]
        assert times == sorted(times)
        assert len(observations) == 45

    def test_the_session_clock_is_strictly_increasing_across_runs(self, cohort, adapter):
        observations = adapter.observations(cohort.for_subject(STRONG))
        times = [o.t for o in observations]
        assert all(b > a for a, b in zip(times, times[1:]))

    def test_deterministic_replay_end_to_end(self, cohort, adapter):
        """Same epochs, same frozen decoder, same config -> same trajectory hash."""
        first = _twin(cohort, adapter, STRONG)
        rebuilt = encode_cohort()
        second = DigitalTwin(TwinConfig(patient_id=STRONG))
        second.run(Paper1NeuroStateAdapter().observations(rebuilt.for_subject(STRONG)))
        assert first.trajectory_hash() == second.trajectory_hash()

    def test_the_neuro_state_sequence_is_identical_across_runs_of_the_encoder(
            self, cohort, adapter):
        a = [r.neuro.to_dict() for r in adapter.records(cohort.for_subject(BELOW))]
        b = [r.neuro.to_dict()
             for r in Paper1NeuroStateAdapter().records(
                 encode_cohort().for_subject(BELOW))]
        assert a == b

    def test_json_round_trip_preserves_the_twin_states(self, cohort, adapter, tmp_path):
        twin = _twin(cohort, adapter, BELOW)
        states = list(twin.history)
        path = tmp_path / f"{BELOW}.states.jsonl"
        write_states(path, states, twin.config)
        _, restored = read_states(path)
        assert restored == states

    def test_records_survive_json(self, cohort, adapter):
        record = adapter.records(cohort.for_subject(STRONG))[0]
        restored = json.loads(json.dumps(record.to_dict()))
        assert restored["subject_id"] == STRONG
        assert restored["neuro"]["calibrated"] is False
        assert restored["reliability"]["calibrated"] is False
