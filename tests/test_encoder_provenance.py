"""Phase 6B -- provenance, subject-independence and calibration honesty.

These run offline: they exercise the audit layer and the adapter's mechanics on
constructed provenance objects, so a broken guarantee is caught without needing
the PhysioNet data. The real-data behaviour is in
:mod:`tests.test_phase6b_realdata`, which is marked ``physionet``.

Every guarantee is tested twice: the clean case must pass, and a deliberately
contaminated case must be caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.encoder.adapter import (
    DEFAULT_MIN_MARGIN,
    Paper1NeuroStateAdapter,
    subject_summary,
)
from src.encoder.paper1 import PRIMARY_CLASSES, EpochPosterior
from src.encoder.provenance import (
    PAPER1,
    PAPER1_BELOW_CHANCE,
    EncoderProvenanceError,
    FoldProvenance,
    SubjectLeakageError,
    assert_subject_independence,
    parse_encoder_id,
    run_provenance,
    verify_encoder_id,
)
from src.twin.states import LEFT, RIGHT

COHORT = tuple(f"S{i:03d}" for i in range(1, 6))


def fold(held_out: str = "S001", *, training=None, **overrides) -> FoldProvenance:
    kwargs = dict(
        held_out_subject=held_out,
        training_subjects=tuple(s for s in COHORT if s != held_out)
        if training is None else tuple(training),
        n_train_epochs=180,
        n_test_epochs=45,
        model="logistic_regression",
        representation="frozen",
        channel_set="roi21",
        task="left_vs_right",
        n_features=96,
        seed=20260815,
        dataset_fingerprint="deadbeefdeadbeef",
    )
    kwargs.update(overrides)
    return FoldProvenance(**kwargs)


def posterior(p_left: float = 0.8, *, subject: str = "S001",
              label: str = PRIMARY_CLASSES[0], f: FoldProvenance | None = None,
              t: float = 0.0, uid: str = "E0000001") -> EpochPosterior:
    return EpochPosterior(
        epoch_uid=uid, subject_id=subject, run_id="R04", run=4,
        onset_time=t, session_time=t, true_label=label,
        left_probability=p_left, right_probability=1.0 - p_left,
        lateralisation=-0.2, fold=f or fold(subject),
    )


# ================================================== subject independence audit


class TestSubjectIndependence:
    def test_a_clean_fold_passes(self):
        fold("S001").audit()

    def test_control_a_leaky_fold_is_rejected_at_construction(self):
        """The leak that matters: a fold trained on its own held-out subject."""
        with pytest.raises(SubjectLeakageError, match="S001"):
            fold("S001", training=COHORT)

    def test_control_a_wrong_cohort_is_caught_by_the_run_level_audit(self):
        """The constructor catches self-training; the run audit catches the rest.

        A fold can be individually valid -- it does not train on its own held-out
        subject -- and still be wrong, by having quietly dropped another subject.
        That fold would report a legitimate-looking accuracy computed against a
        different training population from every other fold.
        """
        clean = [fold(s) for s in COHORT]
        dropped = fold("S002", training=[s for s in COHORT
                                         if s not in ("S002", "S003")])
        assert dropped.held_out_subject not in dropped.training_subjects
        with pytest.raises(EncoderProvenanceError, match="wrong cohort"):
            assert_subject_independence([f for f in clean if f.held_out_subject != "S002"]
                                        + [dropped])

    def test_a_full_clean_run_passes(self):
        summary = assert_subject_independence([fold(s) for s in COHORT])
        assert summary["n_folds"] == len(COHORT)
        assert summary["cohort_size"] == len(COHORT)
        assert summary["leakage_detected"] is False
        assert summary["all_uncalibrated"] is True

    def test_control_a_duplicated_hold_out_is_rejected(self):
        folds = [fold(s) for s in COHORT] + [fold("S001")]
        with pytest.raises(EncoderProvenanceError, match="more than once"):
            assert_subject_independence(folds)

    def test_control_a_missing_training_subject_is_rejected(self):
        """A fold trained on a subset is not the protocol Paper 1 reported."""
        folds = [fold(s) for s in COHORT]
        folds[0] = fold("S001", training=("S002", "S003"))
        with pytest.raises(EncoderProvenanceError, match="missing"):
            assert_subject_independence(folds)

    def test_an_empty_run_is_rejected(self):
        with pytest.raises(EncoderProvenanceError):
            assert_subject_independence([])

    def test_a_fold_with_no_training_subjects_is_rejected(self):
        with pytest.raises(EncoderProvenanceError, match="no training subjects"):
            fold("S001", training=())


# ============================================================ calibration honesty


class TestCalibrationHonesty:
    def test_a_fold_cannot_declare_itself_calibrated(self):
        """Structural: there is no honest way to set this True in Phase 6B."""
        with pytest.raises(EncoderProvenanceError, match="uncalibrated"):
            fold("S001", calibrated=True)

    def test_the_adapter_always_emits_uncalibrated_states(self):
        state = Paper1NeuroStateAdapter().to_neuro_state(posterior())
        assert state.calibrated is False

    def test_the_adapter_has_no_calibration_parameter(self):
        """Control: a calibrated flag must not be reachable through the API."""
        import inspect

        signature = inspect.signature(Paper1NeuroStateAdapter.__init__)
        assert "calibrated" not in signature.parameters
        signature = inspect.signature(Paper1NeuroStateAdapter.to_neuro_state)
        assert "calibrated" not in signature.parameters

    def test_the_encoder_id_says_uncalibrated(self):
        """Even a bare serialised state must declare it."""
        assert "/uncal/" in fold().encoder_id

    def test_the_reliability_record_never_calls_it_probability_correct(self):
        adapter = Paper1NeuroStateAdapter()
        payload = adapter.reliability(adapter.to_neuro_state(posterior())).to_dict()
        assert "posterior_mass_on_predicted_class" in payload
        assert not any("probability_correct" in key or "p_correct" in key
                       for key in payload)
        assert "Not a probability of correctness" in payload["caveat"]

    def test_the_run_header_carries_the_calibration_caveat(self):
        header = run_provenance([fold(s) for s in COHORT])
        assert header["calibration"]["calibrated"] is False
        assert "forbids" in header["calibration"]["why"]
        assert "NOT a probability of correctness" in header["calibration"]["caveat"]

    def test_the_reliability_floor_is_derived_from_paper1_ece(self):
        """The threshold is derived, not chosen; it must track the source value."""
        assert DEFAULT_MIN_MARGIN == pytest.approx(
            2.0 * float(PAPER1["uncalibrated_ece"]))


# ================================================================= encoder id


class TestEncoderId:
    def test_round_trips_through_parsing(self):
        f = fold("S021")
        parsed = parse_encoder_id(f.encoder_id)
        assert parsed["held_out_subject"] == "S021"
        assert parsed["model"] == "logistic_regression"
        assert parsed["channel_set"] == "roi21"
        assert parsed["representation"] == "frozen"
        assert parsed["calibration"] == "uncal"
        assert parsed["dataset_fingerprint"] == f.dataset_fingerprint

    def test_control_a_foreign_encoder_id_is_rejected(self):
        """Incorrect encoder provenance must not be silently accepted."""
        with pytest.raises(EncoderProvenanceError, match="not a Paper 1 encoder id"):
            parse_encoder_id("simulator")
        with pytest.raises(EncoderProvenanceError, match="not a Paper 1 encoder id"):
            parse_encoder_id("some-other-model/v2")

    def test_control_a_malformed_encoder_id_is_rejected(self):
        with pytest.raises(EncoderProvenanceError, match="malformed"):
            parse_encoder_id("paper1/frozen/roi21/lr/task/S001/uncal/cfg=x")

    def test_control_a_mismatched_encoder_id_is_caught(self):
        with pytest.raises(EncoderProvenanceError, match="encoder_id mismatch"):
            verify_encoder_id(fold("S002").encoder_id, fold("S001"))

    def test_states_from_the_wrong_fold_are_rejected(self):
        """Accidental contamination: one subject's epoch scored by another's fold."""
        wrong = posterior(subject="S002", f=fold("S001"))
        with pytest.raises(EncoderProvenanceError, match="belongs to S002"):
            Paper1NeuroStateAdapter().to_record(wrong)


# ==================================================================== adapter


class TestAdapterMechanics:
    def test_posterior_becomes_a_neuro_state_with_derived_quantities(self):
        state = Paper1NeuroStateAdapter().to_neuro_state(posterior(0.8))
        assert state.left_probability == pytest.approx(0.8)
        assert state.right_probability == pytest.approx(0.2)
        assert state.decoder_confidence == pytest.approx(0.8)
        assert state.margin == pytest.approx(0.6)
        assert 0.0 < state.uncertainty < 1.0
        assert state.predicted_class == LEFT

    def test_a_chance_posterior_has_maximum_entropy_and_zero_margin(self):
        state = Paper1NeuroStateAdapter().to_neuro_state(posterior(0.5))
        assert state.uncertainty == pytest.approx(1.0)
        assert state.margin == pytest.approx(0.0)

    def test_a_near_chance_epoch_is_not_usable(self):
        adapter = Paper1NeuroStateAdapter()
        weak = adapter.reliability(adapter.to_neuro_state(posterior(0.55)))
        assert not weak.usable and "too close to chance" in weak.reason

    def test_a_separated_epoch_is_usable_but_still_flagged_uncalibrated(self):
        adapter = Paper1NeuroStateAdapter()
        strong = adapter.reliability(adapter.to_neuro_state(posterior(0.95)))
        assert strong.usable
        assert strong.calibrated is False
        assert "ordinal" in strong.reason

    def test_the_reliability_verdict_uses_no_label_or_outcome(self):
        """It must be computable for a genuinely unseen user."""
        adapter = Paper1NeuroStateAdapter()
        left_cued = posterior(0.8, label=PRIMARY_CLASSES[0])
        right_cued = posterior(0.8, label=PRIMARY_CLASSES[1], subject="S001")
        a = adapter.reliability(adapter.to_neuro_state(left_cued))
        b = adapter.reliability(adapter.to_neuro_state(right_cued))
        assert a == b, "the verdict changed when only the true label changed"

    def test_record_carries_the_required_provenance(self):
        record = Paper1NeuroStateAdapter().to_record(posterior())
        payload = record.to_dict()
        for key in ("subject_id", "run_id", "epoch_uid", "session_time",
                    "onset_time", "cued_side", "encoder_id", "reliability"):
            assert key in payload, key
        for key in ("left_probability", "right_probability", "margin",
                    "uncertainty", "decoder_confidence", "lateralisation",
                    "calibrated", "encoder_id"):
            assert key in payload["neuro"], key

    def test_records_are_json_serialisable(self):
        record = Paper1NeuroStateAdapter().to_record(posterior())
        assert json.loads(json.dumps(record.to_dict()))["subject_id"] == "S001"

    def test_observations_carry_the_cue_and_never_fabricate_motor_data(self):
        """Paper 1 has no motor performance; inventing it would mix synthetic in."""
        f = fold("S001")
        posteriors = [posterior(0.8, f=f, t=float(i), uid=f"E{i:07d}")
                      for i in range(5)]
        observations = Paper1NeuroStateAdapter().observations(posteriors)
        assert len(observations) == 5
        assert all(o.motor is None for o in observations)
        assert all(o.neuro is not None for o in observations)
        assert all(o.task.target_side in (LEFT, RIGHT) for o in observations)

    def test_observations_are_ordered_by_session_time(self):
        f = fold("S001")
        posteriors = [posterior(0.8, f=f, t=t, uid=f"E{int(t):07d}")
                      for t in (30.0, 10.0, 20.0)]
        times = [o.t for o in Paper1NeuroStateAdapter().observations(posteriors)]
        assert times == sorted(times)

    def test_the_cued_side_follows_the_true_label(self):
        adapter = Paper1NeuroStateAdapter()
        left = adapter.to_record(posterior(label=PRIMARY_CLASSES[0]))
        right = adapter.to_record(posterior(label=PRIMARY_CLASSES[1]))
        assert left.cued_side == LEFT and right.cued_side == RIGHT

    def test_an_unexpected_class_label_is_rejected(self):
        with pytest.raises(EncoderProvenanceError, match="binary"):
            Paper1NeuroStateAdapter().to_record(posterior(label="REST"))

    def test_summary_reports_agreement_without_feeding_it_back(self):
        f = fold("S001")
        records = Paper1NeuroStateAdapter().records([
            posterior(0.9, f=f, label=PRIMARY_CLASSES[0], t=0.0, uid="E0"),
            posterior(0.1, f=f, label=PRIMARY_CLASSES[1], t=1.0, uid="E1"),
            posterior(0.9, f=f, label=PRIMARY_CLASSES[1], t=2.0, uid="E2"),
        ])
        summary = subject_summary(records)
        assert summary["n_epochs"] == 3
        assert summary["agreement_with_cue"] == pytest.approx(2 / 3)
        assert summary["calibrated"] is False

    def test_summary_rejects_mixed_subjects(self):
        adapter = Paper1NeuroStateAdapter()
        records = [adapter.to_record(posterior(subject="S001", f=fold("S001"))),
                   adapter.to_record(posterior(subject="S002", f=fold("S002")))]
        with pytest.raises(EncoderProvenanceError, match="more than one subject"):
            subject_summary(records)


# ============================================================ inherited numbers


class TestPaper1ReferenceValues:
    """The numbers Phase 6B cites must still match the frozen artifacts."""

    def test_mean_balanced_accuracy_matches_the_artifact(self):
        path = Path("data/metadata/phase4b/phase4b_summary.json")
        if not path.exists():
            pytest.skip("Paper 1 artifact not present in this checkout")
        summary = json.loads(path.read_text())
        endpoint = summary["endpoints"]["left_vs_right/logistic_regression/frozen"]
        assert endpoint["mean"] == pytest.approx(PAPER1["mean_balanced_accuracy"])
        assert endpoint["below_chance"] == PAPER1["below_chance_subjects"]
        assert endpoint["n"] == PAPER1["n_subjects"]

    def test_uncalibrated_ece_matches_the_artifact(self):
        path = Path("data/metadata/phase4a/phase4a_summary.json")
        if not path.exists():
            pytest.skip("Paper 1 artifact not present in this checkout")
        summary = json.loads(path.read_text())
        assert summary["calibration"]["none"]["ece"] == pytest.approx(
            PAPER1["uncalibrated_ece"])

    def test_the_below_chance_group_matches_the_frozen_definition(self):
        path = Path("data/metadata/phase5/phase5_outcome_definition.json")
        if not path.exists():
            pytest.skip("Paper 1 artifact not present in this checkout")
        definition = json.loads(path.read_text())
        assert set(definition["below_subjects"]) == set(PAPER1_BELOW_CHANCE)
        assert definition["n_below"] == len(PAPER1_BELOW_CHANCE)
