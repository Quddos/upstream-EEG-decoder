"""Phase 6B -- the Paper 1 to Phase 6A adapter.

Turns a frozen-decoder posterior into a :class:`~src.twin.states.NeuroState` the
unchanged Phase 6A twin can consume, and attaches the provenance and reliability
information that makes the state safe to hand to a later decision engine.

The one thing this adapter exists to prevent
--------------------------------------------
An uncalibrated posterior of 0.85 is **not** an 85% chance of being right. Paper 1
measured an expected calibration error of 0.169 on exactly this configuration,
which means the stated confidence was systematically wrong by about that much.
The adapter therefore:

* sets ``calibrated=False`` on every state, with no code path that can set it
  otherwise;
* stamps ``uncal`` into the ``encoder_id`` so a bare state still says so;
* names the quantity ``posterior_mass_on_predicted_class`` in the reliability
  record, never ``probability_correct``;
* refuses to emit a state whose fold provenance fails the subject-independence
  audit.

Downstream, the Phase 6A twin already discounts an uncalibrated posterior twice
over -- it inflates the observation variance and shrinks the stated magnitude
towards neutral -- so a weak or unreliable neuro channel widens the twin's
uncertainty instead of producing a confident state.

Layering
--------
``src.twin`` imports nothing from Paper 1, and a test enforces that. This module
is the one place that depends on both, which is what keeps the twin reusable with
a different encoder in a later phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.encoder.paper1 import PRIMARY_CLASSES, EpochPosterior
from src.encoder.provenance import (
    PAPER1,
    EncoderProvenanceError,
    FoldProvenance,
    verify_encoder_id,
)
from src.twin.observation import Observation
from src.twin.states import LEFT, NeuroState, RIGHT, TaskState

#: Paper 1's class labels mapped to the twin's side vocabulary.
LABEL_TO_SIDE: Mapping[str, str] = {
    PRIMARY_CLASSES[0]: LEFT,
    PRIMARY_CLASSES[1]: RIGHT,
}

#: Minimum posterior margin for an epoch to be called reliable.
#:
#: Derived, not invented: Paper 1 measured an uncalibrated expected calibration
#: error of 0.1693 for this configuration. A stated confidence within that of
#: chance is not distinguishable from chance once the miscalibration is
#: accounted for, so the confidence floor is ``0.5 + ECE`` and the equivalent
#: margin is ``2 x ECE``.
#:
#: This is a *heuristic with a stated derivation*, not a calibrated threshold.
#: Paper 1 established no calibrated operating point for the zero-calibration
#: setting, and Phase 6B does not invent one. The verdict is ordinal: it ranks
#: epochs by separation, it does not assign a probability.
DEFAULT_MIN_MARGIN: float = 2.0 * float(PAPER1["uncalibrated_ece"])  # 0.3387

#: Task identifier used for the cued motor-imagery epochs of EEGMMIDB runs
#: R04/R08/R12.
IMAGERY_TASK_ID = "eegmmidb_unilateral_imagery"


@dataclass(frozen=True)
class EpochReliability:
    """Per-epoch reliability of the *decoder output*, from measured quantities.

    Every field is computed from the posterior alone -- no label, no outcome, no
    held-out-subject information -- so this verdict is available at inference
    time for a genuinely unseen user.
    """

    usable: bool
    reason: str
    margin: float
    entropy: float
    posterior_mass_on_predicted_class: float
    calibrated: bool
    min_margin: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "usable": self.usable,
            "reason": self.reason,
            "margin": self.margin,
            "entropy": self.entropy,
            # Deliberately verbose. Naming this "confidence" invites a reader to
            # treat it as P(correct), which for an uncalibrated posterior it is
            # not.
            "posterior_mass_on_predicted_class":
                self.posterior_mass_on_predicted_class,
            "calibrated": self.calibrated,
            "min_margin": self.min_margin,
            "caveat": (
                "Uncalibrated decoder score. Not a probability of correctness, "
                "not patient confidence, not clinical confidence."
            ),
        }


@dataclass(frozen=True)
class NeuroStateRecord:
    """A ``NeuroState`` plus everything needed to trace and audit it.

    The provenance lives here rather than on ``NeuroState`` on purpose. Phase 6A
    is validated and its state model is deliberately encoder-agnostic; pushing
    EEG-specific fields into it would couple the twin to this encoder and undo
    the isolation that makes the twin reusable. Nothing is lost: the
    ``encoder_id`` stamped on the state carries the representation, model, task,
    held-out subject, calibration status and dataset fingerprint, so even a bare
    serialised ``NeuroState`` remains traceable.
    """

    neuro: NeuroState
    subject_id: str
    run_id: str
    epoch_uid: str
    session_time: float
    onset_time: float
    cued_side: str
    reliability: EpochReliability
    fold: FoldProvenance

    #: The true class of the epoch. Present because EEGMMIDB is a cued paradigm
    #: and the cue is known to the environment at run time -- it is the same
    #: information ``TaskState.target_side`` carries. It is **not** used to
    #: compute the reliability verdict.
    true_label: str

    @property
    def decoded_side(self) -> str:
        return self.neuro.predicted_class

    @property
    def decoder_agrees_with_cue(self) -> bool:
        return self.decoded_side == self.cued_side

    def to_dict(self) -> Dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "run_id": self.run_id,
            "epoch_uid": self.epoch_uid,
            "session_time": self.session_time,
            "onset_time": self.onset_time,
            "cued_side": self.cued_side,
            "true_label": self.true_label,
            "decoded_side": self.decoded_side,
            "decoder_agrees_with_cue": self.decoder_agrees_with_cue,
            "neuro": self.neuro.to_dict(),
            "reliability": self.reliability.to_dict(),
            "encoder_id": self.neuro.encoder_id,
            "fold_fingerprint": self.fold.fingerprint,
        }


class Paper1NeuroStateAdapter:
    """Frozen Paper 1 posteriors in, Phase 6A ``NeuroState`` out.

    Parameters
    ----------
    min_margin
        Posterior-margin floor for the reliability verdict. Defaults to
        :data:`DEFAULT_MIN_MARGIN`, which is derived from Paper 1's measured
        uncalibrated ECE rather than chosen.
    """

    def __init__(self, *, min_margin: float = DEFAULT_MIN_MARGIN) -> None:
        if not 0.0 <= min_margin < 1.0:
            raise ValueError("min_margin must lie in [0, 1)")
        self.min_margin = float(min_margin)

    # ------------------------------------------------------------- one epoch

    def to_neuro_state(self, posterior: EpochPosterior) -> NeuroState:
        """Build the state. Audits the fold first; never sets ``calibrated``."""
        posterior.fold.audit()
        return NeuroState(
            left_probability=posterior.left_probability,
            right_probability=posterior.right_probability,
            lateralisation=posterior.lateralisation,
            # Not a parameter. Paper 1 established calibration only with 20
            # labelled epochs from the held-out subject, which this protocol
            # forbids, so there is no honest way to set this True here.
            calibrated=False,
            encoder_id=posterior.fold.encoder_id,
        )

    def reliability(self, neuro: NeuroState) -> EpochReliability:
        """Ordinal reliability of one decoder output."""
        margin = neuro.margin
        if margin < self.min_margin:
            usable = False
            reason = (f"posterior too close to chance: margin {margin:.3f} < "
                      f"{self.min_margin:.3f} (floor = 2 x Paper 1 uncalibrated ECE)")
        else:
            usable = True
            reason = "separated posterior; still uncalibrated, so ordinal only"
        return EpochReliability(
            usable=usable,
            reason=reason,
            margin=margin,
            entropy=neuro.uncertainty,
            posterior_mass_on_predicted_class=neuro.decoder_confidence,
            calibrated=neuro.calibrated,
            min_margin=self.min_margin,
        )

    def to_record(self, posterior: EpochPosterior) -> NeuroStateRecord:
        neuro = self.to_neuro_state(posterior)
        verify_encoder_id(neuro.encoder_id, posterior.fold)
        if posterior.subject_id != posterior.fold.held_out_subject:
            raise EncoderProvenanceError(
                f"epoch {posterior.epoch_uid} belongs to {posterior.subject_id} "
                f"but was scored by the fold holding out "
                f"{posterior.fold.held_out_subject}"
            )
        cued = LABEL_TO_SIDE.get(posterior.true_label)
        if cued is None:
            raise EncoderProvenanceError(
                f"unexpected class label {posterior.true_label!r}; the Paper 1 "
                f"primary task is binary {PRIMARY_CLASSES}"
            )
        return NeuroStateRecord(
            neuro=neuro,
            subject_id=posterior.subject_id,
            run_id=posterior.run_id,
            epoch_uid=posterior.epoch_uid,
            session_time=posterior.session_time,
            onset_time=posterior.onset_time,
            cued_side=cued,
            reliability=self.reliability(neuro),
            fold=posterior.fold,
            true_label=posterior.true_label,
        )

    # ------------------------------------------------------------- a subject

    def records(self, posteriors: Iterable[EpochPosterior]) -> List[NeuroStateRecord]:
        return [self.to_record(p) for p in posteriors]

    def observations(self, posteriors: Iterable[EpochPosterior], *,
                     difficulty: float = 0.5,
                     assistance_level: float = 0.0) -> List[Observation]:
        """Twin-ready observations for one subject, in session-time order.

        ``motor`` is always ``None``. Paper 1 is an offline EEG study with no
        motor-performance data at all, and fabricating trial outcomes to fill the
        channel would mix synthetic validation into a real-data run. The twin
        handles the absence natively: latents with no supporting channel stay
        uninformed and report themselves as unusable, which is the correct
        behaviour and is demonstrated rather than asserted.

        ``TaskState.target_side`` carries the cue, which the environment knows in
        a cued paradigm. It is what lets the twin's channel-confidence estimate
        compare the decoded side against the side that was actually asked for.
        """
        records = self.records(posteriors)
        records.sort(key=lambda r: (r.session_time, r.epoch_uid))
        observations: List[Observation] = []
        for record in records:
            task = TaskState(
                task_id=IMAGERY_TASK_ID,
                difficulty=difficulty,
                assistance_level=assistance_level,
                target_side=record.cued_side,
                target_configuration={},
            )
            observations.append(Observation(
                t=record.session_time,
                task=task,
                neuro=record.neuro,
                motor=None,
                label=record.run_id,
            ))
        return observations


def subject_summary(records: Sequence[NeuroStateRecord]) -> Dict[str, object]:
    """Descriptive summary of one subject's decoder behaviour.

    Includes decoder accuracy against the cue, which is a *retrospective*
    evaluation quantity: it uses the labels and is therefore reported, never fed
    back into the adapter or the twin.
    """
    if not records:
        raise ValueError("no records to summarise")
    subject = records[0].subject_id
    if any(r.subject_id != subject for r in records):
        raise EncoderProvenanceError("records span more than one subject")

    margins = [r.neuro.margin for r in records]
    entropies = [r.neuro.uncertainty for r in records]
    confidences = [r.neuro.decoder_confidence for r in records]
    agree = [r.decoder_agrees_with_cue for r in records]
    usable = [r.reliability.usable for r in records]
    lateralisations = [r.neuro.lateralisation for r in records
                       if r.neuro.lateralisation is not None]

    def mean(values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else float("nan")

    left_cued = [r for r in records if r.cued_side == LEFT]
    right_cued = [r for r in records if r.cued_side == RIGHT]

    return {
        "subject_id": subject,
        "n_epochs": len(records),
        "runs": sorted({r.run_id for r in records}),
        "encoder_id": records[0].neuro.encoder_id,
        "calibrated": False,
        "mean_margin": mean(margins),
        "mean_entropy": mean(entropies),
        "mean_posterior_mass_on_predicted_class": mean(confidences),
        "agreement_with_cue": mean([float(a) for a in agree]),
        "usable_epoch_fraction": mean([float(u) for u in usable]),
        "n_usable_epochs": int(sum(usable)),
        "mean_lateralisation": mean(lateralisations) if lateralisations else None,
        "mean_lateralisation_left_cue": (
            mean([r.neuro.lateralisation for r in left_cued
                  if r.neuro.lateralisation is not None]) if left_cued else None),
        "mean_lateralisation_right_cue": (
            mean([r.neuro.lateralisation for r in right_cued
                  if r.neuro.lateralisation is not None]) if right_cued else None),
    }
