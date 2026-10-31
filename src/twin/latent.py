"""The latent state-transition and update mechanism.

This module is where the twin infers what it cannot measure. Everything in it is
a **modelling assumption**, and the assumptions are collected in one frozen,
hashable :class:`LatentModelConfig` so that a state trajectory can be traced back
to the exact coefficients that produced it -- the same discipline the EEG
pipeline uses for preprocessing.

How the update works
--------------------
Each latent is a scalar Gaussian belief updated by a two-step filter:

1. **Predict.** The mean drifts by a model-specific amount proportional to
   elapsed time and the variance grows by ``process_variance_rate * dt``. This
   step runs whether or not anything was observed, so a latent that stops being
   informed decays towards its prior instead of staying confidently wrong.

2. **Update.** Measured variables are mapped to a scalar *pseudo-observation*
   with an attached observation variance, and the belief is combined with it in
   the usual scalar Kalman form. A degraded or missing measured channel raises
   the observation variance, which shrinks the gain, which is exactly the
   behaviour that "we saw less, so we learned less" should produce.

Why a filter and not a heuristic
--------------------------------
A heuristic would give a number. The filter also gives a variance, and the
variance is the part that carries the scientific constraint: it is what stops a
decision engine treating ``fatigue_estimate = 0.7`` after one noisy trial the
same way it treats ``0.7`` after forty consistent ones.

What this is not
----------------
None of these mappings has been validated against human data, none is fitted to
anything, and none should be read as a physiological model. They are plausible
monotone relationships chosen so that the seven Phase 6A scenarios produce
distinguishable trajectories. Their only current claim to correctness is that
they are deterministic, documented and testable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Mapping, Optional, Tuple

from src.twin.observation import Observation
from src.twin.variability import (
    VarianceEstimate,
    VariabilityConfig,
    observation_variance,
)
from src.twin.normalisation import (
    MotorNormalisationConfig,
    fatigue_error_ref,
    motor_error_ref,
)
from src.twin.states import LEFT, LatentState, RIGHT
from src.twin.uncertainty import Estimate, PRIOR_VARIANCE

# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class LatentModelConfig:
    """Every coefficient the latent estimator uses.

    Each field is an assumption. The docstring of each group says what the
    assumption is, so that ``docs/PHASE6A_ASSUMPTIONS.md`` can be generated from
    here rather than drifting away from it.
    """

    # --- reference scales -------------------------------------------------
    #: Reaction time at which the attention pseudo-observation is neutral (0.5).
    #: Assumption: slower responses indicate lower engagement, monotonically.
    reaction_time_ref_s: float = 1.2
    #: Completion time used to normalise deviations from the session baseline.
    completion_time_ref_s: float = 6.0
    #: Movement error used to normalise deviations from the session baseline.
    movement_error_ref: float = 0.25
    #: Time constant of the time-on-task load term, in seconds.
    #: Assumption: accumulated load saturates rather than growing without bound.
    fatigue_time_constant_s: float = 900.0

    # --- attention --------------------------------------------------------
    #: Weight on decoder confidence in the attention pseudo-observation.
    attention_w_confidence: float = 0.5
    #: Weight on the reaction-time score in the attention pseudo-observation.
    attention_w_reaction: float = 0.5
    #: Base observation variance for attention when all channels are present.
    attention_observation_variance: float = 0.02
    #: Minimum channel confidence before a decoder posterior is allowed to
    #: inform the attention estimate at all.
    #:
    #: Added in Phase 6B after real EEG exposed a defect. Subject S037's decoder
    #: produced sharply separated posteriors (mean margin 0.847, every epoch past
    #: the reliability floor) while agreeing with the cue at chance (0.511) -- a
    #: confidently wrong channel. Because the attention term reads posterior
    #: sharpness as engagement, the twin reported attention 0.86 +- 0.11 and
    #: marked it usable: weak EEG evidence converted into a confident statement
    #: about the person, which is exactly what this architecture exists to
    #: prevent.
    #:
    #: The principle, not a patch: a pseudo-observation derived from a channel
    #: the twin does not trust must not inform a latent about the person. Below
    #: this threshold the neuro term is withheld entirely, so attention stays
    #: uninformed and reports itself unusable rather than becoming confidently
    #: wrong. 0.55 is "meaningfully better than neutral"; at 0.5 the twin has no
    #: opinion about the channel.
    attention_min_channel_trust: float = 0.55
    #: Process variance per second for attention. Larger than the others because
    #: attention is assumed to be the fastest-moving latent.
    attention_process_rate: float = 2.0e-4

    # --- fatigue ----------------------------------------------------------
    #: Weight on accumulated time-on-task.
    fatigue_w_load: float = 0.45
    #: Weight on within-session performance degradation.
    fatigue_w_degradation: float = 0.55
    fatigue_observation_variance: float = 0.03
    #: Assumption: fatigue does not spontaneously fall during a session, so the
    #: predict step applies a small positive drift per second.
    fatigue_drift_per_s: float = 2.0e-5
    fatigue_process_rate: float = 8.0e-5

    # --- recovery ---------------------------------------------------------
    #: Weight on difficulty when crediting a success. Assumption: succeeding at a
    #: harder task is stronger evidence of expressed capability.
    recovery_difficulty_gain: float = 0.5
    #: How much assistance discounts a success. At assistance 1.0 a success
    #: carries no evidence about the patient at all.
    recovery_assistance_discount: float = 1.0
    recovery_observation_variance: float = 0.05
    recovery_process_rate: float = 3.0e-5

    # --- motor state ------------------------------------------------------
    #: Weight on task success in the motor-state pseudo-observation.
    motor_w_success: float = 0.4
    #: Weight on normalised accuracy.
    motor_w_accuracy: float = 0.3
    #: Weight on smoothness.
    motor_w_smoothness: float = 0.3
    motor_observation_variance: float = 0.025
    motor_process_rate: float = 6.0e-5

    # --- twin self-confidence --------------------------------------------
    #: Weight on agreement between the decoded class and the cued side.
    confidence_w_agreement: float = 0.45
    #: Weight on tick-to-tick stability of the decoder posterior.
    confidence_w_stability: float = 0.25
    #: Weight on the decoder's own confidence.
    confidence_w_decoder: float = 0.30
    confidence_observation_variance: float = 0.04
    confidence_process_rate: float = 1.2e-4
    #: Multiplier applied to every neuro-derived observation variance when the
    #: posterior is not calibrated. Paper 1 found calibration improves
    #: reliability without improving accuracy, so an uncalibrated posterior is a
    #: worse guide to trust at identical accuracy.
    uncalibrated_variance_penalty: float = 2.5
    #: Fraction by which an *uncalibrated* confidence magnitude is shrunk towards
    #: neutral before it contributes to the twin's self-confidence.
    #:
    #: Inflating the variance alone is not enough. Variance controls how fast the
    #: belief moves; it does not stop an overconfident channel from dragging the
    #: belief to the wrong place, only from doing it quickly. Paper 1 measured an
    #: expected calibration error of 0.169 on the uncalibrated model, i.e. its
    #: stated confidence was systematically wrong, so the magnitude itself is
    #: discounted and not merely distrusted.
    uncalibrated_confidence_discount: float = 0.5

    # --- baseline ---------------------------------------------------------
    #: Number of early trials that define the within-subject performance
    #: baseline. Deliberately within-subject: Paper 1 showed that an absolute
    #: cross-subject threshold on an amplitude-like quantity fails badly.
    baseline_trials: int = 5

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in asdict(self).items()}

    @property
    def fingerprint(self) -> str:
        """Stable 16-hex hash of every coefficient, mirroring ``DatasetConfig``.

        A state trajectory is only reproducible with respect to a specific set
        of assumptions, so the fingerprint travels with every artifact.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


DEFAULT_LATENT_CONFIG = LatentModelConfig()


# --------------------------------------------------------------------- memory


@dataclass(frozen=True)
class EstimatorMemory:
    """Everything the estimator remembers between ticks, outside the beliefs.

    Kept separate from :class:`~src.twin.states.LatentState` and immutable for
    one reason: replay. The twin's entire future depends on ``(latent, memory)``
    plus the observation sequence, so snapshotting those two objects is enough to
    resume or reproduce a session exactly.
    """

    #: Running means over the first ``baseline_trials`` trials, used as the
    #: subject's own reference. ``None`` until the baseline is complete.
    baseline_movement_error: Optional[float] = None
    baseline_completion_time: Optional[float] = None
    baseline_smoothness: Optional[float] = None

    #: Partial sums while the baseline is still being accumulated.
    baseline_n: int = 0
    baseline_sum_error: float = 0.0
    baseline_sum_time: float = 0.0
    baseline_sum_smoothness: float = 0.0

    #: Previous *signed* class margin, ``p_left - p_right`` in ``[-1, 1]``, and
    #: the cued side it was produced under.
    #:
    #: Signed, not the confidence magnitude: a decoder that goes from
    #: confidently-LEFT to confidently-RIGHT is maximally unstable, and a
    #: magnitude-only comparison scores that as perfectly stable.
    #:
    #: The side is carried with it because the two are only comparable under the
    #: same cue. Between trials the cue alternates, so a sign change is the
    #: expected behaviour of a *working* decoder; measured on the seven
    #: scenarios, ignoring the cue makes a clean decoder look 23% unstable purely
    #: from target alternation. Stability is therefore undefined -- and the term
    #: dropped -- whenever the cue changed.
    previous_signed_margin: Optional[float] = None
    previous_target_side: Optional[str] = None

    #: Consecutive trials with ``task_success == False``. Exposed to the
    #: decision-engine view because a failure streak is an interface concern,
    #: not something a downstream component should have to re-derive.
    consecutive_failures: int = 0

    #: Phase 6H. Recent motor pseudo-observations, oldest first, bounded by the
    #: variability window. Empty on the frozen path, so a default memory is
    #: identical to Phase 6A's in both value and serialised form.
    #:
    #: Held here rather than read back out of ``StateHistory`` for the reason
    #: this class exists: the twin's future must depend on ``(latent, memory)``
    #: and the observation sequence alone, so anything the estimator needs to
    #: remember has to be *in* the memory or replay stops being exact.
    motor_z_window: Tuple[float, ...] = ()

    @property
    def has_baseline(self) -> bool:
        return self.baseline_movement_error is not None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EstimatorMemory":
        fields = {k: payload[k] for k in asdict(cls()).keys() if k in payload}
        # JSON has no tuples. Without this the window round-trips as a list and
        # the memory compares unequal to itself after a save/load cycle.
        if "motor_z_window" in fields and fields["motor_z_window"] is not None:
            fields["motor_z_window"] = tuple(
                float(v) for v in fields["motor_z_window"])  # type: ignore[union-attr]
        return cls(**fields)  # type: ignore[arg-type]


# ------------------------------------------------------------------ utilities


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _saturating(elapsed_s: float, tau: float) -> float:
    """``1 - exp(-t/tau)``, the standard bounded accumulator."""
    if tau <= 0:
        raise ValueError("time constant must be positive")
    return 1.0 - math.exp(-max(0.0, elapsed_s) / tau)


@dataclass(frozen=True)
class PseudoObservation:
    """A scalar derived from measured variables, with how much to trust it.

    Recorded per tick alongside the state so that a trajectory can be audited:
    if a latent moved, this says which measured quantities moved it.
    """

    value: float
    variance: float
    sources: Tuple[str, ...]

    #: Phase 6H. How ``variance`` was arrived at, when it was estimated rather
    #: than read from the configuration. ``None`` on the frozen path, which is
    #: what keeps the default record byte-identical to Phase 6A.
    variance_detail: Optional["VarianceEstimate"] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "value": self.value, "variance": self.variance,
            "sources": list(self.sources)}
        if self.variance_detail is not None:
            payload["variance_detail"] = self.variance_detail.to_dict()
        return payload


# ------------------------------------------------------------------ estimator


class LatentEstimator:
    """Pure, deterministic latent update. Holds no mutable state of its own.

    Every method takes the current belief and memory and returns new ones. That
    is what lets :class:`~src.twin.twin.DigitalTwin` be replayed exactly and what
    lets the estimator be tested without constructing a twin at all.
    """

    def __init__(self, config: LatentModelConfig = DEFAULT_LATENT_CONFIG,
                 variability: Optional[VariabilityConfig] = None,
                 normalisation: Optional[MotorNormalisationConfig] = None
                 ) -> None:
        self.config = config
        #: Phase 6H, opt-in. ``None`` reproduces Phase 6A exactly: the frozen
        #: constant is used, no window is kept, and no record gains a field.
        self.variability = variability
        #: Phase 8.2, opt-in. ``None`` reproduces Phase 6A exactly: both the
        #: motor accuracy term and the fatigue error_rise term use the frozen
        #: ``movement_error_ref``. Supplying a config separates the two, which
        #: is the whole point -- the motor measurement is corrected without
        #: silently reinterpreting fatigue.
        self.normalisation = normalisation

    # -------------------------------------------------- pseudo-observations

    def _neuro_variance(self, base: float, neuro) -> float:
        """Inflate an observation variance for a low-quality neuro channel.

        Two multipliers, both monotone: an uncalibrated posterior is penalised by
        a fixed factor, and a high-entropy posterior is penalised in proportion
        to its entropy. A coin-flip posterior therefore contributes almost
        nothing, which is the correct treatment of a decoder that has told you it
        does not know.
        """
        variance = base
        if not neuro.calibrated:
            variance *= self.config.uncalibrated_variance_penalty
        variance *= 1.0 + 3.0 * neuro.uncertainty
        return variance

    def attention_observation(self, obs: Observation,
                              latent: LatentState) -> Optional[PseudoObservation]:
        """Attention from decoder confidence and reaction time.

        Assumption: engagement shows up both as a sharper class posterior and as
        a faster response. Neither is a measurement of attention; both are
        plausible correlates and are weighted equally by default.

        The posterior term is **gated on the twin's current trust in the neuro
        channel**. A sharp posterior only indicates engagement if the posterior
        means anything; from a channel the twin has learned to distrust it
        indicates nothing, and letting it through produced a confidently wrong
        attention estimate on real EEG (see ``attention_min_channel_trust``).
        The gate reads the belief from the *previous* tick, before this tick's
        update, so the two latents cannot chase each other within a tick.
        """
        cfg = self.config
        terms, sources, variance = [], [], 0.0

        if obs.neuro is not None:
            trust = latent.confidence_estimate
            trusted = trust.is_informed and trust.value >= cfg.attention_min_channel_trust
            if trusted:
                confidence_score = (obs.neuro.decoder_confidence - 0.5) * 2.0
                terms.append((cfg.attention_w_confidence, _clip01(confidence_score)))
                sources.append("neuro.decoder_confidence")
                variance += self._neuro_variance(
                    cfg.attention_observation_variance, obs.neuro)

        if obs.motor is not None:
            reaction_score = 1.0 - obs.motor.reaction_time / (2.0 * cfg.reaction_time_ref_s)
            terms.append((cfg.attention_w_reaction, _clip01(reaction_score)))
            sources.append("motor.reaction_time")
            variance += cfg.attention_observation_variance

        if not terms:
            return None

        weight = sum(w for w, _ in terms)
        value = sum(w * v for w, v in terms) / weight
        # Halving the variance for two independent contributions is the usual
        # inverse-variance intuition; here it is applied crudely because the two
        # terms are certainly not independent.
        return PseudoObservation(_clip01(value), variance / len(terms), tuple(sources))

    def fatigue_observation(self, obs: Observation, memory: EstimatorMemory,
                            time_on_task_s: float) -> Optional[PseudoObservation]:
        """Fatigue from accumulated time on task and within-session degradation.

        The degradation term is computed against the subject's *own* early-session
        baseline, never against a cohort value. That choice is inherited directly
        from Paper 1, where a single absolute amplitude threshold was shown to
        delete an entire subject whose data were in fact the cleanest in the set.
        """
        cfg = self.config
        load = _saturating(time_on_task_s, cfg.fatigue_time_constant_s)
        sources = ["performance.time_on_task_s"]

        if obs.motor is None or not memory.has_baseline:
            # Time still passes without a trial, so load alone is informative --
            # but weakly, so the variance is inflated.
            return PseudoObservation(
                _clip01(load), cfg.fatigue_observation_variance * 3.0, tuple(sources)
            )

        # Phase 8.2: deliberately the *fatigue* reference, which stays at the
        # frozen 0.25 by default and under the Phase 8.2 config alike. Fatigue
        # is not what this phase corrects.
        error_rise = ((obs.motor.movement_error - float(memory.baseline_movement_error))
                      / fatigue_error_ref(cfg, self.normalisation))
        time_rise = ((obs.motor.completion_time - float(memory.baseline_completion_time))
                     / cfg.completion_time_ref_s)
        smoothness_fall = float(memory.baseline_smoothness) - obs.motor.smoothness
        degradation = _clip01(0.5 + (error_rise + time_rise + smoothness_fall) / 3.0)
        sources += ["motor.movement_error", "motor.completion_time", "motor.smoothness"]

        value = cfg.fatigue_w_load * load + cfg.fatigue_w_degradation * degradation
        return PseudoObservation(_clip01(value), cfg.fatigue_observation_variance,
                                 tuple(sources))

    def recovery_observation(self, obs: Observation) -> Optional[PseudoObservation]:
        """Recovery from difficulty-weighted, assistance-discounted success.

        Assumption: a success only counts as evidence of expressed capability to
        the extent the system did not supply it. At full assistance the term is
        zero regardless of outcome, and the variance is inflated accordingly, so
        an assisted session cannot drive the recovery estimate anywhere.
        """
        cfg = self.config
        if obs.motor is None:
            return None

        difficulty_weight = 1.0 - cfg.recovery_difficulty_gain * (1.0 - obs.task.difficulty)
        credit = 1.0 - cfg.recovery_assistance_discount * obs.task.assistance_level
        value = (1.0 if obs.motor.task_success else 0.0) * difficulty_weight * max(0.0, credit)

        # As assistance approaches 1 the observation says nothing, so it must not
        # be allowed to pull the belief towards zero with high confidence.
        variance = cfg.recovery_observation_variance / max(1e-3, max(0.0, credit))
        return PseudoObservation(_clip01(value), variance,
                                 ("motor.task_success", "task.difficulty",
                                  "task.assistance_level"))

    def motor_state_observation(self, obs: Observation,
                                memory: EstimatorMemory) -> Optional[PseudoObservation]:
        """How well the trained motor pattern is currently being expressed."""
        cfg = self.config
        if obs.motor is None:
            return None

        # Phase 8.2: the motor reference. At the frozen 0.25 the accuracy term
        # reads only the best quarter of movement_error's range; at 1.00 it is
        # the exact complement of the contract-normalised error.
        accuracy = _clip01(
            1.0 - obs.motor.movement_error / motor_error_ref(cfg, self.normalisation))
        value = (cfg.motor_w_success * (1.0 if obs.motor.task_success else 0.0)
                 + cfg.motor_w_accuracy * accuracy
                 + cfg.motor_w_smoothness * obs.motor.smoothness)
        weight = cfg.motor_w_success + cfg.motor_w_accuracy + cfg.motor_w_smoothness
        z = _clip01(value / weight)
        sources = ("motor.task_success", "motor.movement_error",
                   "motor.smoothness")

        if self.variability is None:
            # The frozen path, unchanged and byte-identical to Phase 6A.
            return PseudoObservation(z, cfg.motor_observation_variance, sources)

        # Phase 6H. The window deliberately excludes the observation being
        # weighed: letting a measurement set its own trust is circular, and
        # would let one surprising value both move the mean and excuse itself
        # by inflating the variance.
        detail = observation_variance(memory.motor_z_window, self.variability)
        return PseudoObservation(z, detail.r, sources, variance_detail=detail)

    def confidence_observation(self, obs: Observation,
                               memory: EstimatorMemory) -> Optional[PseudoObservation]:
        """How much the twin should currently trust its own neuro channel.

        Note what this is *not*: it is not the patient's confidence. It is the
        twin's belief about the reliability of the decoder for this person right
        now, which is the quantity a decision engine actually needs before it
        acts on a LEFT/RIGHT posterior.
        """
        cfg = self.config
        if obs.neuro is None:
            return None

        stated = _clip01((obs.neuro.decoder_confidence - 0.5) * 2.0)
        if not obs.neuro.calibrated:
            stated *= 1.0 - cfg.uncalibrated_confidence_discount
        terms = [(cfg.confidence_w_decoder, stated)]
        sources = ["neuro.decoder_confidence"]

        if obs.task.target_side in (LEFT, RIGHT):
            agreement = 1.0 if obs.neuro.predicted_class == obs.task.target_side else 0.0
            terms.append((cfg.confidence_w_agreement, agreement))
            sources.append("task.target_side")

        comparable = (memory.previous_signed_margin is not None
                      and memory.previous_target_side is not None
                      and memory.previous_target_side == obs.task.target_side)
        if comparable:
            margin = obs.neuro.left_probability - obs.neuro.right_probability
            # Divided by 2 because the signed margin spans [-1, 1]: a full
            # confident reversal is a distance of 2 and must score 0, not -1.
            stability = 1.0 - abs(margin - float(memory.previous_signed_margin)) / 2.0
            terms.append((cfg.confidence_w_stability, _clip01(stability)))
            sources.append("neuro.signed_margin[t-1]")

        weight = sum(w for w, _ in terms)
        value = sum(w * v for w, v in terms) / weight
        variance = self._neuro_variance(cfg.confidence_observation_variance, obs.neuro)
        return PseudoObservation(_clip01(value), variance, tuple(sources))

    # ------------------------------------------------------------- the update

    def update(self, latent: LatentState, memory: EstimatorMemory,
               obs: Observation, *, dt: float,
               time_on_task_s: float) -> Tuple[LatentState, EstimatorMemory,
                                               Dict[str, Optional[PseudoObservation]]]:
        """Advance every latent by one tick.

        Returns the new belief, the new memory and the pseudo-observations that
        produced the change, so a caller can log why a latent moved rather than
        only that it did.
        """
        cfg = self.config
        if dt < 0:
            raise ValueError("dt cannot be negative")

        memory = self._fold_baseline(memory, obs)

        pseudo: Dict[str, Optional[PseudoObservation]] = {
            "motor_state_estimate": self.motor_state_observation(obs, memory),
            "attention_estimate": self.attention_observation(obs, latent),
            "fatigue_estimate": self.fatigue_observation(obs, memory, time_on_task_s),
            "recovery_estimate": self.recovery_observation(obs),
            "confidence_estimate": self.confidence_observation(obs, memory),
        }

        drift_and_rate = {
            "motor_state_estimate": (0.0, cfg.motor_process_rate),
            "attention_estimate": (0.0, cfg.attention_process_rate),
            "fatigue_estimate": (cfg.fatigue_drift_per_s * dt, cfg.fatigue_process_rate),
            "recovery_estimate": (0.0, cfg.recovery_process_rate),
            "confidence_estimate": (0.0, cfg.confidence_process_rate),
        }

        updated: Dict[str, Estimate] = {}
        for name in LatentState.FIELDS:
            drift, rate = drift_and_rate[name]
            belief: Estimate = getattr(latent, name)
            belief = belief.predicted(drift=drift, process_variance=rate * dt)
            candidate = pseudo[name]
            if candidate is not None:
                belief = belief.updated(candidate.value, candidate.variance, at=obs.t)
            updated[name] = belief

        memory = self._fold_trailing(memory, obs, pseudo["motor_state_estimate"])
        return LatentState(**updated), memory, pseudo

    # ------------------------------------------------------------ memory folds

    def _fold_baseline(self, memory: EstimatorMemory,
                       obs: Observation) -> EstimatorMemory:
        """Accumulate the within-subject baseline over the first trials."""
        if obs.motor is None or memory.has_baseline:
            return memory

        n = memory.baseline_n + 1
        sum_error = memory.baseline_sum_error + obs.motor.movement_error
        sum_time = memory.baseline_sum_time + obs.motor.completion_time
        sum_smooth = memory.baseline_sum_smoothness + obs.motor.smoothness

        if n < self.config.baseline_trials:
            return replace(memory, baseline_n=n, baseline_sum_error=sum_error,
                           baseline_sum_time=sum_time,
                           baseline_sum_smoothness=sum_smooth)

        return replace(
            memory,
            baseline_n=n,
            baseline_sum_error=sum_error,
            baseline_sum_time=sum_time,
            baseline_sum_smoothness=sum_smooth,
            baseline_movement_error=sum_error / n,
            baseline_completion_time=sum_time / n,
            baseline_smoothness=sum_smooth / n,
        )

    def _fold_trailing(self, memory: EstimatorMemory, obs: Observation,
                       motor_pseudo: Optional[PseudoObservation] = None
                       ) -> EstimatorMemory:
        """Carry forward the per-tick memories the next update will need."""
        updates: Dict[str, object] = {}
        if self.variability is not None and motor_pseudo is not None:
            # Appended *after* this tick's update, so trial t is weighed using
            # trials 0..t-1 only. Bounded to window+1 values, which is exactly
            # what ``window`` differences needs.
            keep = self.variability.window + 1
            updates["motor_z_window"] = tuple(
                (*memory.motor_z_window, motor_pseudo.value))[-keep:]
        if obs.neuro is not None:
            updates["previous_signed_margin"] = (obs.neuro.left_probability
                                                 - obs.neuro.right_probability)
            updates["previous_target_side"] = obs.task.target_side
        if obs.motor is not None:
            updates["consecutive_failures"] = (
                0 if obs.motor.task_success else memory.consecutive_failures + 1
            )
        return replace(memory, **updates) if updates else memory
