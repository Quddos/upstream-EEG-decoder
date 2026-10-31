"""Phase 8.2 -- opt-in motor-error normalisation, separated from fatigue.

The defect
----------
``LatentModelConfig.movement_error_ref`` is 0.25 and is applied to
``movement_error``, which the Phase 6E contract has *already* normalised to
``[0, 1]`` by dividing the endpoint error by ``ERROR_REFERENCE_M = 0.30``. The
composition means the accuracy term reaches zero at an endpoint error of
``0.30 x 0.25 = 0.075 m`` -- one and a half times the default target radius --
so it reads only the best quarter of its own input's range. Measured over the
declared calibration population, **21.6% of observations were clipped to
exactly 0** and all further error was discarded (see
``docs/PHASE8_2_BASELINE_AUDIT.md``).

The fix
-------
``motor_error_ref = 1.00`` makes ``accuracy = 1 - movement_error``: the exact
complement of the contract-normalised error, with no second normalisation and
nothing discarded. It is the only value with an interpretation that does not
require an argument nobody has made.

Why the two references are split
--------------------------------
``movement_error_ref`` has **two** consumers, and Phase 8.2 was commissioned to
fix only one of them:

* ``motor_state_observation`` -- the accuracy term, the target of this phase;
* ``fatigue_observation`` -- the ``error_rise`` term, which is *not*.

Changing a single shared constant would have moved ``fatigue_estimate`` by
-0.062 on the ``deteriorating`` scenario as an unannounced side effect. The
audit measured that, and the phase brief accordingly requires two references.
``fatigue_error_ref`` therefore stays at 0.25: **Phase 8.2 changes the motor
measurement chain, not the fatigue estimator.** No claim is made that fatigue is
improved, and nothing about its interpretation, thresholds or policy use
changes.

Why this is a separate config rather than an edit
-------------------------------------------------
``movement_error_ref`` is a field of ``LatentModelConfig``, so it is inside the
frozen Phase 6A fingerprint ``42dcd25435726422``. Editing it would end a freeze
that Phases 6B-8.1 preserved, including two changes (6G, 6H) built by
composition precisely to avoid touching it. This module follows the same
pattern: the default is ``None`` everywhere, the frozen path is byte-identical,
and the Phase 8.2 behaviour has a fingerprint of its own.

What this does NOT do
---------------------
It does not change what the twin estimates. The Phase 8.0 estimand is unchanged:
``motor_state_estimate`` estimates **recent expressed task performance**,
conditioned on task geometry, assistance level and the measurement chain.
``movement_error_ref`` is part of that chain, which the estimand already names
as a conditioning variable -- so this alters the chain the estimate is
conditioned on, exactly as Phase 8.1 did, not the thing being estimated.

It does not estimate motor capability, physiological function, neurological
function, rehabilitation efficacy, clinical motor function or human reaching
ability. It does not touch the Kalman recursion, so it **cannot** affect
posterior uncertainty or the Phase 6G decidability band, and it is not intended
to.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, Optional

#: The frozen Phase 6A value, kept here so the default path can be written
#: without importing the estimator config and so tests can assert the identity.
FROZEN_ERROR_REF: float = 0.25


@dataclass(frozen=True)
class MotorNormalisationConfig:
    """Separate error references for the motor and fatigue observations.

    The defaults are the Phase 8.2 specification: the motor term is corrected,
    the fatigue term is deliberately left where Phase 6A put it.
    """

    #: Denominator of the motor accuracy term. 1.00 means
    #: ``accuracy = 1 - movement_error``, i.e. no second normalisation.
    motor_error_ref: float = 1.00

    #: Denominator of the fatigue ``error_rise`` term. Held at the frozen 0.25
    #: on purpose: Phase 8.2 is a motor-measurement change and must not
    #: silently reinterpret fatigue.
    fatigue_error_ref: float = FROZEN_ERROR_REF

    def __post_init__(self) -> None:
        for name in ("motor_error_ref", "fatigue_error_ref"):
            value = getattr(self, name)
            if not (0.0 < float(value) <= 1.0):
                raise ValueError(
                    f"{name} must lie in (0, 1]: movement_error is already "
                    f"contract-normalised to [0, 1], so a reference above 1.0 "
                    f"would make the term unreachable. Got {value!r}")

    def to_dict(self) -> Dict[str, float]:
        return dict(asdict(self))

    @property
    def fingerprint(self) -> str:
        """Stable 16-hex hash, mirroring every other config in the twin."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @property
    def is_frozen_equivalent(self) -> bool:
        """True when this config reproduces Phase 6A exactly.

        Useful to a caller that wants to assert it has *not* opted in, and to
        the report, which must never describe the frozen path as changed.
        """
        return (self.motor_error_ref == FROZEN_ERROR_REF
                and self.fatigue_error_ref == FROZEN_ERROR_REF)


#: The Phase 8.2 specification, named so artifacts can record what they ran.
PHASE8_2_NORMALISATION = MotorNormalisationConfig(motor_error_ref=1.00,
                                                  fatigue_error_ref=0.25)

#: The control: both references at the frozen constant. Behaviourally identical
#: to passing ``None``, and used by the tests to prove that.
FROZEN_EQUIVALENT = MotorNormalisationConfig(motor_error_ref=FROZEN_ERROR_REF,
                                             fatigue_error_ref=FROZEN_ERROR_REF)


def motor_error_ref(config, normalisation: Optional[MotorNormalisationConfig]
                    ) -> float:
    """The denominator the motor accuracy term should use.

    ``None`` yields the frozen ``LatentModelConfig.movement_error_ref``, so the
    default path is byte-identical to Phase 6A.
    """
    if normalisation is None:
        return config.movement_error_ref
    return normalisation.motor_error_ref


def fatigue_error_ref(config, normalisation: Optional[MotorNormalisationConfig]
                      ) -> float:
    """The denominator the fatigue ``error_rise`` term should use."""
    if normalisation is None:
        return config.movement_error_ref
    return normalisation.fatigue_error_ref
