"""Phase 6B -- frozen Paper 1 neuro-state encoder integration.

The adapter layer between the frozen Paper 1 decoder and the validated Phase 6A
digital twin.

```
Paper 1 frozen EEG representation   (src.features, src.io, src.config)
        |
Frozen decoder, re-executed         (src.encoder.paper1)
        |
NeuroState adapter + provenance     (src.encoder.adapter, src.encoder.provenance)
        |
Phase 6A DigitalTwin                (src.twin, unchanged)
        |
VirtualPatientState
```

Layering rule
-------------
``src.twin`` imports nothing from Paper 1 and a test enforces it. ``src.encoder``
is the single layer that depends on both, which is what keeps the twin reusable
with a different encoder later.

Three things this package guarantees
------------------------------------
1. **Paper 1 is re-executed, not retrained.** No saved decoder exists, so the
   frozen code is re-run with the frozen configuration and verified against the
   published per-subject results before anything is emitted.
2. **Subject independence is audited, not assumed.** Every fold records the
   subjects it trained on, and a state cannot be produced from a fold whose
   held-out subject appears in that list.
3. **Nothing is presented as calibrated.** Paper 1 established calibration only
   with labelled epochs from the held-out subject, which this protocol forbids,
   so every posterior is emitted uncalibrated and labelled as such.
"""

from src.encoder.adapter import (
    DEFAULT_MIN_MARGIN,
    EpochReliability,
    NeuroStateRecord,
    Paper1NeuroStateAdapter,
    subject_summary,
)
from src.encoder.paper1 import (
    PRIMARY_CHANNEL_SET,
    PRIMARY_CLASSES,
    PRIMARY_MODEL,
    PRIMARY_REPRESENTATION,
    PRIMARY_TASK,
    CohortEncoding,
    EpochPosterior,
    encode_cohort,
)
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

__all__ = [
    "DEFAULT_MIN_MARGIN",
    "PAPER1",
    "PAPER1_BELOW_CHANCE",
    "PRIMARY_CHANNEL_SET",
    "PRIMARY_CLASSES",
    "PRIMARY_MODEL",
    "PRIMARY_REPRESENTATION",
    "PRIMARY_TASK",
    "CohortEncoding",
    "EncoderProvenanceError",
    "EpochPosterior",
    "EpochReliability",
    "FoldProvenance",
    "NeuroStateRecord",
    "Paper1NeuroStateAdapter",
    "SubjectLeakageError",
    "assert_subject_independence",
    "encode_cohort",
    "parse_encoder_id",
    "run_provenance",
    "subject_summary",
    "verify_encoder_id",
]
