"""Phase 6B -- provenance and the subject-independence audit.

Every ``NeuroState`` this package emits was produced by a decoder fitted on a
specific set of subjects. If the held-out subject is among them, the state is
worthless and every downstream number is contaminated. That is not a risk to be
managed by care; it is checked, on every fold, by
:func:`assert_subject_independence`, and the check has a failing control in
:mod:`tests.test_encoder_provenance`.

Why the provenance is structural rather than a log line
-------------------------------------------------------
A log can be read after the fact. A :class:`FoldProvenance` has to exist before
an adapter will emit anything, it carries the training-subject list, and its
``encoder_id`` is stamped onto every ``NeuroState``. So even a single state that
has been serialised, moved between machines and reloaded still names the fold
that produced it and the subject that fold held out.

Paper 1 reference values
------------------------
:data:`PAPER1` holds the numbers Phase 6B inherits rather than recomputes.
:mod:`tests.test_encoder_provenance` asserts that each still matches the frozen
artifact on disk, so this module cannot drift away from the paper it cites.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

#: Numbers inherited from the frozen Paper 1 primary configuration
#: (ROI-21, frozen band-power features, logistic regression, LOSOCV,
#: LEFT vs RIGHT). Sources are named so each can be re-checked.
PAPER1: Dict[str, object] = {
    "representation": "frozen band-power features",
    "channel_set": "roi21",
    "model": "logistic_regression",
    "task": "left_vs_right",
    "protocol": "leave-one-subject-out cross-validation",
    "n_subjects": 40,
    "n_binary_epochs": 1800,
    "n_features": 96,
    "mean_balanced_accuracy": 0.5987727625635234,
    "below_chance_subjects": 8,
    # Uncalibrated expected calibration error of this exact configuration.
    # Source: data/metadata/phase4a/phase4a_summary.json -> calibration.none.ece
    "uncalibrated_ece": 0.16933407808402773,
    "source_artifacts": (
        "data/metadata/phase4b/phase4b_summary.json",
        "data/metadata/phase4b/phase4b_subject_results.csv",
        "data/metadata/phase4a/phase4a_summary.json",
    ),
}

#: The eight subjects Paper 1 found at or below chance under the primary
#: configuration. Read from the frozen Phase 5 outcome definition, never
#: recomputed. Used **only** to select stress-test cases for the report; it is
#: never available to the adapter at inference time.
PAPER1_BELOW_CHANCE: Tuple[str, ...] = (
    "S003", "S009", "S018", "S021", "S024", "S028", "S036", "S038",
)


class SubjectLeakageError(RuntimeError):
    """Raised when a held-out subject appears in its own training provenance."""


class EncoderProvenanceError(RuntimeError):
    """Raised when a state's encoder provenance is missing or inconsistent."""


@dataclass(frozen=True)
class FoldProvenance:
    """Who trained the decoder that produced one subject's neuro states.

    Parameters
    ----------
    held_out_subject
        The subject whose epochs this fold scores. Must not appear in
        ``training_subjects``.
    training_subjects
        Every subject whose epochs were used to fit the pipeline, including the
        scaler. Stored explicitly rather than derived, so the audit checks what
        actually happened rather than what should have.
    calibrated
        Always ``False`` in Phase 6B, and there is no code path that sets it
        otherwise. Paper 1's Platt calibration required 20 labelled epochs from
        the held-out subject, which the subject-independent protocol forbids.
    """

    held_out_subject: str
    training_subjects: Tuple[str, ...]
    n_train_epochs: int
    n_test_epochs: int
    model: str
    representation: str
    channel_set: str
    task: str
    n_features: int
    seed: int
    dataset_fingerprint: str
    calibrated: bool = False

    def __post_init__(self) -> None:
        if self.calibrated:
            raise EncoderProvenanceError(
                "Phase 6B emits uncalibrated posteriors only. Paper 1 established "
                "calibration solely with 20 labelled epochs from the held-out "
                "subject, which the subject-independent protocol forbids."
            )
        if not self.training_subjects:
            raise EncoderProvenanceError("a fold with no training subjects is not a fold")
        self.audit()

    # ------------------------------------------------------------------ audit

    def audit(self) -> None:
        """Raise if the held-out subject contributed to training."""
        if self.held_out_subject in self.training_subjects:
            raise SubjectLeakageError(
                f"held-out subject {self.held_out_subject} appears in its own "
                f"training provenance ({len(self.training_subjects)} subjects). "
                "Every state from this fold is contaminated."
            )

    # ------------------------------------------------------------------- id

    @property
    def encoder_id(self) -> str:
        """Stamped onto every ``NeuroState`` this fold produces.

        Deliberately readable and self-describing: a state that has been
        serialised and moved still names its representation, its model, the
        subject it held out, its calibration status and the dataset it came
        from. ``uncal`` is in the string because a reader should not have to
        look anything up to know the posterior is uncalibrated.
        """
        return (
            f"paper1/{self.representation}/{self.channel_set}/{self.model}"
            f"/{self.task}/holdout={self.held_out_subject}"
            f"/uncal/cfg={self.dataset_fingerprint}"
        )

    @property
    def fingerprint(self) -> str:
        payload = {k: (sorted(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(self).items()}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["training_subjects"] = list(self.training_subjects)
        payload["encoder_id"] = self.encoder_id
        payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FoldProvenance":
        return cls(
            held_out_subject=str(payload["held_out_subject"]),
            training_subjects=tuple(payload["training_subjects"]),  # type: ignore[arg-type]
            n_train_epochs=int(payload["n_train_epochs"]),  # type: ignore[arg-type]
            n_test_epochs=int(payload["n_test_epochs"]),  # type: ignore[arg-type]
            model=str(payload["model"]),
            representation=str(payload["representation"]),
            channel_set=str(payload["channel_set"]),
            task=str(payload["task"]),
            n_features=int(payload["n_features"]),  # type: ignore[arg-type]
            seed=int(payload["seed"]),  # type: ignore[arg-type]
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
            calibrated=bool(payload.get("calibrated", False)),
        )


def assert_subject_independence(folds: Iterable[FoldProvenance]) -> Dict[str, object]:
    """Audit a whole run. Raises on the first violation; returns a summary.

    Checks three things, because each catches a different mistake:

    1. no fold trains on its own held-out subject (the leak that matters);
    2. every subject is held out exactly once (a fold silently dropped would
       make the cohort mean quietly wrong);
    3. every fold's training set is the full cohort minus that one subject
       (a fold trained on a subset is not the protocol Paper 1 reported).
    """
    folds = list(folds)
    if not folds:
        raise EncoderProvenanceError("no folds to audit")

    for fold in folds:
        fold.audit()

    held_out = [f.held_out_subject for f in folds]
    duplicates = {s for s in held_out if held_out.count(s) > 1}
    if duplicates:
        raise EncoderProvenanceError(
            f"subjects held out more than once: {sorted(duplicates)}"
        )

    cohort = set(held_out)
    for fold in folds:
        expected = cohort - {fold.held_out_subject}
        actual = set(fold.training_subjects)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise EncoderProvenanceError(
                f"fold {fold.held_out_subject} trained on the wrong cohort; "
                f"missing={missing} unexpected={extra}"
            )

    return {
        "n_folds": len(folds),
        "cohort_size": len(cohort),
        "subjects": sorted(cohort),
        "all_uncalibrated": all(not f.calibrated for f in folds),
        "leakage_detected": False,
    }


def verify_encoder_id(encoder_id: str, expected: FoldProvenance) -> None:
    """Raise if a state's ``encoder_id`` does not match the fold that owns it.

    Guards the case where states from different folds are concatenated and one
    subject's rows silently carry another subject's decoder.
    """
    if encoder_id != expected.encoder_id:
        raise EncoderProvenanceError(
            f"encoder_id mismatch:\n  found    {encoder_id}\n  expected {expected.encoder_id}"
        )


def parse_encoder_id(encoder_id: str) -> Dict[str, str]:
    """Recover the fields stamped into an ``encoder_id``.

    Used by the audit tests and by anyone holding a bare ``NeuroState``.
    """
    if not encoder_id.startswith("paper1/"):
        raise EncoderProvenanceError(
            f"not a Paper 1 encoder id: {encoder_id!r}. A state whose provenance "
            "cannot be parsed must not be treated as a Paper 1 state."
        )
    parts = encoder_id.split("/")
    if len(parts) != 8:
        raise EncoderProvenanceError(f"malformed encoder id: {encoder_id!r}")
    _, representation, channel_set, model, task, holdout, calibration, cfg = parts
    if not holdout.startswith("holdout=") or not cfg.startswith("cfg="):
        raise EncoderProvenanceError(f"malformed encoder id: {encoder_id!r}")
    return {
        "representation": representation,
        "channel_set": channel_set,
        "model": model,
        "task": task,
        "held_out_subject": holdout.removeprefix("holdout="),
        "calibration": calibration,
        "dataset_fingerprint": cfg.removeprefix("cfg="),
    }


# --------------------------------------------------------------- run record


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=False)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def _library_versions() -> Dict[str, str]:
    import numpy
    import pandas
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
    }


def run_provenance(folds: Sequence[FoldProvenance], *,
                   reproduction: Optional[Mapping[str, object]] = None,
                   extra: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """The header written at the top of every Phase 6B artifact."""
    audit = assert_subject_independence(folds)
    record: Dict[str, object] = {
        "kind": "header",
        "phase": "6B",
        "schema_version": "phase6b.1",
        "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "libraries": _library_versions(),
        "paper1": dict(PAPER1),
        "subject_independence_audit": audit,
        "calibration": {
            "calibrated": False,
            "why": (
                "Paper 1 established calibration only via Platt scaling on 20 "
                "labelled epochs from the held-out subject. The subject-"
                "independent protocol forbids using held-out-subject data, so "
                "Phase 6B emits the raw uncalibrated posterior."
            ),
            "uncalibrated_ece_paper1": PAPER1["uncalibrated_ece"],
            "caveat": (
                "The posterior is the decoder's score for a class. It is NOT a "
                "probability of correctness, NOT patient confidence and NOT "
                "clinical confidence."
            ),
        },
    }
    if reproduction is not None:
        record["paper1_reproduction"] = dict(reproduction)
    if extra:
        record.update(extra)
    return record
