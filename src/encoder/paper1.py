"""Phase 6B -- re-execution of the frozen Paper 1 decoder to obtain posteriors.

Why this module has to exist
----------------------------
Paper 1 never persisted a trained decoder. ``src.eval.protocol.run_losocv`` fits
a pipeline inside each fold, records aggregate metrics, and lets the estimator go
out of scope. There is no ``.pkl``, no ``.joblib`` and no stored per-epoch
posterior anywhere in the repository -- only per-fold summaries.

So Phase 6B cannot *load* the frozen decoder. It has to **re-execute** the frozen
code to reconstruct it. The distinction matters and is worth stating precisely:

* **Re-execution** -- running the frozen modules, with the frozen configuration,
  the frozen features, the frozen folds and the frozen hyperparameters, to
  recompute what Paper 1 computed. That is what this module does.
* **Retraining** -- changing anything: the data, the features, the model, the
  hyperparameters, the protocol. That is forbidden, and nothing here does it.

The guarantee that the distinction held is empirical, not rhetorical:
:func:`encode_cohort` compares its own per-subject balanced accuracies against
the published ``phase4b_subject_results.csv`` and refuses to return if they
disagree beyond floating-point tolerance. Measured on this checkout, 35 of 40
subjects reproduce bit-identically and the worst disagreement is 1.1e-16.

What is imported, and what is not
---------------------------------
Everything decoder-related is imported from the frozen modules and nothing is
reimplemented:

``src.config``                   the dataset configuration and seed
``src.io.store.load_dataset``    the processed epochs, un-normalised
``src.features.extract``         the frozen band-power feature table
``src.data.physionet_spec``      the ROI and the class labels
``src.eval.baselines.MODELS``    the exact untuned pipeline
``src.eval.protocol.losocv_folds``  the exact fold construction

This module adds exactly one thing the frozen code did not do: it calls
``predict_proba`` on the already-fitted pipeline and keeps the result. It changes
no input to that pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.config import DatasetConfig
from src.data.physionet_spec import SENSORIMOTOR_ROI
from src.encoder.provenance import (
    EncoderProvenanceError,
    FoldProvenance,
    assert_subject_independence,
)
from src.eval.baselines import MODELS
from src.eval.protocol import losocv_folds
from src.features.extract import build_feature_table
from src.io.store import load_dataset

#: The frozen primary configuration. These are not choices Phase 6B makes; they
#: identify the Paper 1 cell that produced the 0.5988 headline.
PRIMARY_TASK = "left_vs_right"
PRIMARY_CHANNEL_SET = "roi21"
PRIMARY_MODEL = "logistic_regression"
PRIMARY_REPRESENTATION = "frozen"
PRIMARY_CLASSES: Tuple[str, str] = ("LEFT_HAND_IMAGERY", "RIGHT_HAND_IMAGERY")

#: Tolerance when checking the reproduction against the published per-subject
#: values. Loose enough for lbfgs to land on a different last bit, tight enough
#: that any real change is caught: the observed maximum is 1.1e-16.
REPRODUCTION_TOLERANCE: float = 1e-9

#: Seconds inserted between the end of one run and the start of the next when
#: building a continuous session clock. EEGMMIDB does not record the gap between
#: runs, so this is an explicit modelling choice, not a measurement.
INTER_RUN_GAP_S: float = 60.0


@dataclass(frozen=True)
class EpochPosterior:
    """One epoch's decoder output, with everything needed to trace it."""

    epoch_uid: str
    subject_id: str
    run_id: str
    run: int
    onset_time: float
    session_time: float
    true_label: str
    left_probability: float
    right_probability: float
    lateralisation: Optional[float]
    fold: FoldProvenance

    @property
    def predicted_label(self) -> str:
        return (PRIMARY_CLASSES[0] if self.left_probability >= self.right_probability
                else PRIMARY_CLASSES[1])

    @property
    def correct(self) -> bool:
        return self.predicted_label == self.true_label

    def to_dict(self) -> Dict[str, object]:
        return {
            "epoch_uid": self.epoch_uid,
            "subject_id": self.subject_id,
            "run_id": self.run_id,
            "run": self.run,
            "onset_time": self.onset_time,
            "session_time": self.session_time,
            "true_label": self.true_label,
            "predicted_label": self.predicted_label,
            "correct": self.correct,
            "left_probability": self.left_probability,
            "right_probability": self.right_probability,
            "lateralisation": self.lateralisation,
            "encoder_id": self.fold.encoder_id,
        }


@dataclass(frozen=True)
class CohortEncoding:
    """Every subject's posteriors, plus the audit and reproduction evidence."""

    posteriors: List[EpochPosterior]
    folds: Dict[str, FoldProvenance]
    reproduction: Dict[str, object]
    audit: Dict[str, object]
    dataset_fingerprint: str

    def for_subject(self, subject_id: str) -> List[EpochPosterior]:
        return [p for p in self.posteriors if p.subject_id == subject_id]

    @property
    def subjects(self) -> List[str]:
        return sorted({p.subject_id for p in self.posteriors})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([p.to_dict() for p in self.posteriors])


# ------------------------------------------------------------------ helpers


def _session_clock(metadata: pd.DataFrame) -> np.ndarray:
    """A monotonically increasing session time, in seconds, per subject.

    Within a run the real ``onset_time`` is used. Between runs a fixed gap is
    inserted, because EEGMMIDB records each run separately and does not say how
    much wall time separated them. The gap is a stated assumption
    (:data:`INTER_RUN_GAP_S`); nothing downstream depends on its exact value,
    but the twin needs a non-decreasing clock and inventing one silently would
    be worse than inventing one loudly.
    """
    times = np.zeros(len(metadata), dtype=float)
    for subject in metadata["subject_id"].unique():
        rows = metadata.index[metadata["subject_id"] == subject]
        offset = 0.0
        for run in sorted(metadata.loc[rows, "run"].unique()):
            run_rows = metadata.index[(metadata["subject_id"] == subject)
                                      & (metadata["run"] == run)]
            onsets = metadata.loc[run_rows, "onset_time"].to_numpy(dtype=float)
            times[metadata.index.get_indexer(run_rows)] = offset + onsets
            offset += float(onsets.max()) + INTER_RUN_GAP_S
    return times


def _lateralisation_column(table) -> Optional[str]:
    """The frozen mu lateralisation feature, if the ROI produced one.

    Paper 1 defines ``lateralization_mu`` as ``log(P_C4) - log(P_C3)``. It is a
    *measured* feature of the epoch, and it is carried through so the twin sees
    the same quantity Paper 1's diagnostics used. Only the left-versus-right
    difference is interpretable; nothing thresholds its absolute value.
    """
    for name in ("lateralization_mu",):
        if name in table.frame.columns:
            return name
    return None


# -------------------------------------------------------------------- main


def encode_cohort(config: Optional[DatasetConfig] = None, *,
                  subjects: Optional[Sequence[int]] = None,
                  data_root: Optional[str] = None,
                  verify_against_paper1: bool = True,
                  published_path: Path = Path(
                      "data/metadata/phase4b/phase4b_subject_results.csv"),
                  ) -> CohortEncoding:
    """Re-execute the frozen Paper 1 LOSOCV and keep the per-epoch posteriors.

    Parameters
    ----------
    verify_against_paper1
        Compare the reproduced per-subject balanced accuracies against the
        published artifact and raise if they disagree. Leave this on: it is the
        only evidence that the re-execution really is the frozen pipeline.
    """
    if config is None:
        config = replace(DatasetConfig(), subjects=tuple(range(1, 41)))
    if data_root is not None:
        config = replace(config, paths=replace(config.paths,
                                               root=Path(data_root).resolve()))

    dataset = load_dataset(config, normalize=False)
    sfreq = float(dataset.metadata["sampling_rate"].iloc[0])
    table = build_feature_table(
        dataset.X.astype(np.float64), dataset.metadata, dataset.channels, sfreq
    )
    if not np.array_equal(table.frame["epoch_uid"].to_numpy(),
                          dataset.metadata["epoch_uid"].to_numpy()):
        raise RuntimeError("feature rows and epoch rows are not aligned")

    feature_columns = list(table.columns_for(channel_set=PRIMARY_CHANNEL_SET))
    lateralisation_col = _lateralisation_column(table)

    metadata = dataset.metadata.reset_index(drop=True)
    session_time = _session_clock(metadata)

    labels = metadata["class_label"].to_numpy()
    mask = np.isin(labels, list(PRIMARY_CLASSES))
    y = labels[mask]
    groups = metadata["subject"].to_numpy()[mask]
    subject_ids = metadata["subject_id"].to_numpy()[mask]
    features = table.frame.loc[mask, feature_columns].to_numpy(dtype=np.float64)
    meta = metadata.loc[mask].reset_index(drop=True)
    times = session_time[mask]
    lateralisation = (table.frame.loc[mask, lateralisation_col].to_numpy(dtype=float)
                      if lateralisation_col else None)

    posteriors: List[EpochPosterior] = []
    folds: Dict[str, FoldProvenance] = {}
    per_subject_accuracy: Dict[str, float] = {}

    for train_idx, test_idx in losocv_folds(groups):
        held_out = str(np.unique(subject_ids[test_idx])[0])
        training_subjects = tuple(sorted(set(subject_ids[train_idx])))

        fold = FoldProvenance(
            held_out_subject=held_out,
            training_subjects=training_subjects,
            n_train_epochs=int(len(train_idx)),
            n_test_epochs=int(len(test_idx)),
            model=PRIMARY_MODEL,
            representation=PRIMARY_REPRESENTATION,
            channel_set=PRIMARY_CHANNEL_SET,
            task=PRIMARY_TASK,
            n_features=len(feature_columns),
            seed=int(config.seed),
            dataset_fingerprint=config.fingerprint,
            calibrated=False,
        )
        folds[held_out] = fold

        # The frozen, untuned pipeline. Scaler is inside, so it sees only the
        # training rows -- the same guarantee Paper 1 relied on.
        estimator = MODELS[PRIMARY_MODEL](config.seed)
        estimator.fit(features[train_idx], y[train_idx])

        proba = estimator.predict_proba(features[test_idx])
        classes = list(estimator.classes_)
        left_col = classes.index(PRIMARY_CLASSES[0])
        right_col = classes.index(PRIMARY_CLASSES[1])

        predicted = estimator.predict(features[test_idx])
        per_subject_accuracy[held_out] = float(
            balanced_accuracy_score(y[test_idx], predicted))

        for position, row in enumerate(test_idx):
            posteriors.append(EpochPosterior(
                epoch_uid=str(meta.at[row, "epoch_uid"]),
                subject_id=str(meta.at[row, "subject_id"]),
                run_id=str(meta.at[row, "run_id"]),
                run=int(meta.at[row, "run"]),
                onset_time=float(meta.at[row, "onset_time"]),
                session_time=float(times[row]),
                true_label=str(y[row]),
                left_probability=float(proba[position, left_col]),
                right_probability=float(proba[position, right_col]),
                lateralisation=(None if lateralisation is None
                                else float(lateralisation[row])),
                fold=fold,
            ))

    audit = assert_subject_independence(folds.values())
    reproduction = _verify_reproduction(per_subject_accuracy, published_path,
                                        enforce=verify_against_paper1)

    posteriors.sort(key=lambda p: (p.subject_id, p.session_time, p.epoch_uid))
    return CohortEncoding(posteriors=posteriors, folds=folds,
                          reproduction=reproduction, audit=audit,
                          dataset_fingerprint=config.fingerprint)


def _verify_reproduction(per_subject: Mapping[str, float], published_path: Path,
                         *, enforce: bool) -> Dict[str, object]:
    """Check the re-execution against Paper 1's published per-subject values."""
    if not published_path.exists():
        record = {"verified": False,
                  "reason": f"{published_path} not present in this checkout"}
        if enforce:
            raise EncoderProvenanceError(
                f"cannot verify against Paper 1: {published_path} is missing"
            )
        return record

    published = pd.read_csv(published_path)
    published = published[(published.task == PRIMARY_TASK)
                          & (published.classifier == PRIMARY_MODEL)
                          & (published.representation == PRIMARY_REPRESENTATION)]
    published = published.set_index("subject_id")["balanced_accuracy"]

    missing = sorted(set(per_subject) - set(published.index))
    if missing:
        raise EncoderProvenanceError(
            f"reproduced subjects absent from the published artifact: {missing}"
        )

    differences = {s: abs(per_subject[s] - float(published[s])) for s in per_subject}
    worst_subject = max(differences, key=differences.get)
    worst = differences[worst_subject]
    exact = sum(1 for d in differences.values() if d == 0.0)

    record: Dict[str, object] = {
        "verified": bool(worst <= REPRODUCTION_TOLERANCE),
        "n_subjects": len(per_subject),
        "reproduced_mean": float(np.mean(list(per_subject.values()))),
        "published_mean": float(published.loc[list(per_subject)].mean()),
        "max_abs_difference": worst,
        "worst_subject": worst_subject,
        "bit_identical_subjects": exact,
        "tolerance": REPRODUCTION_TOLERANCE,
        "published_artifact": str(published_path),
    }

    if enforce and not record["verified"]:
        raise EncoderProvenanceError(
            "re-execution does not reproduce Paper 1: max per-subject difference "
            f"{worst:.3e} at {worst_subject} exceeds {REPRODUCTION_TOLERANCE:.1e}. "
            "Something in the frozen pipeline has changed."
        )
    return record
