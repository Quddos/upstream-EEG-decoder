"""Phase 2E -- leakage detection.

These are checks, not assertions, so they can be reused: the CLI runs them on the
real study and :mod:`tests.test_leakage` runs them on constructed cases including
**deliberately leaky ones that must be caught**. A test suite that only ever sees
clean input proves nothing about whether it can detect dirt.

The four ways this study could leak, and the function that catches each:

``subject overlap``
    the same person in train and test -- :func:`check_fold_subject_disjoint`
``scaler fitted on everything``
    standardising before splitting, so the test subject's mean and variance are
    baked into training -- :func:`check_scaler_fitted_on_train_only`
``global feature statistics``
    any featurisation step that looks across epochs, making a row depend on which
    other rows are present -- :func:`check_features_are_per_epoch`
``identity drift``
    labels or IDs silently reordered relative to the source index --
    :func:`check_labels_unchanged`, :func:`check_ids_traceable`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class AuditResult:
    """Outcome of one leakage check."""

    name: str
    passed: bool
    detail: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.passed


@dataclass
class AuditReport:
    """Outcome across all checks."""

    results: List[AuditResult] = field(default_factory=list)

    def add(self, result: AuditResult) -> "AuditReport":
        self.results.append(result)
        return self

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def failures(self) -> List[AuditResult]:
        return [r for r in self.results if not r.passed]

    def render(self) -> str:
        lines = []
        for result in self.results:
            mark = "PASS" if result.passed else "FAIL"
            lines.append(f"  [{mark}] {result.name}" + (f" -- {result.detail}" if result.detail else ""))
        lines.append("")
        lines.append(
            "LEAKAGE AUDIT PASSED" if self.passed
            else f"LEAKAGE AUDIT FAILED: {len(self.failures)} check(s)"
        )
        return "\n".join(lines)


def check_fold_subject_disjoint(
    folds: Sequence, groups: np.ndarray
) -> AuditResult:
    """No subject may appear in both sides of any fold."""
    overlaps = []
    for i, (train_idx, test_idx) in enumerate(folds):
        shared = set(np.unique(groups[train_idx])) & set(np.unique(groups[test_idx]))
        if shared:
            overlaps.append(f"fold {i}: subjects {sorted(shared)} in train and test")
    return AuditResult(
        name="no test-subject samples enter training",
        passed=not overlaps,
        detail="; ".join(overlaps) or f"{len(folds)} folds disjoint by subject",
    )


def check_every_subject_held_out_once(
    folds: Sequence, groups: np.ndarray
) -> AuditResult:
    """Leave-one-subject-out must hold out each subject exactly once."""
    held = [tuple(np.unique(groups[test_idx])) for _, test_idx in folds]
    flat = [s for group in held for s in group]
    subjects = sorted(np.unique(groups).tolist())
    ok = sorted(flat) == subjects and all(len(g) == 1 for g in held)
    return AuditResult(
        name="each subject held out exactly once",
        passed=ok,
        detail=f"{len(folds)} folds for {len(subjects)} subjects",
    )


def check_scaler_fitted_on_train_only(
    pipeline,
    X_train: np.ndarray,
    X_all: np.ndarray,
    *,
    step: str = "scale",
    tolerance: float = 1e-8,
) -> AuditResult:
    """The fitted scaler must carry the training fold's statistics, not everyone's.

    Both halves matter. Matching the training mean shows the scaler saw the right
    rows; *not* matching the global mean shows it did not also see the held-out
    ones. Checking only the first would pass a pipeline fitted on all the data
    whenever the fold happens to resemble the whole.
    """
    scaler = pipeline.named_steps.get(step)
    if scaler is None or not hasattr(scaler, "mean_"):
        return AuditResult(
            name="scaler fitted on training fold only",
            passed=False,
            detail=f"no fitted step named {step!r} in the pipeline",
        )

    matches_train = np.allclose(scaler.mean_, X_train.mean(axis=0), atol=tolerance)
    global_mean = X_all.mean(axis=0)
    matches_global = np.allclose(scaler.mean_, global_mean, atol=tolerance)

    # If the fold genuinely equals the whole dataset there is nothing to detect.
    distinguishable = not np.allclose(
        X_train.mean(axis=0), global_mean, atol=tolerance
    )

    passed = matches_train and (not matches_global or not distinguishable)
    # Report the most diagnostic cause first. When a scaler was fitted on
    # everything, both "does not match the training fold" and "matches the full
    # dataset" are true, but only the second tells the reader what went wrong.
    if matches_global and distinguishable:
        detail = "scaler statistics match the FULL dataset -- fitted before splitting"
    elif not matches_train:
        detail = "scaler statistics do not match the training fold"
    else:
        detail = "matches train fold, differs from global"
    return AuditResult(
        name="scaler fitted on training fold only", passed=passed, detail=detail
    )


def check_features_are_per_epoch(
    extract: Callable[[np.ndarray, pd.DataFrame], pd.DataFrame],
    X: np.ndarray,
    metadata: pd.DataFrame,
    subset: Sequence[int],
    *,
    tolerance: float = 1e-10,
) -> AuditResult:
    """Featurising a subset must equal the corresponding rows of the full run.

    This is the structural proof that no feature uses a statistic pooled across
    epochs. If any step normalised by a dataset-wide mean, a rank, a PCA basis or
    a quantile, the subset's values would differ from the full run's -- so this
    single equality rules out an entire family of leaks at once, including ones
    nobody thought to write a specific test for.
    """
    picks = list(subset)
    full = extract(X, metadata)
    partial = extract(X[picks], metadata.iloc[picks].reset_index(drop=True))

    numeric = [c for c in full.columns if pd.api.types.is_numeric_dtype(full[c])]
    expected = full.iloc[picks][numeric].to_numpy(dtype=float)
    actual = partial[numeric].to_numpy(dtype=float)

    if expected.shape != actual.shape:
        return AuditResult(
            name="features computed per epoch, not from global statistics",
            passed=False,
            detail=f"shape {actual.shape} != {expected.shape}",
        )

    largest = float(np.nanmax(np.abs(expected - actual))) if expected.size else 0.0
    return AuditResult(
        name="features computed per epoch, not from global statistics",
        passed=largest <= tolerance,
        detail=f"largest deviation {largest:.3e} over {len(picks)} epochs",
    )


def check_labels_unchanged(
    features: pd.DataFrame, index: pd.DataFrame, *, key: str = "epoch_uid"
) -> AuditResult:
    """Every epoch's class label must still equal the one in the epoch index."""
    merged = features[[key, "class_label"]].merge(
        index[[key, "class_label"]], on=key, suffixes=("_features", "_index")
    )
    if len(merged) != len(features):
        return AuditResult(
            name="labels unchanged from the epoch index",
            passed=False,
            detail=f"{len(merged)} of {len(features)} rows matched the index",
        )
    mismatched = int(
        (merged["class_label_features"] != merged["class_label_index"]).sum()
    )
    return AuditResult(
        name="labels unchanged from the epoch index",
        passed=mismatched == 0,
        detail=f"{mismatched} mismatch(es) across {len(merged)} epochs",
    )


def check_ids_traceable(
    features: pd.DataFrame, index: pd.DataFrame, *, key: str = "epoch_uid"
) -> AuditResult:
    """Epoch and subject identity must survive featurisation intact."""
    problems = []
    if key not in features.columns:
        problems.append(f"{key} missing from the feature table")
    elif features[key].duplicated().any():
        problems.append(f"{key} is not unique")
    elif set(features[key]) != set(index[key]):
        problems.append("epoch id set differs from the index")

    for column in ("subject_id", "run_id"):
        if column not in features.columns:
            problems.append(f"{column} missing from the feature table")
        elif features[column].isna().any():
            problems.append(f"{column} contains nulls")

    if not problems and "subject_id" in features.columns:
        merged = features[[key, "subject_id"]].merge(
            index[[key, "subject_id"]], on=key, suffixes=("_f", "_i")
        )
        if (merged["subject_id_f"] != merged["subject_id_i"]).any():
            problems.append("subject_id does not match the index for every epoch")

    return AuditResult(
        name="epoch and subject ids remain traceable",
        passed=not problems,
        detail="; ".join(problems) or f"{len(features)} epochs traceable",
    )


def check_flag_not_a_feature(
    feature_columns: Sequence[str], forbidden: Sequence[str] = ("artifact_flag", "peak_to_peak")
) -> AuditResult:
    """The 150 uV QC flag must not be used as a classification input."""
    leaked = [
        c for c in feature_columns if any(bad in c.lower() for bad in forbidden)
    ]
    return AuditResult(
        name="artifact flag is not a model feature",
        passed=not leaked,
        detail=f"found {leaked}" if leaked else f"{len(feature_columns)} features clean",
    )
