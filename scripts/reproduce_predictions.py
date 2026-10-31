#!/usr/bin/env python
"""Re-execute the frozen decoder and compare against the deposited predictions.

This requires the processed epoch archives under ``data/processed/`` (40 files,
``S001.npz`` ... ``S040.npz``), which are built from the PhysioNet recordings by

    python -m src.cli.build_dataset --subjects all --source physionet

The recordings themselves are not redistributed here; see the README.

What this script does
---------------------
1. Prints the configuration it is about to run, read from the source rather than
   from documentation.
2. Calls ``src.encoder.paper1.encode_cohort``, which re-executes the frozen
   leave-one-subject-out protocol and keeps the per-epoch posterior that the
   original Paper-1 run computed and discarded. The call verifies itself against
   the published per-subject balanced accuracies and aborts on any disagreement
   above 1e-9.
3. Compares all 1,800 rows against
   ``data/metadata/phase6b/phase6b_neurostates.csv``: epoch identity, subject,
   run, true label, decoded side, and the two class posteriors.
4. Recomputes accuracy, balanced accuracy, macro F1 and the confusion matrix
   from both the fresh run and the deposited predictions, and compares each
   against the reported values.

    python scripts/reproduce_predictions.py
    python scripts/reproduce_predictions.py --json out.json

Nothing under ``data/`` is written.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix,
                             precision_recall_fscore_support)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import DatasetConfig                                # noqa: E402
from src.data.physionet_spec import SENSORIMOTOR_ROI                 # noqa: E402
from src.encoder.paper1 import (PRIMARY_CHANNEL_SET, PRIMARY_CLASSES,  # noqa: E402
                                PRIMARY_MODEL, PRIMARY_REPRESENTATION,
                                PRIMARY_TASK, encode_cohort)
from src.eval.baselines import MODELS                                # noqa: E402

LABELS = ["LEFT", "RIGHT"]
REPORTED = {"accuracy": 0.5994, "balanced_accuracy": 0.5995,
            "macro_f1": 0.5994,
            "confusion_matrix": [[535, 370], [351, 544]]}
#: lbfgs may land on a different final bit between runs and platforms. The
#: original run recorded a worst per-subject disagreement of 1.11e-16; this is a
#: generous bound on the same phenomenon, and the *predictions* are compared
#: exactly rather than within a tolerance.
POSTERIOR_TOL = 1e-12


def banner(text: str) -> None:
    print()
    print("=" * 74)
    print(text)
    print("=" * 74)


def metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_precision": float(p), "macro_recall": float(r),
            "macro_f1": float(f),
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=LABELS).tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=pathlib.Path, default=None,
                    help="repository root (default: this script's parent)")
    ap.add_argument("--data-root", type=str, default=None,
                    help="project root holding data/processed and "
                         "data/metadata, if not --root. This is the same "
                         "meaning the frozen CLIs give --data-root: the "
                         "parent of data/, not data/ itself.")
    ap.add_argument("--json", type=pathlib.Path, default=None,
                    help="also write the full report as JSON")
    args = ap.parse_args()
    root = (args.root or pathlib.Path(__file__).resolve().parents[1]).resolve()

    report: dict = {}

    # ----------------------------------------------- 1. declared configuration
    banner("1. configuration, read from the source")
    cfg = DatasetConfig()
    pre = cfg.preprocessing
    est = MODELS[PRIMARY_MODEL](cfg.seed)
    lr = est.named_steps["model"]
    conf = {
        "band_pass_hz": [pre.l_freq, pre.h_freq],
        "filter_method": pre.filter_method,
        "filter_phase": pre.filter_phase,
        "notch_enabled": pre.notch_enabled,
        "common_average_reference": pre.common_average_reference,
        "epoch_window_s": [pre.epoch_tmin, pre.epoch_tmax],
        "include_tmax": pre.include_tmax,
        "epoch_samples": pre.expected_epoch_samples,
        "normalisation_at_feature_time": "none (load with normalize=False)",
        "seed": cfg.seed,
        "roi_name": "sensorimotor_21",
        "roi_n_channels": len(SENSORIMOTOR_ROI),
        "roi_channels": list(SENSORIMOTOR_ROI),
        "task": PRIMARY_TASK,
        "channel_set": PRIMARY_CHANNEL_SET,
        "model": PRIMARY_MODEL,
        "representation": PRIMARY_REPRESENTATION,
        "pipeline_steps": [n for n, _ in est.steps],
        "logistic_regression": {"C": lr.C, "solver": lr.solver,
                                "max_iter": lr.max_iter,
                                "effective_penalty": "l2 (scikit-learn default; "
                                                     "not passed explicitly)"},
        "protocol": "leave-one-subject-out",
    }
    for k, v in conf.items():
        if k == "roi_channels":
            print(f"  {k:32} {', '.join(v)}")
        else:
            print(f"  {k:32} {v}")
    report["configuration"] = conf

    # --------------------------------------------------- 2. re-execute the run
    banner("2. re-executing the frozen leave-one-subject-out protocol")
    data_root = (pathlib.Path(args.data_root).resolve() if args.data_root
                 else root)
    processed = data_root / "data" / "processed"
    n_npz = len(list(processed.glob("S*.npz"))) if processed.exists() else 0
    print(f"  processed archives found: {n_npz} in {processed}")
    if n_npz == 0:
        print("\n  Cannot re-execute: no processed epoch archives.", file=sys.stderr)
        print("  Build them first (see the README), or run", file=sys.stderr)
        print("      python scripts/metrics_from_deposited.py", file=sys.stderr)
        print("  which reproduces the reported metrics from the deposited "
              "predictions alone.", file=sys.stderr)
        return 3

    enc = encode_cohort(data_root=args.data_root, verify_against_paper1=True)
    fold = next(iter(enc.folds.values()))
    print(f"  posteriors               : {len(enc.posteriors)}")
    print(f"  subjects / folds         : {len(enc.subjects)} / {len(enc.folds)}")
    print(f"  features per fold        : {fold.n_features}")
    print(f"  train / test epochs      : {fold.n_train_epochs} / "
          f"{fold.n_test_epochs}")
    print(f"  dataset fingerprint      : {enc.dataset_fingerprint}")
    print(f"  self-verification passed : {enc.reproduction['verified']}")
    print(f"  bit-identical subjects   : "
          f"{enc.reproduction['bit_identical_subjects']}/"
          f"{enc.reproduction['n_subjects']}")
    print(f"  worst per-subject diff   : "
          f"{enc.reproduction['max_abs_difference']:.3e}")
    print(f"  subject leakage detected : {enc.audit['leakage_detected']}")
    report["reexecution"] = {
        "n_posteriors": len(enc.posteriors), "n_folds": len(enc.folds),
        "n_features": fold.n_features,
        "dataset_fingerprint": enc.dataset_fingerprint,
        "reproduction": dict(enc.reproduction),
        "subject_independence_audit": dict(enc.audit)}

    fresh = enc.to_frame().sort_values("epoch_uid").reset_index(drop=True)

    # ------------------------------------------ 3. compare against the deposit
    banner("3. row-by-row comparison against the deposited predictions")
    dep_path = root / "data/metadata/phase6b/phase6b_neurostates.csv"
    if not dep_path.exists():
        dep_path = data_root / "data/metadata/phase6b/phase6b_neurostates.csv"
    dep = pd.read_csv(dep_path).sort_values("epoch_uid").reset_index(drop=True)
    print(f"  deposited   : {len(dep)} rows from {dep_path.name}")
    print(f"  re-executed : {len(fresh)} rows")

    checks: dict = {}
    checks["row_count"] = len(fresh) == len(dep)
    for col in ("epoch_uid", "subject_id", "run_id", "true_label"):
        checks[col] = bool((fresh[col].values == dep[col].values).all())
    dep_side = dep.decoded_side.map({"LEFT": PRIMARY_CLASSES[0],
                                     "RIGHT": PRIMARY_CLASSES[1]}).values
    checks["predicted_label"] = bool(
        (fresh.predicted_label.values == dep_side).all())
    n_diff = int((fresh.predicted_label.values != dep_side).sum())

    for col, depcol in (("left_probability", "neuro.left_probability"),
                        ("right_probability", "neuro.right_probability"),
                        ("lateralisation", "neuro.lateralisation")):
        a = fresh[col].to_numpy(dtype=float)
        b = dep[depcol].to_numpy(dtype=float)
        worst = float(np.nanmax(np.abs(a - b)))
        checks[f"{col}_bit_identical"] = bool(np.array_equal(a, b))
        checks[f"{col}_max_abs_diff"] = worst
        checks[f"{col}_within_tolerance"] = worst <= POSTERIOR_TOL
        print(f"  {col:20} bit-identical={checks[f'{col}_bit_identical']}  "
              f"max|diff|={worst:.3e}  "
              f"within {POSTERIOR_TOL:.0e}={checks[f'{col}_within_tolerance']}")
    for name in ("epoch_uid", "subject_id", "run_id", "true_label",
                 "predicted_label"):
        print(f"  {name:20} identical={checks[name]}")
    print(f"\n  predictions differing: {n_diff} of {len(dep)}")
    report["row_comparison"] = checks
    report["n_predictions_differing"] = n_diff

    # ----------------------------------------------------------- 4. the metrics
    banner("4. metrics")
    short = {PRIMARY_CLASSES[0]: "LEFT", PRIMARY_CLASSES[1]: "RIGHT"}
    m_fresh = metrics([short[x] for x in fresh.true_label],
                      [short[x] for x in fresh.predicted_label])
    m_dep = metrics(dep.true_label.map({"LEFT_HAND_IMAGERY": "LEFT",
                                        "RIGHT_HAND_IMAGERY": "RIGHT"}).tolist(),
                    dep.decoded_side.tolist())
    print(f"  {'metric':20} {'re-executed':>13} {'deposited':>13} "
          f"{'reported':>10}  match")
    metric_ok = True
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        a, b, c = m_fresh[key], m_dep[key], REPORTED[key]
        ok = round(a, 4) == c and round(b, 4) == c
        metric_ok &= ok
        print(f"  {key:20} {a:13.6f} {b:13.6f} {c:10.4f}  "
              f"{'YES' if ok else 'NO'}")
    cm_ok = (m_fresh["confusion_matrix"] == REPORTED["confusion_matrix"]
             and m_dep["confusion_matrix"] == REPORTED["confusion_matrix"])
    print(f"  {'confusion matrix':20} {str(m_fresh['confusion_matrix'])} "
          f"(re-executed)")
    print(f"  {'':20} {str(m_dep['confusion_matrix'])} (deposited)")
    print(f"  {'':20} {str(REPORTED['confusion_matrix'])} (reported)   "
          f"{'YES' if cm_ok else 'NO'}")
    report["metrics"] = {"re_executed": m_fresh, "deposited": m_dep,
                         "reported": REPORTED, "confusion_matrix_match": cm_ok}

    # ----------------------------------------------------------- 5. the verdict
    banner("5. verdict")
    verdict = {
        "all_predictions_reproduce_exactly":
            bool(checks["row_count"] and n_diff == 0
                 and all(checks[k] for k in ("epoch_uid", "subject_id",
                                             "run_id", "true_label",
                                             "predicted_label"))),
        "posteriors_agree_within_float_tolerance":
            bool(all(checks[f"{c}_within_tolerance"]
                     for c in ("left_probability", "right_probability",
                               "lateralisation"))),
        "confusion_matrix_matches": bool(cm_ok),
        "reported_metrics_match": bool(metric_ok),
    }
    for k, v in verdict.items():
        print(f"  {'OK  ' if v else 'FAIL'}  {k}")
    report["verdict"] = verdict

    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=str),
                             encoding="utf-8")
        print(f"\n  report written to {args.json}")
    return 0 if all(verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
