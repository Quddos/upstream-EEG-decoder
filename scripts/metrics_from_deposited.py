#!/usr/bin/env python
"""Compute the reported decoder metrics from the deposited out-of-fold predictions.

Needs no EEG recordings and no processed arrays: it reads only
``data/metadata/phase6b/phase6b_neurostates.csv``, which holds one row per
held-out epoch with the true label and the leave-one-subject-out decoded side.

    python scripts/metrics_from_deposited.py

Expected output (the values reported in the manuscript):

    accuracy           0.5994
    balanced accuracy  0.5995
    macro F1           0.5994
    confusion matrix   [[535, 370], [351, 544]]

Precision, recall, F1 and the confusion matrix were not stored by the original
run; they are computed here from the stored predictions. The quantities that
*were* stored -- per-subject balanced accuracy, posterior margin, cue agreement
and usable-epoch fraction -- are cross-checked against
``phase6b_subjects.csv`` so that the reproduction can be trusted.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix,
                             precision_recall_fscore_support)

LABELS = ["LEFT", "RIGHT"]
TO_SIDE = {"LEFT_HAND_IMAGERY": "LEFT", "RIGHT_HAND_IMAGERY": "RIGHT"}
REPORTED = {"accuracy": 0.5994, "balanced_accuracy": 0.5995,
            "macro_f1": 0.5994,
            "confusion_matrix": [[535, 370], [351, 544]]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."),
                    help="repository root (default: the current directory)")
    args = ap.parse_args()

    base = args.root / "data" / "metadata" / "phase6b"
    states = base / "phase6b_neurostates.csv"
    subjects = base / "phase6b_subjects.csv"
    for p in (states, subjects):
        if not p.exists():
            print(f"missing deposited artifact: {p}", file=sys.stderr)
            return 2

    ns = pd.read_csv(states)
    y_true = ns["true_label"].map(TO_SIDE).to_numpy()
    y_pred = ns["decoded_side"].to_numpy()

    print("evaluation unit  : one held-out EEG epoch")
    print(f"epochs           : {len(ns)}")
    print(f"subjects         : {ns.subject_id.nunique()}")
    print(f"runs             : {', '.join(sorted(ns.run_id.unique()))}")
    counts = pd.Series(y_true).value_counts()
    print(f"class balance    : LEFT {counts.get('LEFT', 0)} / "
          f"RIGHT {counts.get('RIGHT', 0)}")

    print("\n--- pooled over all out-of-fold epochs (micro) ---")
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS,
                                                 zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print(f"accuracy          : {acc:.6f}  (reported {REPORTED['accuracy']})")
    print(f"balanced accuracy : {bal:.6f}  "
          f"(reported {REPORTED['balanced_accuracy']})")
    for i, lab in enumerate(LABELS):
        print(f"  {lab:<6} precision {p[i]:.4f}  recall {r[i]:.4f}  "
              f"F1 {f[i]:.4f}  n={s[i]}")
    print(f"macro             : precision {pm:.4f}  recall {rm:.4f}  "
          f"F1 {fm:.6f}  (reported {REPORTED['macro_f1']})")
    print("confusion matrix  : rows true (LEFT, RIGHT), "
          "columns predicted (LEFT, RIGHT)")
    print(f"                    {cm.tolist()}")

    print("\n--- per subject, averaged over the 40 folds (macro) ---")
    rows = []
    for sid, g in ns.groupby("subject_id"):
        t = g["true_label"].map(TO_SIDE).to_numpy()
        q = g["decoded_side"].to_numpy()
        _, _, f1m, _ = precision_recall_fscore_support(
            t, q, labels=LABELS, average="macro", zero_division=0)
        rows.append({"subject_id": sid,
                     "balanced_accuracy": balanced_accuracy_score(t, q),
                     "accuracy": accuracy_score(t, q),
                     "macro_f1": f1m,
                     "agreement_with_cue": float(g.decoder_agrees_with_cue.mean()),
                     "mean_margin": float(g["neuro.margin"].mean()),
                     "usable_fraction": float(g["reliability.usable"].mean())})
    per = pd.DataFrame(rows)
    print(f"balanced accuracy : mean {per.balanced_accuracy.mean():.4f}  "
          f"sd {per.balanced_accuracy.std(ddof=1):.4f}  "
          f"min {per.balanced_accuracy.min():.4f} "
          f"({per.loc[per.balanced_accuracy.idxmin(), 'subject_id']})  "
          f"max {per.balanced_accuracy.max():.4f} "
          f"({per.loc[per.balanced_accuracy.idxmax(), 'subject_id']})")
    print(f"macro F1          : mean {per.macro_f1.mean():.4f}  "
          f"sd {per.macro_f1.std(ddof=1):.4f}")
    print(f"at or above chance: {(per.balanced_accuracy >= 0.5).sum()}/"
          f"{len(per)} subjects")
    print(f"usable fraction   : mean {per.usable_fraction.mean():.4f}  "
          f"min {per.usable_fraction.min():.4f}  "
          f"max {per.usable_fraction.max():.4f}")

    print("\n--- cross-check against the values the original run DID store ---")
    rec = pd.read_csv(subjects)
    m = per.merge(rec[["subject_id", "paper1_balanced_accuracy", "mean_margin",
                       "agreement_with_cue", "usable_epoch_fraction"]],
                  on="subject_id", suffixes=("_recomputed", "_recorded"))
    deltas = {
        "balanced accuracy": (m.balanced_accuracy
                              - m.paper1_balanced_accuracy).abs().max(),
        "mean margin": (m.mean_margin_recomputed
                        - m.mean_margin_recorded).abs().max(),
        "cue agreement": (m.agreement_with_cue_recomputed
                          - m.agreement_with_cue_recorded).abs().max(),
        "usable fraction": (m.usable_fraction
                            - m.usable_epoch_fraction).abs().max(),
    }
    for k, v in deltas.items():
        print(f"  max |recomputed - recorded| {k:18}: {v:.3e}")

    print("\n--- verdict ---")
    ok = {
        "accuracy matches the reported value":
            round(acc, 4) == REPORTED["accuracy"],
        "balanced accuracy matches":
            round(bal, 4) == REPORTED["balanced_accuracy"],
        "macro F1 matches": round(fm, 4) == REPORTED["macro_f1"],
        "confusion matrix matches":
            cm.tolist() == REPORTED["confusion_matrix"],
        "stored per-subject values reproduce":
            bool(np.all([v < 1e-9 for v in deltas.values()])),
    }
    for k, v in ok.items():
        print(f"  {'OK  ' if v else 'FAIL'}  {k}")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
