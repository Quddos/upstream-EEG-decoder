"""Phase 2B-2F -- run the subject-independent baseline study.

    python -m src.cli.phase2_baselines --data-root . --json data/metadata/phase2_summary.json

Writes:

    data/metadata/phase2_features.csv      one row per epoch, features + identity
    data/metadata/phase2_fold_results.csv  one row per (condition, held-out fold)
    data/metadata/phase2_summary.json      aggregates, permutation nulls, provenance

No model is tuned, no epoch is dropped, no threshold is applied, and every split
is by subject (or, for the clearly-labelled within-subject diagnostic, by run).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.cli._common import setup_logging
from src.config import DatasetConfig
from src.data.physionet_spec import LEFT_HAND_IMAGERY, REST, RIGHT_HAND_IMAGERY
from src.eval.baselines import MODELS, REAL_MODELS
from src.eval.protocol import (
    permutation_null,
    run_losocv,
    run_within_subject_cv,
    summarise,
)
from src.features.extract import CHANNEL_SETS, extract_features

#: The two classification problems the brief asks for.
TASKS: Dict[str, Sequence[str]] = {
    "left_vs_right": (LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY),
    "left_vs_right_vs_rest": (LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY, REST),
}

#: Feature families evaluated separately, to test whether scale-free features
#: transfer across subjects better than absolute log power (see extract.py).
FAMILIES: Dict[str, Optional[Sequence[str]]] = {
    "all": None,
    "absolute": ("absolute",),
    "relative": ("relative",),
}

#: The condition the permutation null and the headline numbers use.
PRIMARY_CHANNEL_SET = "roi21"
PRIMARY_FAMILY = "all"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 subject-independent baselines")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--permutations",
        type=int,
        default=200,
        help="label permutations for the empirical null (0 disables)",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    config = DatasetConfig()
    out_dir = args.out_dir or (config.paths.root / "data" / "metadata")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 2A: features -------------------------------------------------------
    table = extract_features(
        config, data_root=str(args.data_root) if args.data_root else None
    )
    features_path = out_dir / "phase2_features.csv"
    table.frame.to_csv(features_path, index=False)
    print(f"features : {len(table.frame)} epochs x {len(table.feature_columns)} features")
    print(f"           -> {features_path}")

    frame = table.frame
    all_results: List = []

    # --- 2B / 2D: LOSOCV across tasks, channel sets and feature families ----
    for task, classes in TASKS.items():
        mask = frame["class_label"].isin(list(classes)).to_numpy()
        y = frame.loc[mask, "class_label"].to_numpy()
        groups = frame.loc[mask, "subject"].to_numpy()
        subject_ids = frame.loc[mask, "subject_id"].to_numpy()
        runs = frame.loc[mask, "run"].to_numpy()

        for channel_set in CHANNEL_SETS:
            for family, family_filter in FAMILIES.items():
                columns = table.columns_for(
                    channel_set=channel_set, families=family_filter
                )
                if not columns:
                    continue
                X = frame.loc[mask, list(columns)].to_numpy(dtype=np.float64)

                all_results.extend(
                    run_losocv(
                        X,
                        y,
                        groups,
                        subject_ids,
                        task=task,
                        channel_set=channel_set,
                        feature_family=family,
                        models=list(MODELS),
                        seed=config.seed,
                    )
                )

        # within-subject diagnostic, primary channel set only
        columns = table.columns_for(channel_set=PRIMARY_CHANNEL_SET)
        X = frame.loc[mask, list(columns)].to_numpy(dtype=np.float64)
        within = run_within_subject_cv(
            X,
            y,
            groups,
            subject_ids,
            runs,
            task=task,
            channel_set=f"{PRIMARY_CHANNEL_SET}_within_subject",
            models=list(MODELS),
            seed=config.seed,
        )
        all_results.extend(within)

    fold_frame = pd.DataFrame([r.to_row() for r in all_results])
    fold_path = out_dir / "phase2_fold_results.csv"
    fold_frame.to_csv(fold_path, index=False)
    print(f"folds    : {len(fold_frame)} rows -> {fold_path}")

    summary = summarise(all_results)

    # --- permutation null, primary condition only ---------------------------
    nulls: Dict[str, Dict[str, object]] = {}
    if args.permutations > 0:
        for task, classes in TASKS.items():
            mask = frame["class_label"].isin(list(classes)).to_numpy()
            y = frame.loc[mask, "class_label"].to_numpy()
            groups = frame.loc[mask, "subject"].to_numpy()
            subject_ids = frame.loc[mask, "subject_id"].to_numpy()
            columns = table.columns_for(
                channel_set=PRIMARY_CHANNEL_SET, families=FAMILIES[PRIMARY_FAMILY]
            )
            X = frame.loc[mask, list(columns)].to_numpy(dtype=np.float64)

            for model in REAL_MODELS:
                observed, null, p_value = permutation_null(
                    X,
                    y,
                    groups,
                    subject_ids,
                    model=model,
                    seed=config.seed,
                    n_permutations=args.permutations,
                )
                nulls[f"{task}/{model}"] = {
                    "task": task,
                    "model": model,
                    "channel_set": PRIMARY_CHANNEL_SET,
                    "observed_pooled_accuracy": observed,
                    "null_mean": float(null.mean()),
                    "null_sd": float(null.std(ddof=1)),
                    "null_p95": float(np.percentile(null, 95)),
                    "p_value": p_value,
                    "n_permutations": int(args.permutations),
                }
                print(
                    f"null     : {task:24s} {model:20s} "
                    f"observed={observed:.3f} null={null.mean():.3f}"
                    f"+-{null.std(ddof=1):.3f} p={p_value:.4f}"
                )

    # --- report --------------------------------------------------------------
    print()
    for task in TASKS:
        subset = summary[
            (summary["task"] == task)
            & (summary["channel_set"] == PRIMARY_CHANNEL_SET)
            & (summary["feature_family"] == PRIMARY_FAMILY)
        ]
        print(f"--- {task} ({PRIMARY_CHANNEL_SET}, all features), LOSOCV ---")
        print(
            subset[
                ["model", "accuracy_mean", "accuracy_std", "accuracy_min",
                 "accuracy_max", "balanced_accuracy_mean", "macro_f1_mean", "roc_auc_mean"]
            ].to_string(index=False, float_format=lambda v: "%.3f" % v)
        )
        print()

    payload = {
        "n_epochs": int(len(frame)),
        "n_features": int(len(table.feature_columns)),
        "n_subjects": int(frame["subject_id"].nunique()),
        "channels": list(table.channels),
        "sampling_rate": table.sfreq,
        "tasks": {k: list(v) for k, v in TASKS.items()},
        "channel_sets": {
            k: (list(v) if v is not None else list(table.channels))
            for k, v in CHANNEL_SETS.items()
        },
        "feature_counts": {
            f"{cs}/{fam}": len(table.columns_for(channel_set=cs, families=filt))
            for cs in CHANNEL_SETS
            for fam, filt in FAMILIES.items()
        },
        "models": list(MODELS),
        "seed": config.seed,
        "config_fingerprint": config.fingerprint,
        "protocol": "LeaveOneGroupOut over subject; scaler fitted inside each fold",
        "summary": summary.to_dict(orient="records"),
        "permutation_nulls": nulls,
        "notes": [
            "Features are computed from un-normalised volts; the default "
            "zscore_per_channel would erase between-channel power contrasts.",
            "No hyperparameter was tuned. No epoch was excluded. The 150 uV "
            "artifact flag is not a feature and was never used as a rule.",
            "Within-subject rows (channel_set ending _within_subject) are a "
            "diagnostic, not a generalisation estimate.",
        ],
    }
    json_path = args.json or (out_dir / "phase2_summary.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"summary  : {json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
