"""Phase 3 -- run the calibration-budget study.

    python -m src.cli.phase3_calibration --data-root . --repeats 20

Writes:

    data/metadata/phase3_budget_results.csv   one row per (subject, repeat, strategy, budget)
    data/metadata/phase3_reliability.csv      per-epoch confidence/outcome pairs
    data/metadata/phase3_summary.json         aggregates, gains, shift reduction

Phase 1 preprocessing and Phase 2 feature definitions are imported unchanged; this
entry point adds no features and edits no config.
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
from src.eval.calibration import LABEL_FREE_STRATEGIES, STRATEGIES
from src.eval.personalization import (
    DEFAULT_BUDGETS,
    make_split,
    per_subject_gain,
    run_personalization,
    summarise_budgets,
)
from src.features.extract import extract_features

TASKS: Dict[str, Sequence[str]] = {
    "left_vs_right": (LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY),
    "left_vs_right_vs_rest": (LEFT_HAND_IMAGERY, RIGHT_HAND_IMAGERY, REST),
}

#: Phase 2's primary channel set, carried over unchanged.
CHANNEL_SET = "roi21"

#: Models with a probability output, which the confidence metrics require.
MODELS_TO_RUN = ("logistic_regression", "random_forest")


def subject_shift_reduction(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    budgets: Sequence[int],
    seed: int,
    n_repeats: int,
) -> pd.DataFrame:
    """How much of the Phase 2 subject shift a given calibration budget removes.

    Phase 2 measured, for each feature, the between-subject spread of its mean
    against the within-subject LEFT/RIGHT separation it is meant to detect --
    1.03x for ``lateralization_mu`` up to 10.43x for ``C4_log_mu``. This repeats
    that measurement after subtracting each subject's own mean **estimated from
    only ``budget`` calibration epochs**, which is what the recentring strategy
    actually has available.

    The residual is the part of the shift a finite calibration set cannot remove.
    With an unlimited budget the ratio would fall to zero by construction; the
    interesting question is how fast it falls.
    """
    task = frame[frame["class_label"] != REST]
    rows: List[Dict[str, object]] = []

    for budget in budgets:
        for repeat in range(n_repeats):
            centred = []
            for subject_id, block in task.groupby("subject_id"):
                values = block[list(feature_columns)].to_numpy(dtype=float)
                labels = block["class_label"].to_numpy()
                if budget == 0:
                    centred.append(pd.DataFrame(values, columns=list(feature_columns)).assign(
                        subject_id=subject_id, class_label=labels))
                    continue
                split = make_split(
                    labels,
                    subject_id=str(subject_id),
                    repeat=repeat,
                    pool_size=budget,
                    seed=seed + repeat * 1009,
                )
                # Estimated from calibration epochs only -- never from the epochs
                # whose spread is then measured on the evaluation side.
                centre = values[split.budget_indices(budget)].mean(axis=0)
                adjusted = values - centre
                centred.append(pd.DataFrame(adjusted, columns=list(feature_columns)).assign(
                    subject_id=subject_id, class_label=labels))

            combined = pd.concat(centred, ignore_index=True)
            for column in feature_columns:
                subject_means = combined.groupby("subject_id")[column].mean()
                between = float(subject_means.std(ddof=1))
                separations = []
                for _, block in combined.groupby("subject_id"):
                    left = block.loc[block["class_label"] == LEFT_HAND_IMAGERY, column].mean()
                    right = block.loc[block["class_label"] == RIGHT_HAND_IMAGERY, column].mean()
                    separations.append(abs(right - left))
                signal = float(np.mean(separations))
                rows.append(
                    {
                        "budget": budget,
                        "repeat": repeat,
                        "feature": column,
                        "between_subject_sd": between,
                        "class_separation": signal,
                        "nuisance_to_signal": between / signal if signal > 0 else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 calibration-budget study")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS)
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS_TO_RUN))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    config = DatasetConfig()
    out_dir = args.out_dir or (config.paths.root / "data" / "metadata")
    out_dir.mkdir(parents=True, exist_ok=True)

    table = extract_features(
        config, data_root=str(args.data_root) if args.data_root else None
    )
    frame = table.frame
    columns = list(table.columns_for(channel_set=CHANNEL_SET))
    print(f"features : {len(frame)} epochs x {len(columns)} ({CHANNEL_SET})")

    all_results: List = []
    all_reliability: List[Dict[str, object]] = []

    for task, classes in TASKS.items():
        mask = frame["class_label"].isin(list(classes)).to_numpy()
        X = frame.loc[mask, columns].to_numpy(dtype=np.float64)
        y = frame.loc[mask, "class_label"].to_numpy()
        groups = frame.loc[mask, "subject"].to_numpy()
        subject_ids = frame.loc[mask, "subject_id"].to_numpy()
        rest_label = REST if REST in classes else None

        for model in args.models:
            results, reliability = run_personalization(
                X,
                y,
                groups,
                subject_ids,
                task=task,
                model=model,
                seed=config.seed,
                budgets=args.budgets,
                n_repeats=args.repeats,
                rest_label=rest_label,
            )
            all_results.extend(results)
            all_reliability.extend(reliability)
            eval_n = int(np.mean([r.n_evaluation for r in results]))
            print(f"  {task:24s} {model:20s} {len(results):5d} runs, "
                  f"{eval_n} eval epochs/subject")

    results_frame = pd.DataFrame([r.to_row() for r in all_results])
    results_path = out_dir / "phase3_budget_results.csv"
    results_frame.to_csv(results_path, index=False)

    reliability_frame = pd.DataFrame(all_reliability)
    reliability_path = out_dir / "phase3_reliability.csv"
    reliability_frame.to_csv(reliability_path, index=False)

    summary = summarise_budgets(all_results)

    # --- subject-shift reduction (figure 24) --------------------------------
    shift_features = [
        c for c in ("lateralization_mu", "erd_C3_vs_roi_mu", "C3_log_mu", "C4_log_mu")
        if c in frame.columns
    ]
    shift = subject_shift_reduction(
        frame, shift_features, args.budgets, config.seed, min(args.repeats, 10)
    )
    shift_summary = (
        shift.groupby(["budget", "feature"])["nuisance_to_signal"]
        .agg(["mean", "std"])
        .reset_index()
    )
    shift_path = out_dir / "phase3_subject_shift.csv"
    shift.to_csv(shift_path, index=False)

    # --- console report ------------------------------------------------------
    print()
    for task in TASKS:
        for model in args.models:
            subset = summary[(summary["task"] == task) & (summary["model"] == model)]
            if subset.empty:
                continue
            pivot = subset.pivot_table(
                index="strategy", columns="budget", values="balanced_accuracy_mean"
            )
            print(f"--- {task} / {model}: balanced accuracy by budget ---")
            print(pivot.round(3).to_string())
            print()

    print("--- subject-shift reduction (nuisance-to-signal ratio) ---")
    print(
        shift_summary.pivot_table(index="feature", columns="budget", values="mean")
        .round(2).to_string()
    )
    print()

    # --- gains, per subject --------------------------------------------------
    gains: Dict[str, object] = {}
    best_budget = max(args.budgets)
    for task in TASKS:
        for model in args.models:
            for strategy in STRATEGIES:
                subset = [
                    r for r in all_results
                    if r.task == task and r.model == model
                ]
                if not subset:
                    continue
                gain = per_subject_gain(subset, strategy=strategy, budget=best_budget)
                if gain.empty:
                    continue
                gains[f"{task}/{model}/{strategy}"] = gain.to_dict(orient="records")

    payload = {
        "channel_set": CHANNEL_SET,
        "n_features": len(columns),
        "budgets": list(args.budgets),
        "n_repeats": args.repeats,
        "seed": config.seed,
        "models": list(args.models),
        "strategies": list(STRATEGIES),
        "label_free_strategies": list(LABEL_FREE_STRATEGIES),
        "activation_threshold": 0.7,
        "summary": summary.to_dict(orient="records"),
        "subject_shift": shift_summary.to_dict(orient="records"),
        "per_subject_gain_at_max_budget": gains,
        "notes": [
            "Evaluation epochs are fixed across budgets within a repeat, so "
            "budget comparisons are paired.",
            "Calibration epochs are drawn only from the held-out subject's "
            "calibration pool and are never evaluated on.",
            "Latency is wall-clock and therefore the one non-reproducible column.",
            "No preprocessing or Phase 2 feature definition was changed.",
        ],
    }
    json_path = out_dir / "phase3_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"results  : {results_path}")
    print(f"reliability: {reliability_path}")
    print(f"shift    : {shift_path}")
    print(f"summary  : {json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
