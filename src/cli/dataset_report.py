"""Phase 1.7 -- validate and describe the five-subject motor-imagery dataset.

    python -m src.cli.dataset_report --subjects dev

Read-only. It loads what ``build_dataset`` wrote and answers the Phase 1.7
questions: how many recordings, subjects and epochs; the class balance; epochs per
subject and per run; artifact-flag rates by subject, run and class; and the compact
per-recording validation table.

It also runs the checks the brief says must **stop** the phase rather than be
quietly repaired:

* every requested subject x run present, and no forbidden run included;
* identity columns (subject, run, epoch, class, event code, onset, source) present
  and populated on every epoch;
* the T0/T1/T2 -> class mapping resolved through ``event_map_for_run`` and
  consistent across R04, R08 and R12;
* per-recording event counts within the expected shape.

Any failure is reported and the exit code is non-zero. Nothing is repaired, no
epoch is excluded, and the artifact flag is never used as a filter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.cli._common import setup_logging
from src.config import DatasetConfig, parse_runs, parse_subjects
from src.data.physionet_spec import (
    LEFT_HAND_IMAGERY,
    PHASE1_CLASSES,
    REST,
    RIGHT_HAND_IMAGERY,
    SENSORIMOTOR_ROI,
    SENSORIMOTOR_ROI_NAME,
    UNILATERAL_IMAGERY_RUNS,
    event_map_for_run,
    roi_indices,
)

#: Columns the brief requires every epoch to retain, mapped to the pipeline's names.
IDENTITY_COLUMNS: Dict[str, str] = {
    "subject_id": "subject_id",
    "run_id": "run_id",
    "epoch_id": "epoch_uid",
    "class": "class_label",
    "event_code": "event_code",
    "event_onset": "onset_time",
    "source": "source",
}

#: Runs that are explicitly not part of this phase.
EXCLUDED_RUNS: Tuple[int, ...] = (3, 5, 6, 7, 9, 10, 11, 13, 14)


def _fmt(frame: pd.DataFrame, floatfmt: str = "%.1f") -> str:
    return frame.to_string(index=False, float_format=lambda v: floatfmt % v)


def check_dataset(
    index: pd.DataFrame, config: DatasetConfig, channels: Sequence[str]
) -> List[str]:
    """Every stop-the-phase check. Returns a list of problems; empty means clean."""
    problems: List[str] = []

    # --- identity columns ----------------------------------------------------
    # Structural columns the later checks themselves depend on.  ``subject`` and
    # ``run`` are the integer forms used for coverage and event-map lookups.
    required_columns = set(IDENTITY_COLUMNS.values()) | {"subject", "run"}
    absent = sorted(c for c in required_columns if c not in index.columns)

    for required, actual in IDENTITY_COLUMNS.items():
        if actual not in index.columns:
            problems.append(f"identity column {required!r} ({actual}) is missing")
        elif index[actual].isna().any():
            n = int(index[actual].isna().sum())
            problems.append(f"identity column {required!r} is null in {n} epoch(s)")
    for column in ("subject", "run"):
        if column not in index.columns:
            problems.append(f"structural column {column!r} is missing")

    if absent:
        # Stop here rather than pressing on: every remaining check groups or joins
        # on these columns, so continuing would raise a KeyError and hide the
        # actual finding behind a traceback -- precisely when the dataset is
        # malformed and the report matters most.
        problems.append(
            f"skipping the remaining structural checks: they require {absent}"
        )
        return problems

    if index["epoch_uid"].duplicated().any():
        problems.append("epoch_uid is not unique across the dataset")

    # --- coverage: every requested subject x run present ---------------------
    present = set(zip(index["subject"], index["run"]))
    for subject in config.subjects:
        for run in config.runs:
            if (subject, run) not in present:
                problems.append(f"S{subject:03d} R{run:02d} produced no epochs")

    # --- no forbidden run leaked in ------------------------------------------
    leaked = sorted(set(index["run"]) & set(EXCLUDED_RUNS))
    if leaked:
        problems.append(f"runs excluded by the brief are present in the dataset: {leaked}")
    unexpected = sorted(set(index["run"]) - set(config.runs))
    if unexpected:
        problems.append(f"unexpected runs in the dataset: {unexpected}")

    # --- event mapping, resolved per run through the spec --------------------
    # The point of this check is that the mapping is *looked up*, not assumed:
    # T1 means left fist in runs 4/8/12 and both fists in 6/10/14, so a global
    # rule would mislabel half the database while every count still looked right.
    for run in sorted(set(index["run"])):
        run_map = event_map_for_run(run)
        rows = index[index["run"] == run]
        for code, expected_label in run_map.items():
            actual = set(rows.loc[rows["event_code"] == code, "class_label"])
            if not actual:
                continue
            if actual != {expected_label}:
                problems.append(
                    f"R{run:02d}: event code {code} maps to {sorted(actual)} in the "
                    f"dataset but event_map_for_run says {expected_label!r}"
                )

    # --- cross-run label consistency -----------------------------------------
    maps = {run: event_map_for_run(run) for run in sorted(set(index["run"]))}
    distinct = {tuple(sorted(m.items())) for m in maps.values()}
    if len(distinct) > 1:
        problems.append(
            f"event maps differ across the imagery runs, which they must not: {maps}"
        )

    # --- per-recording event counts ------------------------------------------
    per_recording = (
        index.groupby(["subject_id", "run_id"])["class_label"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for label in PHASE1_CLASSES:
        if label not in per_recording.columns:
            problems.append(f"class {label} is absent from the entire dataset")
            continue
        empty = per_recording.index[per_recording[label] == 0].tolist()
        if empty:
            problems.append(f"class {label} missing from recording(s): {empty}")

    # --- montage -------------------------------------------------------------
    if len(channels) != 64:
        problems.append(f"expected 64 channels, dataset has {len(channels)}")
    try:
        roi_indices(channels, SENSORIMOTOR_ROI)
    except ValueError as exc:
        problems.append(f"sensorimotor ROI cannot be resolved: {exc}")

    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1.7 dataset validation and description (read-only)"
    )
    parser.add_argument("--subjects", default="dev")
    parser.add_argument("--runs", default="imagery")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="directory holding data/processed and data/metadata, if not the project root",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    config = DatasetConfig(
        subjects=parse_subjects(args.subjects), runs=parse_runs(args.runs)
    )
    root = Path(args.data_root).resolve() if args.data_root else config.paths.root
    metadata_dir = root / "data" / "metadata"
    processed_dir = root / "data" / "processed"

    index = pd.read_csv(metadata_dir / "epoch_index.csv")
    recordings = pd.read_csv(metadata_dir / "recordings.csv")
    summary = json.loads((metadata_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    index = index[
        index["subject"].isin(config.subjects) & index["run"].isin(config.runs)
    ].reset_index(drop=True)

    channels = list(summary.get("channels", []))

    out: List[str] = []
    out.append("=" * 78)
    out.append("PHASE 1.7 -- FIVE-SUBJECT MOTOR-IMAGERY DATASET")
    out.append("=" * 78)
    out.append(f"source            : {summary.get('source')} "
               f"(synthetic={summary.get('synthetic')})")
    out.append(f"dataset           : {summary.get('dataset')}")
    out.append(f"built at          : {summary.get('generated_utc')}")
    out.append(f"config fingerprint: {summary.get('config_fingerprint')}")
    out.append("")

    # --- stop-the-phase checks ------------------------------------------------
    problems = check_dataset(index, config, channels)
    if problems:
        out.append("VALIDATION FAILED -- stopping, nothing repaired:")
        for problem in problems:
            out.append(f"  * {problem}")
        print("\n".join(out))
        return 1
    out.append("validation        : all structural checks passed")
    out.append("")

    # --- 1-6: headline counts -------------------------------------------------
    counts = index["class_label"].value_counts()
    n_recordings = index.groupby(["subject_id", "run_id"]).ngroups
    out.append("--- counts " + "-" * 66)
    out.append(f"  1. recordings          : {n_recordings}")
    out.append(f"  2. subjects            : {index['subject_id'].nunique()}")
    out.append(f"  3. total epochs        : {len(index)}")
    out.append(f"  4. LEFT_HAND_IMAGERY   : {int(counts.get(LEFT_HAND_IMAGERY, 0))}")
    out.append(f"  5. RIGHT_HAND_IMAGERY  : {int(counts.get(RIGHT_HAND_IMAGERY, 0))}")
    out.append(f"  6. REST                : {int(counts.get(REST, 0))}")
    out.append("")

    # --- 7-8: epochs per subject and per run ---------------------------------
    per_subject = (
        index.groupby("subject_id")
        .agg(epochs=("epoch_uid", "size"), flagged=("artifact_flag", "sum"))
        .reset_index()
    )
    per_subject["pct_flagged"] = 100.0 * per_subject["flagged"] / per_subject["epochs"]

    per_run = (
        index.groupby("run_id")
        .agg(epochs=("epoch_uid", "size"), flagged=("artifact_flag", "sum"))
        .reset_index()
    )
    per_run["pct_flagged"] = 100.0 * per_run["flagged"] / per_run["epochs"]

    per_class = (
        index.groupby("class_label")
        .agg(epochs=("epoch_uid", "size"), flagged=("artifact_flag", "sum"))
        .reset_index()
    )
    per_class["pct_flagged"] = 100.0 * per_class["flagged"] / per_class["epochs"]

    out.append("--- 7. epochs per subject / 9. artifact rate per subject " + "-" * 21)
    out.append(_fmt(per_subject))
    out.append("")
    out.append("--- 8. epochs per run / 10. artifact rate per run " + "-" * 28)
    out.append(_fmt(per_run))
    out.append("")
    out.append("--- 11. artifact rate by class " + "-" * 47)
    out.append(_fmt(per_class))
    out.append("")

    # Phase 1.6 raised a possible class-linked contamination confound from a single
    # recording (S001 R04, n=23). With 15x the data it can finally be tested.
    from src.qc.quality import class_association, pairwise_class_contrasts

    flag_frame = pd.DataFrame(
        {"class": index["class_label"], "artifact_flag": index["artifact_flag"].astype(bool)}
    )
    by_class = class_association(flag_frame, n_permutations=20000, seed=config.seed)
    subject_frame = pd.DataFrame(
        {"class": index["subject_id"], "artifact_flag": index["artifact_flag"].astype(bool)}
    )
    by_subject = class_association(subject_frame, n_permutations=20000, seed=config.seed)

    out.append("--- is the flag associated with class, or with subject? " + "-" * 22)
    out.append(
        f"  by class   : Cramer's V={by_class.cramers_v:.4f}  "
        f"perm p={by_class.p_permutation:.4f}  "
        f"spread={by_class.max_rate_spread_pp:.1f} pp  -> {by_class.verdict()}"
    )
    out.append(
        f"  by subject : Cramer's V={by_subject.cramers_v:.4f}  "
        f"perm p={by_subject.p_permutation:.5f}  "
        f"spread={by_subject.max_rate_spread_pp:.1f} pp"
    )
    out.append("")
    out.append(_fmt(pairwise_class_contrasts(flag_frame), floatfmt="%.4f"))
    out.append("")

    # --- class balance per subject -------------------------------------------
    balance = (
        index.groupby("subject_id")["class_label"].value_counts().unstack(fill_value=0)
    )
    out.append("--- class balance per subject " + "-" * 48)
    out.append(balance.to_string())
    out.append("")

    # --- compact validation table --------------------------------------------
    table = (
        index.assign(one=1)
        .groupby(["subject_id", "run_id"])
        .apply(
            lambda g: pd.Series(
                {
                    "epochs": len(g),
                    "left": int((g["class_label"] == LEFT_HAND_IMAGERY).sum()),
                    "right": int((g["class_label"] == RIGHT_HAND_IMAGERY).sum()),
                    "rest": int((g["class_label"] == REST).sum()),
                    "artifact_flagged": int(g["artifact_flag"].sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    table["pct_flagged"] = 100.0 * table["artifact_flagged"] / table["epochs"]
    out.append("--- per-recording validation table " + "-" * 43)
    out.append(_fmt(table))
    out.append("")
    out.append(
        f"  totals: {len(table)} recordings, {int(table['epochs'].sum())} epochs, "
        f"{int(table['left'].sum())} left, {int(table['right'].sum())} right, "
        f"{int(table['rest'].sum())} rest, "
        f"{int(table['artifact_flagged'].sum())} flagged"
    )
    out.append("")

    # --- event mapping actually in force -------------------------------------
    out.append("--- event mapping (resolved per run, not assumed) " + "-" * 28)
    for run in sorted(set(index["run"])):
        mapping = event_map_for_run(run)
        rendered = ", ".join(f"{k} -> {v}" for k, v in sorted(mapping.items()))
        out.append(f"  R{run:02d}: {rendered}")
    out.append("  consistent across all three imagery runs: yes")

    # Independent cross-check against the .edf.event sidecars, when present.
    # This compares event *counts* only -- see count_sidecar_annotations.
    from src.data.download import event_filename
    from src.data.integrity import count_sidecar_annotations
    from src.data.physionet_spec import RunSpec

    raw_dir = root / "data" / "raw"
    sidecar_rows: List[Dict[str, object]] = []
    for subject in sorted(set(index["subject"])):
        for run in sorted(set(index["run"])):
            spec = RunSpec(subject=int(subject), run=int(run))
            path = raw_dir / spec.subject_id / event_filename(spec)
            if not path.exists():
                continue
            counts = count_sidecar_annotations(path)
            edf_counts = (
                recordings.loc[
                    (recordings["subject_id"] == spec.subject_id)
                    & (recordings["run_id"] == spec.run_id),
                    ["n_T0", "n_T1", "n_T2"],
                ]
                .iloc[0]
                .to_dict()
            )
            sidecar_rows.append(
                {
                    "recording": f"{spec.subject_id}{spec.run_id}",
                    "sidecar_T0": counts["T0"],
                    "sidecar_T1": counts["T1"],
                    "sidecar_T2": counts["T2"],
                    "edf_T0": int(edf_counts["n_T0"]),
                    "edf_T1": int(edf_counts["n_T1"]),
                    "edf_T2": int(edf_counts["n_T2"]),
                    "agree": (
                        counts["T0"] == int(edf_counts["n_T0"])
                        and counts["T1"] == int(edf_counts["n_T1"])
                        and counts["T2"] == int(edf_counts["n_T2"])
                    ),
                }
            )

    if sidecar_rows:
        sidecars = pd.DataFrame(sidecar_rows)
        agreeing = int(sidecars["agree"].sum())
        out.append(
            f"  .edf.event sidecar cross-check: {agreeing}/{len(sidecars)} recordings "
            f"agree with the EDF+ annotation channel on T0/T1/T2 counts"
        )
        if agreeing != len(sidecars):
            out.append("  DISAGREEMENTS:")
            out.append(_fmt(sidecars[~sidecars["agree"]]))
        out.append("  (counts only -- this is a substring scan, not a WFDB parse, so it")
        out.append("   says nothing about event onsets)")
    else:
        out.append("  .edf.event sidecars: not present (run src.cli.verify_dataset "
                   "--fetch-events)")
    out.append("")

    # --- ROI ------------------------------------------------------------------
    out.append("--- sensorimotor ROI " + "-" * 57)
    out.append(f"  name     : {SENSORIMOTOR_ROI_NAME}")
    out.append(f"  channels : {len(SENSORIMOTOR_ROI)} of {len(channels)} retained on disk")
    out.append(f"  members  : {' '.join(SENSORIMOTOR_ROI)}")
    out.append(f"  indices  : {roi_indices(channels, SENSORIMOTOR_ROI)}")
    out.append("  full 64-channel arrays are unchanged; the ROI is applied at model")
    out.append("  input time via EEGDataset.sensorimotor_roi(), so the 64-vs-21")
    out.append("  ablation stays available.")
    out.append("")

    # --- archives, per-subject amplitude, and ROI cleanliness -----------------
    # The per-subject artifact rates above range from 10% to 100%, which is a
    # statement about amplitude, not about data quality per se. These two tables
    # say where that variation comes from and whether it reaches the ROI.
    from src.qc.quality import measure_amplitudes

    threshold_v = config.preprocessing.reject_log_threshold_v
    archive_rows: List[Dict[str, object]] = []
    amplitude_rows: List[Dict[str, object]] = []
    roi_rows: List[Dict[str, object]] = []

    for subject_id in sorted(index["subject_id"].unique()):
        archive = processed_dir / f"{subject_id}.npz"
        if not archive.exists():
            out.append(f"  {subject_id}: MISSING {archive}")
            continue
        with np.load(archive, allow_pickle=False) as bundle:
            X = bundle["X"].astype(np.float64)
            file_channels = [str(c) for c in bundle["channels"]]

        archive_rows.append(
            {
                "archive": f"{subject_id}.npz",
                "epochs": X.shape[0],
                "channels": X.shape[1],
                "samples": X.shape[2],
                "MB": archive.stat().st_size / 1e6,
            }
        )

        measures = measure_amplitudes(X, file_channels, threshold_v)
        p2p = measures.epoch_p2p_max
        amplitude_rows.append(
            {
                "subject_id": subject_id,
                "min": p2p.min(),
                "median": float(np.median(p2p)),
                "p95": float(np.percentile(p2p, 95)),
                "max": p2p.max(),
                "pct_flagged": 100.0 * measures.artifact_flag.mean(),
            }
        )

        picks = list(roi_indices(file_channels, SENSORIMOTOR_ROI))
        roi_p2p = measures.per_channel_p2p[:, picks]
        roi_over = int((roi_p2p > measures.threshold_uv).sum())
        total_over = int((measures.per_channel_p2p > measures.threshold_uv).sum())
        roi_rows.append(
            {
                "subject_id": subject_id,
                "roi_median": float(np.median(roi_p2p)),
                "roi_max": float(roi_p2p.max()),
                "roi_crossings": roi_over,
                "roi_cells": int(roi_p2p.size),
                "outside_roi_crossings": total_over - roi_over,
            }
        )

    out.append("--- processed archives " + "-" * 55)
    out.append(_fmt(pd.DataFrame(archive_rows)))
    out.append("")

    amplitude = pd.DataFrame(amplitude_rows)
    out.append("--- epoch peak-to-peak by subject (µV, threshold 150) " + "-" * 24)
    out.append(_fmt(amplitude))
    out.append(
        f"  between-subject median amplitude ratio: "
        f"{amplitude['median'].max() / amplitude['median'].min():.2f}x  "
        f"({amplitude['median'].min():.0f} -> {amplitude['median'].max():.0f} µV)"
    )
    out.append("")

    roi_table = pd.DataFrame(roi_rows)
    roi_total = int(roi_table["roi_crossings"].sum())
    roi_cells = int(roi_table["roi_cells"].sum())
    outside_total = int(roi_table["outside_roi_crossings"].sum())
    out.append("--- threshold crossings inside vs outside the ROI " + "-" * 28)
    out.append(_fmt(roi_table))
    out.append(
        f"  ROI     : {roi_total} crossings in {roi_cells} channel-epochs "
        f"({100.0 * roi_total / roi_cells:.2f}%)"
    )
    out.append(
        f"  outside : {outside_total} crossings "
        f"({100.0 * outside_total / (roi_total + outside_total):.1f}% of all crossings)"
    )
    out.append("")
    out.append("No epoch was excluded. The artifact flag was not used as a filter.")

    report = "\n".join(out)
    print(report)

    if args.json:
        payload = {
            "n_recordings": int(n_recordings),
            "n_subjects": int(index["subject_id"].nunique()),
            "n_epochs": int(len(index)),
            "class_counts": {k: int(v) for k, v in counts.items()},
            "per_subject": per_subject.to_dict(orient="records"),
            "per_run": per_run.to_dict(orient="records"),
            "per_class": per_class.to_dict(orient="records"),
            "class_balance_per_subject": balance.to_dict(orient="index"),
            "validation_table": table.to_dict(orient="records"),
            "event_maps": {
                f"R{run:02d}": event_map_for_run(run) for run in sorted(set(index["run"]))
            },
            "roi": {
                "name": SENSORIMOTOR_ROI_NAME,
                "channels": list(SENSORIMOTOR_ROI),
                "indices": list(roi_indices(channels, SENSORIMOTOR_ROI)),
                "montage_size": len(channels),
            },
            "problems": problems,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
