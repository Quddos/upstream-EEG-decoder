"""Phase 6B -- run the frozen Paper 1 encoder into the Phase 6A digital twin.

Re-executes the frozen LOSOCV, verifies the reproduction against Paper 1's
published per-subject results, adapts every epoch into a ``NeuroState``, feeds
each subject's states through the unchanged twin, and writes the artifacts.

Artifacts under ``data/metadata/phase6b/``:

``phase6b_neurostates.csv``   one row per epoch: posterior, margin, entropy,
                              lateralisation, reliability verdict, provenance
``phase6b_subjects.csv``      one row per subject: decoder behaviour and the
                              twin's final latent state with uncertainty
``<subject>.states.jsonl``    twin state log for the report's case subjects
``phase6b_summary.json``      provenance, audit, reproduction, determinism

No model is trained, no hyperparameter is touched, no preprocessing is changed.

Usage
-----
    python -m src.cli.phase6b_encoder
    python -m src.cli.phase6b_encoder --cases S007 S037 S003
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.encoder import (
    PAPER1,
    PAPER1_BELOW_CHANCE,
    Paper1NeuroStateAdapter,
    encode_cohort,
    run_provenance,
    subject_summary,
)
from src.twin.interfaces import DecisionEngineView
from src.twin.serialization import write_states
from src.twin.states import LatentState
from src.twin.twin import DigitalTwin, TwinConfig

DEFAULT_OUT = Path("data/metadata/phase6b")
DEFAULT_FIGURES = Path("reports/figures/phase6b")

#: Subjects carried into the report as the three required cases, chosen from
#: the frozen Paper 1 outcome and named here so the choice is visible.
#:   S007 -- best subject in the cohort (0.932)
#:   S037 -- exactly at chance (0.500); the confidently-wrong decoder
#:   S003 -- worst subject in the cohort (0.398), in the below-chance group
DEFAULT_CASES = ("S007", "S037", "S003", "S028")


def _published_accuracy() -> Dict[str, float]:
    path = Path("data/metadata/phase4b/phase4b_subject_results.csv")
    frame = pd.read_csv(path)
    frame = frame[(frame.task == "left_vs_right")
                  & (frame.classifier == "logistic_regression")
                  & (frame.representation == "frozen")]
    return dict(zip(frame.subject_id, frame.balanced_accuracy))


def run(out_dir: Path, figures_dir: Path, cases: Sequence[str],
        make_figures: bool = True) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    encoding = encode_cohort()
    adapter = Paper1NeuroStateAdapter()
    published = _published_accuracy()

    epoch_rows: List[Dict[str, object]] = []
    subject_rows: List[Dict[str, object]] = []
    twin_frames: Dict[str, pd.DataFrame] = {}
    hashes: Dict[str, str] = {}

    for subject in encoding.subjects:
        posteriors = encoding.for_subject(subject)
        records = adapter.records(posteriors)
        observations = adapter.observations(posteriors)

        twin = DigitalTwin(TwinConfig(patient_id=subject))
        states = twin.run(observations)
        view = DecisionEngineView(twin)
        hashes[subject] = twin.trajectory_hash()

        for record in records:
            row = {k: v for k, v in record.to_dict().items()
                   if k not in ("neuro", "reliability")}
            row.update({f"neuro.{k}": v for k, v in record.neuro.to_dict().items()})
            row.update({f"reliability.{k}": v
                        for k, v in record.reliability.to_dict().items()
                        if k != "caveat"})
            epoch_rows.append(row)

        summary = subject_summary(records)
        summary["paper1_balanced_accuracy"] = published.get(subject)
        summary["paper1_group"] = ("below_chance" if subject in PAPER1_BELOW_CHANCE
                                   else "at_or_above_chance")
        summary["trajectory_hash"] = hashes[subject]
        summary["n_twin_ticks"] = len(states)
        for name in LatentState.FIELDS:
            estimate = getattr(twin.state.latent, name)
            readout = view.latent(name)
            summary[f"twin.{name}.value"] = estimate.value
            summary[f"twin.{name}.sd"] = estimate.sd
            summary[f"twin.{name}.n_updates"] = estimate.n_updates
            summary[f"twin.{name}.usable"] = readout.usable
        subject_rows.append(summary)

        if subject in cases:
            twin_frames[subject] = twin.history.to_frame()
            write_states(out_dir / f"{subject}.states.jsonl", states, twin.config)

    epochs = pd.DataFrame(epoch_rows)
    subjects = pd.DataFrame(subject_rows)
    epochs.to_csv(out_dir / "phase6b_neurostates.csv", index=False)
    subjects.to_csv(out_dir / "phase6b_subjects.csv", index=False)

    # --- determinism: two independent end-to-end runs must agree -------------
    second = encode_cohort()
    second_adapter = Paper1NeuroStateAdapter()
    determinism: Dict[str, object] = {"subjects_checked": 0, "identical": True,
                                      "first_divergence": None}
    for subject in encoding.subjects:
        twin = DigitalTwin(TwinConfig(patient_id=subject))
        twin.run(second_adapter.observations(second.for_subject(subject)))
        determinism["subjects_checked"] += 1  # type: ignore[operator]
        if twin.trajectory_hash() != hashes[subject]:
            determinism["identical"] = False
            determinism["first_divergence"] = subject
            break

    group = subjects.groupby("paper1_group")
    summary: Dict[str, object] = {
        "phase": "6B",
        "n_subjects": len(subjects),
        "n_epochs": len(epochs),
        "paper1_reproduction": encoding.reproduction,
        "subject_independence_audit": encoding.audit,
        "determinism": determinism,
        "reliability_floor": adapter.min_margin,
        "group_means": {
            name: {
                "n": int(len(frame)),
                "paper1_balanced_accuracy": float(frame.paper1_balanced_accuracy.mean()),
                "agreement_with_cue": float(frame.agreement_with_cue.mean()),
                "mean_margin": float(frame.mean_margin.mean()),
                "mean_entropy": float(frame.mean_entropy.mean()),
                "usable_epoch_fraction": float(frame.usable_epoch_fraction.mean()),
                "twin_channel_confidence": float(
                    frame["twin.confidence_estimate.value"].mean()),
                "twin_channel_confidence_sd": float(
                    frame["twin.confidence_estimate.sd"].mean()),
                "twin_attention_usable_fraction": float(
                    frame["twin.attention_estimate.usable"].mean()),
                "twin_motor_state_usable_fraction": float(
                    frame["twin.motor_state_estimate.usable"].mean()),
            }
            for name, frame in group
        },
        "case_subjects": list(cases),
        "trajectory_hashes": hashes,
    }
    header = run_provenance(list(encoding.folds.values()),
                            reproduction=encoding.reproduction)
    header.pop("kind", None)
    summary["provenance"] = header

    (out_dir / "phase6b_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")

    if make_figures:
        from src.viz import encoder_figures

        figures_dir.mkdir(parents=True, exist_ok=True)
        encoder_figures.cohort_figure(figures_dir / "65_decoder_vs_twin_cohort.png",
                                      subjects)
        encoder_figures.case_figure(figures_dir / "66_case_subjects.png",
                                    twin_frames, subjects, list(cases))
        encoder_figures.reliability_figure(
            figures_dir / "67_posterior_reliability.png", epochs, subjects)

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    summary = run(args.out, args.figures, args.cases,
                  make_figures=not args.no_figures)

    rep = summary["paper1_reproduction"]  # type: ignore[index]
    print("Phase 6B -- frozen Paper 1 encoder into the Phase 6A twin")
    print(f"  epochs {summary['n_epochs']}  subjects {summary['n_subjects']}")
    print(f"  Paper 1 reproduction: mean {rep['reproduced_mean']:.4f} vs published "
          f"{rep['published_mean']:.4f}, max diff {rep['max_abs_difference']:.2e}, "
          f"{rep['bit_identical_subjects']}/{rep['n_subjects']} bit-identical")
    print(f"  subject-independence audit: {summary['subject_independence_audit']['n_folds']} "
          f"folds, leakage_detected="
          f"{summary['subject_independence_audit']['leakage_detected']}")
    print(f"  determinism: {summary['determinism']}")
    print()
    print(f"{'group':20s} {'n':>3s} {'p1 acc':>7s} {'agree':>6s} {'margin':>7s} "
          f"{'chanconf':>9s} {'attn usable':>12s} {'motor usable':>13s}")
    for name, record in summary["group_means"].items():  # type: ignore[union-attr]
        print(f"{name:20s} {record['n']:3d} {record['paper1_balanced_accuracy']:7.3f} "
              f"{record['agreement_with_cue']:6.3f} {record['mean_margin']:7.3f} "
              f"{record['twin_channel_confidence']:9.3f} "
              f"{record['twin_attention_usable_fraction']:12.2f} "
              f"{record['twin_motor_state_usable_fraction']:13.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
