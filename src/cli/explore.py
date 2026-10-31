"""Inspect EEGMMIDB recordings without building a dataset.

    python -m src.cli.explore --subjects 1 --runs 4,8,12

Reports, per recording: channel names, sampling rate, duration, annotation counts
and the class labels those annotations resolve to *for that run*.  This is the
step that answers "did I get the run semantics right?" before any epoch is cut.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

import pandas as pd

from src.cli._common import add_common_arguments, config_from_args, setup_logging
from src.data.loader import annotations_table, load_raw, summarize_recording
from src.data.physionet_spec import event_map_for_run, iter_run_specs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.explore",
        description="Inspect EEGMMIDB recordings: channels, sampling rate, annotations.",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--show-annotations",
        action="store_true",
        help="print the full annotation timeline of each recording",
    )
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    config = config_from_args(args)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    rows: List[dict] = []
    failures = 0

    for spec in iter_run_specs(config.subjects, config.runs):
        print(f"\n=== {spec} ===")
        print(f"  file            : {spec.filename}")
        print(
            "  event map       : "
            + ", ".join(f"{k} -> {v}" for k, v in sorted(spec.event_map.items()))
        )
        try:
            raw = load_raw(spec, config, preload=False)
            summary = summarize_recording(spec, config, raw=raw)
        except Exception as exc:  # noqa: BLE001 - exploration reports, never crashes
            print(f"  ERROR           : {exc}")
            failures += 1
            continue

        print(f"  sampling rate   : {summary.sampling_rate} Hz")
        print(f"  channels        : {summary.n_eeg_channels} EEG")
        print(f"  duration        : {summary.duration_s:.2f} s ({summary.n_samples} samples)")
        print(f"  annotations     : {summary.n_annotations}")
        print(f"  event counts    : {summary.event_counts}")
        print(f"  resolved classes: {summary.class_counts}")
        print(f"  first 8 channels: {list(summary.channel_names[:8])}")

        if args.show_annotations:
            print(annotations_table(spec, raw).to_string(index=False))

        rows.append(summary.to_row())

    if rows:
        print("\n=== summary ===")
        frame = pd.DataFrame.from_records(rows)
        columns = [c for c in frame.columns if c not in {"sha256", "path"}]
        print(frame[columns].to_string(index=False))

    if failures:
        print(f"\n{failures} recording(s) could not be inspected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
