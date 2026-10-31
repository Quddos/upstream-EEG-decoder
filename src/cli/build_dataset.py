"""Build the processed Phase 1 epoch dataset.

    python -m src.cli.build_dataset --subjects dev
    python -m src.cli.build_dataset --subjects all      # scale to all 109 subjects

Writes ``data/processed/<SUBJECT>.npz``, ``data/metadata/epoch_index.csv``,
``data/metadata/recordings.csv`` and ``data/metadata/dataset_summary.json``.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from src.cli._common import add_common_arguments, config_from_args, setup_logging
from src.preprocessing.pipeline import build_dataset


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.build_dataset",
        description="Preprocess EEGMMIDB into a reproducible motor-imagery epoch dataset.",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--figures",
        action="store_true",
        help="also generate the diagnostic figures after building",
    )
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    config = config_from_args(args)
    print(
        f"Building dataset: {len(config.subjects)} subject(s), runs {list(config.runs)}, "
        f"source={config.source}"
    )
    if config.source == "synthetic":
        print(
            "  NOTE: synthetic source selected -- the output is simulated data, "
            "structurally identical to EEGMMIDB but not real EEG."
        )

    report = build_dataset(config)
    print()
    print(report.render())

    if args.figures:
        from src.viz.figures import generate_all_figures

        print("\nGenerating figures...")
        for path in generate_all_figures(config):
            print(f"  {path}")

    return 1 if report.failures and config.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
