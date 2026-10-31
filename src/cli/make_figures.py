"""Generate the diagnostic figures from an already-built dataset.

    python -m src.cli.make_figures --subjects dev

Requires ``python -m src.cli.build_dataset`` to have run first.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from src.cli._common import add_common_arguments, config_from_args, setup_logging
from src.data.physionet_spec import RunSpec
from src.viz.figures import generate_all_figures


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.make_figures",
        description="Render the Phase 1 diagnostic figures into reports/figures/.",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--example-subject",
        type=int,
        default=None,
        help="subject used for the single-recording figures (default: first configured)",
    )
    parser.add_argument(
        "--example-run",
        type=int,
        default=None,
        help="run used for the single-recording figures (default: first configured)",
    )
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    config = config_from_args(args)
    spec = RunSpec(
        subject=args.example_subject or config.subjects[0],
        run=args.example_run or config.runs[0],
    )

    print(f"Rendering figures from {spec} into {config.resolved_paths.figures}")
    for path in generate_all_figures(config, spec=spec):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
