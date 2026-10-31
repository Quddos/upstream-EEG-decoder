"""Shared argument parsing for the Phase 1 command-line entry points."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from typing import Optional, Sequence

from src.config import DatasetConfig, PreprocessingConfig, parse_runs, parse_subjects


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--subjects",
        default="dev",
        help=(
            "subjects to process: 'dev' (S001-S005, the default), 'all' (all 109), "
            "or an explicit list/range such as '1-5,20'"
        ),
    )
    parser.add_argument(
        "--runs",
        default="imagery",
        help="runs to process: 'imagery' (4,8,12 - the default) or an explicit list",
    )
    parser.add_argument(
        "--source",
        choices=("physionet", "synthetic"),
        default="physionet",
        help=(
            "where EDF files come from. 'physionet' downloads the real dataset; "
            "'synthetic' generates structurally identical local fixtures, for "
            "environments without access to physionet.org"
        ),
    )
    parser.add_argument("--l-freq", type=float, default=None, help="band-pass low cutoff (Hz)")
    parser.add_argument("--h-freq", type=float, default=None, help="band-pass high cutoff (Hz)")
    parser.add_argument(
        "--notch", action="store_true", help="enable the 60 Hz notch filter"
    )
    parser.add_argument(
        "--no-car",
        action="store_true",
        help="do not apply the common average reference",
    )
    parser.add_argument(
        "--epoch-tmax", type=float, default=None, help="epoch window end (s)"
    )
    parser.add_argument(
        "--no-rest", action="store_true", help="exclude REST epochs (binary L/R dataset)"
    )
    parser.add_argument(
        "--max-rest-per-run",
        type=int,
        default=None,
        help="cap on REST epochs kept per run (-1 keeps all)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="abort on the first failed recording instead of skipping it",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable INFO-level logging"
    )
    return parser


def config_from_args(args: argparse.Namespace) -> DatasetConfig:
    """Build a :class:`DatasetConfig` from parsed CLI arguments."""
    base = DatasetConfig()
    pre_overrides = {}
    if args.l_freq is not None:
        pre_overrides["l_freq"] = args.l_freq
    if args.h_freq is not None:
        pre_overrides["h_freq"] = args.h_freq
    if args.notch:
        pre_overrides["notch_enabled"] = True
    if args.no_car:
        pre_overrides["common_average_reference"] = False
    if args.epoch_tmax is not None:
        pre_overrides["epoch_tmax"] = args.epoch_tmax

    preprocessing: PreprocessingConfig = (
        replace(base.preprocessing, **pre_overrides) if pre_overrides else base.preprocessing
    )

    max_rest = base.max_rest_epochs_per_run
    if args.max_rest_per_run is not None:
        max_rest = None if args.max_rest_per_run < 0 else args.max_rest_per_run

    return replace(
        base,
        subjects=parse_subjects(args.subjects),
        runs=parse_runs(args.runs),
        source=args.source,
        include_rest=not args.no_rest,
        max_rest_epochs_per_run=max_rest,
        strict=args.strict,
        seed=args.seed if args.seed is not None else base.seed,
        preprocessing=preprocessing,
    )


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    # MNE is extremely chatty at INFO; keep its output out of the pipeline log.
    logging.getLogger("mne").setLevel(logging.ERROR)
