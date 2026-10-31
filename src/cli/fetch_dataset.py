"""Download EEGMMIDB recordings for a subject range, with sidecars, and verify.

    python -m src.cli.fetch_dataset --subjects 9-20 --runs imagery --verify

Fetches only the requested subject x run combinations -- for the motor-imagery
runs that is 3 files per subject at ~2.6 MB each, so a 12-subject expansion moves
roughly 95 MB rather than the 3.4 GB of the full record.

Existing files are never re-downloaded or overwritten, so this is safe to re-run
to fill gaps. With ``--verify`` every fetched file is checked against PhysioNet's
official ``SHA256SUMS.txt`` before the command reports success; a mismatch is
reported and the exit code is non-zero rather than being silently accepted.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from src.cli._common import setup_logging
from src.config import DatasetConfig, parse_runs, parse_subjects
from src.data.download import DatasetDownloadError, download_event_file, event_filename
from src.data.download import download_run
from src.data.integrity import (
    MANIFEST_FILENAME,
    ManifestError,
    fetch_manifest,
    files_for_specs,
    verify_files,
)
from src.data.physionet_spec import iter_run_specs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download EEGMMIDB recordings and their .edf.event sidecars"
    )
    parser.add_argument(
        "--subjects",
        default="dev",
        help="'dev', 'all', or a list/range such as '9-20'",
    )
    parser.add_argument("--runs", default="imagery")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="directory holding data/raw, if not the project root",
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="skip the .edf.event sidecars (the pipeline does not need them)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check every file against the official SHA256SUMS.txt afterwards",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    config = DatasetConfig(
        subjects=parse_subjects(args.subjects), runs=parse_runs(args.runs)
    )
    raw_dir = (
        Path(args.data_root).resolve() / "data" / "raw"
        if args.data_root
        else config.resolved_paths.raw
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    specs = iter_run_specs(config.subjects, config.runs)

    print(f"target     : {raw_dir}")
    print(f"subjects   : {len(config.subjects)}  runs: {list(config.runs)}")
    print(f"recordings : {len(specs)}")
    print()

    fetched = skipped = 0
    failures: List[str] = []
    started = time.time()

    for spec in specs:
        subject_dir = raw_dir / spec.subject_id
        edf_target = subject_dir / spec.filename
        try:
            if edf_target.exists():
                skipped += 1
            else:
                download_run(spec, edf_target)
                fetched += 1
                print(f"  fetched {spec.subject_id}/{spec.filename} "
                      f"({edf_target.stat().st_size / 1e6:.1f} MB)")
        except DatasetDownloadError as exc:
            failures.append(f"{spec}: {exc}")
            print(f"  FAILED  {spec.subject_id}/{spec.filename}")
            continue

        if args.no_events:
            continue
        event_target = subject_dir / event_filename(spec)
        try:
            if not event_target.exists():
                download_event_file(spec, event_target)
        except Exception as exc:  # noqa: BLE001 - sidecars are supplementary
            failures.append(f"{spec} sidecar: {exc}")

    print()
    print(f"fetched {fetched}, already present {skipped}, "
          f"failed {len(failures)}  [{time.time() - started:.0f}s]")
    for message in failures:
        print(f"  ! {message}")

    if not args.verify:
        return 1 if failures else 0

    # --- verification --------------------------------------------------------
    manifest_path = raw_dir.parent / MANIFEST_FILENAME
    try:
        fetch_manifest(manifest_path)
    except ManifestError as exc:
        print(f"\nmanifest unavailable: {exc}")

    report = verify_files(
        files_for_specs(specs, raw_dir, include_events=not args.no_events),
        manifest_path if manifest_path.exists() else None,
    )
    print()
    print(report.claim())
    if report.problems:
        for result in report.problems[:20]:
            print(f"  {result.status:16s} {result.relative_path}")
        return 1
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
