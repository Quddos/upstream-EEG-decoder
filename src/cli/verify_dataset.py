"""Phase 1.7 -- acquire the annotation sidecars and verify the raw record.

    python -m src.cli.verify_dataset --subjects dev --fetch-events --fetch-manifest

Two jobs, both about the *raw* tree; neither touches processed epochs:

1. **Acquire** each recording's ``.edf.event`` sidecar. The pipeline does not need
   these -- it epochs on the EDF+ annotation channel -- but they are part of the
   official record and give an independent copy of the event stream.
2. **Verify** every local file against PhysioNet's own ``SHA256SUMS.txt``. Until
   this runs, the recorded SHA-256 values prove only that files have not changed
   since ingestion, not that they are the files PhysioNet published. The exit code
   is non-zero if verification was attempted and failed, so this is usable as a
   gate.

Nothing here downloads bulk data: the sidecars are under 1 KB each and the
manifest is ~260 KB, so a full 5-subject run transfers well under a megabyte.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.cli._common import setup_logging
from src.config import DatasetConfig, parse_runs, parse_subjects
from src.data.download import download_event_file, event_filename
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
        description="Acquire .edf.event sidecars and verify the raw record"
    )
    parser.add_argument("--subjects", default="dev")
    parser.add_argument("--runs", default="imagery")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="directory holding data/raw, if not the project root",
    )
    parser.add_argument(
        "--fetch-events",
        action="store_true",
        help="download any missing .edf.event sidecars (a few hundred bytes each)",
    )
    parser.add_argument(
        "--fetch-manifest",
        action="store_true",
        help="download SHA256SUMS.txt if not already cached (~260 KB)",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="re-download the manifest even if a cached copy exists",
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="verify only the EDFs, ignoring the sidecars",
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
    specs = iter_run_specs(config.subjects, config.runs)

    print(f"raw tree   : {raw_dir}")
    print(f"recordings : {len(specs)} ({len(config.subjects)} subjects x {len(config.runs)} runs)")
    print()

    # --- 1. acquire sidecars --------------------------------------------------
    include_events = not args.no_events
    if args.fetch_events:
        fetched, present, failed = 0, 0, []
        for spec in specs:
            target = raw_dir / spec.subject_id / event_filename(spec)
            if target.exists():
                present += 1
                continue
            try:
                download_event_file(spec, target)
                fetched += 1
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{spec}: {exc}")
        print(f"event sidecars: {fetched} fetched, {present} already present")
        for message in failed:
            print(f"  FAILED {message}")
        if failed:
            print("  -> sidecars are supplementary; the EDF+ annotation channel is "
                  "what the pipeline reads, so a failure here does not block a build")
        print()

    # --- 2. verify against the official manifest ------------------------------
    manifest_path = raw_dir.parent / MANIFEST_FILENAME
    if args.fetch_manifest or args.refresh_manifest:
        try:
            fetch_manifest(manifest_path, force=args.refresh_manifest)
        except ManifestError as exc:
            print(f"manifest unavailable: {exc}")

    report = verify_files(
        files_for_specs(specs, raw_dir, include_events=include_events),
        manifest_path if manifest_path.exists() else None,
    )

    if report.manifest_available:
        print(f"manifest   : {report.manifest_path} ({report.manifest_entries} entries)")
    else:
        print("manifest   : NOT AVAILABLE (pass --fetch-manifest, or check network)")
    print()

    width = max(len(r.relative_path) for r in report.results) if report.results else 20
    for result in report.results:
        marker = {"OK": "ok", "MISSING": "MISSING", "MISMATCH": "MISMATCH",
                  "NOT_IN_MANIFEST": "not in manifest"}[result.status]
        print(f"  {result.relative_path:<{width}}  {marker}")

    print()
    print(report.claim())

    if report.manifest_available and not report.all_verified:
        print()
        print("STOPPING: the raw tree does not match the published record. Do not "
              "silently repair this -- re-download the offending files and re-run.")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
