"""Fetch EEGMMIDB recordings over the direct HTTPS route only.

    python -m src.cli.fetch_direct --subjects 21-40 --runs imagery --verify

Why this exists alongside :mod:`src.cli.fetch_dataset`: the default acquisition
path in :func:`src.data.download.download_run` tries MNE's ``eegbci`` fetcher
first and only falls back to HTTPS if it *raises*. During the Phase 4A download
that fetcher degraded to roughly one 2.6 MB file every five minutes and twice
hung mid-file with no timeout, while a direct HTTPS request to the same host
pulled the same file in 25 seconds. Because the MNE route eventually succeeds, it
never yields to the fallback.

This entry point calls the **frozen** ``_fetch_via_https`` helper directly, so no
Phase 1-3 code is modified -- only the choice of route changes. That helper
carries a 120 s socket timeout, which is what makes a stalled connection fail and
retry instead of hanging forever.

Files already present are skipped, so this is safe to re-run to fill gaps, and it
is interchangeable with ``fetch_dataset`` for any subject range.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from src.cli._common import setup_logging
from src.config import DatasetConfig, parse_runs, parse_subjects
from src.data.download import PHYSIONET_BASE_URL, _fetch_via_https, event_filename
from src.data.integrity import (
    MANIFEST_FILENAME,
    ManifestError,
    fetch_manifest,
    files_for_specs,
    verify_files,
)
from src.data.physionet_spec import RunSpec, iter_run_specs

#: Attempts per file before giving up and moving on to the next one.
MAX_ATTEMPTS = 3


def _fetch_edf(spec: RunSpec, target: Path) -> None:
    """Fetch one EDF over HTTPS, retrying a stalled or failed transfer."""
    errors: List[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            source = _fetch_via_https(spec)
            if source is None or not Path(source).exists():
                raise RuntimeError("fetch returned no file")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            return
        except Exception as exc:  # noqa: BLE001 - any failure is worth retrying
            errors.append(f"attempt {attempt}: {exc}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(2**attempt)
    raise RuntimeError("; ".join(errors[-2:]))


def _fetch_event(spec: RunSpec, target: Path) -> None:
    """Fetch one ``.edf.event`` sidecar over HTTPS."""
    import urllib.request

    url = f"{PHYSIONET_BASE_URL}/{spec.subject_id}/{event_filename(spec)}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "NeuroAdaptiveMirrorWorld/phase4a"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError("empty sidecar")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch EEGMMIDB recordings over direct HTTPS"
    )
    parser.add_argument("--subjects", default="dev")
    parser.add_argument("--runs", default="imagery")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--verify", action="store_true")
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
    print(f"recordings : {len(specs)}  (direct HTTPS route)")
    print()

    fetched = skipped = 0
    failures: List[str] = []
    started = time.time()

    for spec in specs:
        subject_dir = raw_dir / spec.subject_id
        edf_target = subject_dir / spec.filename
        if edf_target.exists():
            skipped += 1
        else:
            began = time.time()
            try:
                _fetch_edf(spec, edf_target)
                fetched += 1
                print(f"  {spec.subject_id}/{spec.filename} "
                      f"{edf_target.stat().st_size / 1e6:.1f} MB "
                      f"[{time.time() - began:.0f}s]", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{spec}: {exc}")
                print(f"  FAILED {spec.subject_id}/{spec.filename}: {exc}", flush=True)
                continue

        if args.no_events:
            continue
        event_target = subject_dir / event_filename(spec)
        if not event_target.exists():
            try:
                _fetch_event(spec, event_target)
            except Exception as exc:  # noqa: BLE001 - sidecars are supplementary
                failures.append(f"{spec} sidecar: {exc}")

    print()
    print(f"fetched {fetched}, already present {skipped}, failed {len(failures)}  "
          f"[{time.time() - started:.0f}s]")
    for message in failures:
        print(f"  ! {message}")

    if not args.verify:
        return 1 if failures else 0

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
