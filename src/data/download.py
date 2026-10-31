"""Retrieval of EEGMMIDB recordings from PhysioNet.

Two acquisition routes are supported, tried in order:

1. :func:`mne.datasets.eegbci.load_data` -- MNE's own PhysioNet fetcher.  This is
   the preferred route: it handles caching, resumes, and is the canonical way the
   community obtains this dataset.
2. A direct HTTPS GET against ``physionet.org/files/eegmmidb/1.0.0/...`` as a
   fallback for environments where MNE's fetcher is unavailable.

Both routes copy the EDF into ``data/raw/<subject>/`` so that the rest of the
pipeline depends only on the project's own directory layout, and so that a
processed dataset can be rebuilt byte-for-byte from ``data/raw`` alone.

Network note
------------
Downloading obviously requires outbound access to ``physionet.org``.  In a
sandboxed or air-gapped environment that host may be unreachable; in that case use
``--source synthetic`` to exercise the pipeline on locally-generated fixtures with
identical structure.  :class:`DatasetDownloadError` carries that hint.
"""

from __future__ import annotations

import logging
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.data.physionet_spec import RunSpec

log = logging.getLogger(__name__)

#: Base URL of the official record, version 1.0.0.
PHYSIONET_BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"

#: Number of attempts per recording, with exponential backoff between them.
MAX_ATTEMPTS = 4

#: Suffix of the WFDB annotation sidecar that accompanies every EDF in the record.
#: The EDF+ file already carries its own annotation channel -- which is what MNE
#: reads and what the pipeline epochs on -- so this sidecar is *not* needed to
#: build the dataset.  It is fetched because it is part of the official record and
#: gives an independent copy of the event stream to cross-check against.
EVENT_SUFFIX = ".event"


class DatasetDownloadError(RuntimeError):
    """Raised when a recording could not be obtained from PhysioNet."""

    def __init__(self, spec: RunSpec, reason: str) -> None:
        super().__init__(
            f"could not download {spec} from PhysioNet: {reason}\n"
            f"If this environment has no access to physionet.org, run the pipeline "
            f"with --source synthetic to validate it on locally generated fixtures "
            f"of identical structure."
        )
        self.spec = spec
        self.reason = reason


def download_run(spec: RunSpec, target: Path) -> Path:
    """Fetch one EEGMMIDB recording and place it at ``target``.

    Returns the path to the local EDF.  Existing files are left untouched.
    """
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        for route in (_fetch_via_mne, _fetch_via_https):
            try:
                source = route(spec)
            except Exception as exc:  # noqa: BLE001 - routes raise many types
                errors.append(f"{route.__name__} attempt {attempt}: {exc}")
                continue

            if source is not None and Path(source).exists():
                if Path(source).resolve() != target.resolve():
                    shutil.copy2(source, target)
                log.info("fetched %s -> %s", spec, target)
                return target

        if attempt < MAX_ATTEMPTS:
            backoff = 2**attempt
            log.warning("download of %s failed, retrying in %ss", spec, backoff)
            time.sleep(backoff)

    raise DatasetDownloadError(spec, "; ".join(errors[-4:]) or "unknown error")


def _fetch_via_mne(spec: RunSpec) -> Path | None:
    """Fetch through :mod:`mne.datasets.eegbci`, returning MNE's cached path."""
    from mne.datasets import eegbci

    paths = eegbci.load_data(
        subjects=[spec.subject], runs=[spec.run], update_path=False, verbose="ERROR"
    )
    return Path(paths[0]) if paths else None


def _fetch_via_https(spec: RunSpec) -> Path | None:
    """Fetch straight from the PhysioNet file server into a temporary file."""
    url = f"{PHYSIONET_BASE_URL}/{spec.subject_id}/{spec.filename}"
    tmp = Path.home() / ".cache" / "eegmmidb" / spec.subject_id / spec.filename
    tmp.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        url, headers={"User-Agent": "NeuroAdaptiveMirrorWorld/phase1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise urllib.error.HTTPError(
                    url, response.status, "unexpected status", response.headers, None
                )
            payload = response.read()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"GET {url} failed: {exc}") from exc

    if len(payload) < 1024:
        raise RuntimeError(f"GET {url} returned only {len(payload)} bytes")

    tmp.write_bytes(payload)
    return tmp


def event_filename(spec: RunSpec) -> str:
    """Name of the WFDB annotation sidecar for a recording, e.g. ``S001R04.edf.event``."""
    return f"{spec.filename}{EVENT_SUFFIX}"


def download_event_file(spec: RunSpec, target: Path) -> Path:
    """Fetch one ``.edf.event`` sidecar and place it at ``target``.

    Only the direct HTTPS route is used: MNE's fetcher retrieves the EDF alone, so
    there is no second route to fall back to.  Existing files are left untouched.

    These sidecars are small (well under 1 KB each), so fetching all 15 costs a
    few kilobytes -- nothing like the 3.4 GB full record.
    """
    if target.exists():
        return target

    url = f"{PHYSIONET_BASE_URL}/{spec.subject_id}/{event_filename(spec)}"
    errors: list[str] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "NeuroAdaptiveMirrorWorld/phase1"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 200:
                    raise RuntimeError(f"status {response.status}")
                payload = response.read()
            if not payload:
                raise RuntimeError("empty response")

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            log.info("fetched %s -> %s", url, target)
            return target
        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {exc}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(2**attempt)

    raise DatasetDownloadError(spec, f"GET {url} failed: {'; '.join(errors[-2:])}")


def download_many(
    specs: list[RunSpec], raw_dir: Path, *, with_events: bool = False
) -> list[Path]:
    """Download several recordings, returning the local EDF paths in order.

    ``with_events`` additionally fetches each recording's ``.edf.event`` sidecar
    next to its EDF.  It is off by default because the pipeline does not read
    those files; the epoching path uses the EDF+ annotation channel.
    """
    out = []
    for spec in specs:
        subject_dir = raw_dir / spec.subject_id
        out.append(download_run(spec, subject_dir / spec.filename))
        if with_events:
            download_event_file(spec, subject_dir / event_filename(spec))
    return out
