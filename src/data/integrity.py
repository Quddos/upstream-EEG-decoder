"""Verification of local recordings against PhysioNet's official checksum manifest.

Phase 1 recorded a SHA-256 for every file it ingested but never compared those
hashes to anything, which meant the provenance record could prove *that a file did
not change since we hashed it* and nothing more.  It could not show the file was
the one PhysioNet published.  This module closes that gap by fetching the record's
own ``SHA256SUMS.txt`` and comparing.

The distinction matters for how results may be described.  A recorded hash
supports "this is the same file we processed"; only a manifest comparison supports
"this is the official EEGMMIDB recording".  Nothing here is allowed to blur the
two: :class:`VerificationReport` carries an explicit
:attr:`~VerificationReport.manifest_available` flag, and every caller must state
which of the two claims it is entitled to make.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.data.download import PHYSIONET_BASE_URL
from src.data.physionet_spec import RunSpec

log = logging.getLogger(__name__)

#: Official per-file checksum manifest for the record.
MANIFEST_URL = f"{PHYSIONET_BASE_URL}/SHA256SUMS.txt"

#: Local cache filename for the manifest, stored beside the raw tree.
MANIFEST_FILENAME = "SHA256SUMS.txt"

#: Read size for hashing.  Large enough to keep syscall overhead low on the ~2.6 MB
#: EDFs, small enough that memory use stays flat regardless of file size.
_CHUNK_BYTES = 1 << 20


class ManifestError(RuntimeError):
    """Raised when the manifest cannot be obtained or parsed."""


def sha256_of(path: Path) -> str:
    """Streaming SHA-256 of a file, as lowercase hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_manifest(target: Path, *, force: bool = False, timeout: int = 120) -> Path:
    """Download ``SHA256SUMS.txt`` to ``target`` and return the path.

    A cached copy is reused unless ``force`` is set, so repeat verification runs
    need no network at all.
    """
    if target.exists() and not force:
        return target

    request = urllib.request.Request(
        MANIFEST_URL, headers={"User-Agent": "NeuroAdaptiveMirrorWorld/phase1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ManifestError(f"GET {MANIFEST_URL} returned {response.status}")
            payload = response.read()
    except ManifestError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ManifestError(f"GET {MANIFEST_URL} failed: {exc}") from exc

    if len(payload) < 1024:
        raise ManifestError(f"manifest looks truncated: {len(payload)} bytes")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def parse_manifest(path: Path) -> Dict[str, str]:
    """Parse ``SHA256SUMS.txt`` into ``{relative_path: sha256}``.

    Lines are ``<hex digest><whitespace><path>``.  Malformed lines are skipped
    rather than raising: the manifest covers the whole 109-subject record and a
    single odd line must not prevent verification of the files we do hold.
    """
    entries: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, name = parts[0].strip(), parts[1].strip().lstrip("*")
        if len(digest) == 64:
            entries[name.replace("\\", "/")] = digest.lower()
    if not entries:
        raise ManifestError(f"no usable entries parsed from {path}")
    return entries


@dataclass(frozen=True)
class FileVerification:
    """Outcome for one file."""

    relative_path: str
    local_path: Path
    exists: bool
    computed_sha256: Optional[str]
    expected_sha256: Optional[str]

    @property
    def in_manifest(self) -> bool:
        return self.expected_sha256 is not None

    @property
    def matches(self) -> bool:
        return bool(
            self.exists
            and self.expected_sha256 is not None
            and self.computed_sha256 == self.expected_sha256
        )

    @property
    def status(self) -> str:
        if not self.exists:
            return "MISSING"
        if not self.in_manifest:
            return "NOT_IN_MANIFEST"
        return "OK" if self.matches else "MISMATCH"


@dataclass
class VerificationReport:
    """Outcome across a set of files."""

    manifest_available: bool
    manifest_path: Optional[Path] = None
    manifest_entries: int = 0
    results: List[FileVerification] = field(default_factory=list)

    @property
    def verified(self) -> List[FileVerification]:
        return [r for r in self.results if r.matches]

    @property
    def problems(self) -> List[FileVerification]:
        return [r for r in self.results if not r.matches]

    @property
    def all_verified(self) -> bool:
        """True only when a manifest was obtained *and* every file matched it.

        Deliberately false when no manifest is available: an unverifiable set is
        not a verified one, and the difference must never collapse to "fine".
        """
        return bool(self.manifest_available and self.results and not self.problems)

    def claim(self) -> str:
        """The strongest statement this report actually supports."""
        if not self.manifest_available:
            return (
                "NOT VERIFIED against the official manifest -- the manifest could "
                "not be obtained. Recorded hashes prove only that the files have "
                "not changed since they were ingested."
            )
        if self.all_verified:
            return (
                f"VERIFIED: all {len(self.verified)} file(s) match the official "
                f"PhysioNet SHA256SUMS.txt for EEGMMIDB v1.0.0."
            )
        return (
            f"VERIFICATION FAILED: {len(self.problems)} of {len(self.results)} "
            f"file(s) did not match the official manifest."
        )


def verify_files(
    files: Sequence[Tuple[str, Path]],
    manifest_path: Optional[Path],
) -> VerificationReport:
    """Verify ``(relative_path, local_path)`` pairs against a manifest.

    ``relative_path`` is the name as it appears in the manifest, e.g.
    ``"S001/S001R04.edf"``.  A ``manifest_path`` of ``None`` (or a missing file)
    produces a report that is explicitly *unverified* rather than an empty pass.
    """
    if manifest_path is None or not Path(manifest_path).exists():
        return VerificationReport(
            manifest_available=False,
            results=[
                FileVerification(
                    relative_path=rel,
                    local_path=Path(local),
                    exists=Path(local).exists(),
                    computed_sha256=None,
                    expected_sha256=None,
                )
                for rel, local in files
            ],
        )

    expected = parse_manifest(Path(manifest_path))
    results = []
    for rel, local in files:
        local = Path(local)
        exists = local.exists()
        results.append(
            FileVerification(
                relative_path=rel,
                local_path=local,
                exists=exists,
                computed_sha256=sha256_of(local) if exists else None,
                expected_sha256=expected.get(rel),
            )
        )

    return VerificationReport(
        manifest_available=True,
        manifest_path=Path(manifest_path),
        manifest_entries=len(expected),
        results=results,
    )


def count_sidecar_annotations(path: Path, codes: Sequence[str] = ("T0", "T1", "T2")) -> Dict[str, int]:
    """Count each annotation code in a ``.edf.event`` WFDB sidecar.

    **This is a substring scan, not a WFDB parse.** The sidecar is a binary WFDB
    annotation file whose aux strings appear verbatim as ``b"T0 duration: 4.2"``
    etc., so counting occurrences of ``b"T0 "`` recovers the number of annotations
    of that type without a WFDB reader (``wfdb`` is not a project dependency).

    What that supports and what it does not: the counts are a genuine independent
    check that the sidecar carries the same *number* of each event as the EDF+
    annotation channel. They say nothing about event *onsets*, and a full WFDB
    parse would be needed to compare timing. Callers must not describe this as
    validating the event stream in general.
    """
    payload = Path(path).read_bytes()
    return {code: payload.count(f"{code} ".encode("ascii")) for code in codes}


def files_for_specs(
    specs: Iterable[RunSpec], raw_dir: Path, *, include_events: bool = True
) -> List[Tuple[str, Path]]:
    """Manifest-relative names and local paths for the EDFs (and sidecars)."""
    from src.data.download import event_filename

    out: List[Tuple[str, Path]] = []
    for spec in specs:
        subject_dir = raw_dir / spec.subject_id
        out.append((f"{spec.subject_id}/{spec.filename}", subject_dir / spec.filename))
        if include_events:
            name = event_filename(spec)
            out.append((f"{spec.subject_id}/{name}", subject_dir / name))
    return out
