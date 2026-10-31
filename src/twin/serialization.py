"""Serialisation, logging and provenance for twin sessions.

Two on-disk formats, with different jobs:

``*.observations.jsonl``
    The **input** log: a provenance header followed by one JSON object per
    observation. This is the replay unit. Storing observations rather than
    states means a replay re-derives every state and would surface any
    accidental dependence on hidden context.

``*.states.jsonl``
    The **output** log: the same header followed by one JSON object per state.
    Written for inspection and for figures, never read back as an input.

The header mirrors the provenance record the EEG pipeline writes
(``src/io/store.py``): git commit, library versions, UTC timestamp and the
configuration fingerprints. It is reimplemented here rather than imported so
that :mod:`src.twin` depends only on the standard library, numpy and pandas --
the twin has to be able to run inside a service next to Unity, where an MNE
import would be dead weight.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from src.twin.observation import Observation
from src.twin.states import VirtualPatientState
from src.twin.twin import DigitalTwin, TwinConfig

#: Version of the on-disk record layout. Bumped if a field is removed or its
#: meaning changes, so an old log can be recognised rather than misread.
SCHEMA_VERSION = "phase6a.1"

#: Marker distinguishing the header line from data lines.
HEADER_KIND = "header"


def _git_commit() -> Optional[str]:
    """Best-effort current commit; ``None`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def _library_versions() -> Dict[str, str]:
    import numpy
    import pandas

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    }


def provenance(config: TwinConfig, *, extra: Optional[Dict[str, object]] = None
               ) -> Dict[str, object]:
    """The header record written at the top of every log."""
    record: Dict[str, object] = {
        "kind": HEADER_KIND,
        "schema_version": SCHEMA_VERSION,
        "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "libraries": _library_versions(),
        "patient_id": config.patient_id,
        "twin_config_fingerprint": config.fingerprint,
        "latent_config_fingerprint": config.latent.fingerprint,
        "latent_config": config.latent.to_dict(),
        "note": (
            "Latent values in this file are simulation estimates, not "
            "measurements, and are not clinically validated."
        ),
    }
    if extra:
        record.update(extra)
    return record


# ------------------------------------------------------------------- writing


def write_observations(path: Path, observations: Sequence[Observation],
                       config: TwinConfig, *,
                       extra: Optional[Dict[str, object]] = None) -> Path:
    """Write the replayable input log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(provenance(config, extra=extra), sort_keys=True) + "\n")
        for obs in observations:
            handle.write(json.dumps({"kind": "observation", **obs.to_dict()},
                                    sort_keys=True) + "\n")
    return path


def write_states(path: Path, states: Sequence[VirtualPatientState],
                 config: TwinConfig, *,
                 extra: Optional[Dict[str, object]] = None) -> Path:
    """Write the human-readable output log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(provenance(config, extra=extra), sort_keys=True) + "\n")
        for state in states:
            handle.write(json.dumps({"kind": "state", **state.to_dict()},
                                    sort_keys=True) + "\n")
    return path


# ------------------------------------------------------------------- reading


def _read_jsonl(path: Path) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    header: Optional[Dict[str, object]] = None
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("kind") == HEADER_KIND:
                header = payload
            else:
                records.append(payload)
    if header is None:
        raise ValueError(f"{path} has no provenance header; refusing to read it")
    if header.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} was written with schema {header.get('schema_version')!r}, "
            f"this build reads {SCHEMA_VERSION!r}"
        )
    return header, records


def read_observations(path: Path) -> Tuple[Dict[str, object], List[Observation]]:
    header, records = _read_jsonl(path)
    return header, [Observation.from_dict(r) for r in records]


def read_states(path: Path) -> Tuple[Dict[str, object], List[VirtualPatientState]]:
    header, records = _read_jsonl(path)
    return header, [VirtualPatientState.from_dict(r) for r in records]


# ------------------------------------------------------------------- replay


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of replaying a log against a freshly constructed twin."""

    identical: bool
    original_hash: str
    replayed_hash: str
    n_observations: int
    first_divergence: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "identical": self.identical,
            "original_hash": self.original_hash,
            "replayed_hash": self.replayed_hash,
            "n_observations": self.n_observations,
            "first_divergence": self.first_divergence,
        }


def replay_and_compare(observations: Sequence[Observation],
                       config: TwinConfig) -> ReplayResult:
    """Run the same observations through two independent twins and compare.

    Two fresh twins rather than one twin twice: a twin that accidentally kept
    state across runs would pass the second test and fail this one.
    """
    first = DigitalTwin.replay(observations, config)
    second = DigitalTwin.replay(observations, config)

    a = first.history.to_list()
    b = second.history.to_list()
    divergence = None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            divergence = i
            break
    if divergence is None and len(a) != len(b):
        divergence = min(len(a), len(b))

    return ReplayResult(
        identical=first.trajectory_hash() == second.trajectory_hash(),
        original_hash=first.trajectory_hash(),
        replayed_hash=second.trajectory_hash(),
        n_observations=len(observations),
        first_divergence=divergence,
    )


def round_trip_observations(observations: Sequence[Observation],
                            path: Path, config: TwinConfig) -> ReplayResult:
    """Write a log, read it back, replay it, and compare against the original.

    This is the test that matters for Phase 6B: the encoder will write logs on
    one machine and they will be replayed on another, so the guarantee has to
    survive a trip through JSON, not merely through memory.
    """
    write_observations(path, observations, config)
    _, restored = read_observations(path)

    original = DigitalTwin.replay(observations, config)
    replayed = DigitalTwin.replay(restored, config)

    a = original.history.to_list()
    b = replayed.history.to_list()
    divergence = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if divergence is None and len(a) != len(b):
        divergence = min(len(a), len(b))

    return ReplayResult(
        identical=original.trajectory_hash() == replayed.trajectory_hash(),
        original_hash=original.trajectory_hash(),
        replayed_hash=replayed.trajectory_hash(),
        n_observations=len(observations),
        first_divergence=divergence,
    )
