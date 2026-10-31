"""Shared pytest fixtures.

The whole suite runs against locally-generated synthetic fixtures, so it needs no
network access and no PhysioNet download.  That is deliberate: the tests validate
the *pipeline*, and a test suite that cannot run offline cannot run in CI.

Tests that require the real dataset are marked ``@pytest.mark.physionet`` and are
deselected by default (see ``pytest.ini``); run them with
``pytest -m physionet`` in an environment with access to physionet.org.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DatasetConfig, PathsConfig  # noqa: E402
from src.io.store import EEGDataset, load_dataset  # noqa: E402
from src.preprocessing.pipeline import BuildReport, build_dataset  # noqa: E402

#: Two subjects and all three imagery runs: enough to exercise multi-subject
#: aggregation and per-run event mapping while keeping the suite fast.
TEST_SUBJECTS = (1, 2)


@pytest.fixture(scope="session")
def synthetic_config(tmp_path_factory: pytest.TempPathFactory) -> DatasetConfig:
    """A pipeline config writing into an isolated temporary tree."""
    root = tmp_path_factory.mktemp("phase1")
    paths = PathsConfig(root=root)
    return replace(
        DatasetConfig(),
        subjects=TEST_SUBJECTS,
        source="synthetic",
        strict=True,
        paths=paths,
    )


@pytest.fixture(scope="session")
def built_dataset(synthetic_config: DatasetConfig) -> BuildReport:
    """Build the dataset once for the whole session."""
    return build_dataset(synthetic_config)


@pytest.fixture(scope="session")
def dataset(synthetic_config: DatasetConfig, built_dataset: BuildReport) -> EEGDataset:
    """The processed dataset, loaded un-normalised (physical units preserved)."""
    return load_dataset(synthetic_config, normalize=False)


@pytest.fixture(scope="session")
def normalized_dataset(
    synthetic_config: DatasetConfig, built_dataset: BuildReport
) -> EEGDataset:
    """The processed dataset with per-epoch normalisation applied."""
    return load_dataset(synthetic_config, normalize=True)
