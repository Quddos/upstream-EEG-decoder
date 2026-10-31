"""Tests for configuration handling and the scale-up path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.config import (
    ALL_SUBJECTS,
    DEV_SUBJECTS,
    DatasetConfig,
    PathsConfig,
    PreprocessingConfig,
    dev_config,
    full_config,
    parse_runs,
    parse_subjects,
)
from src.data.physionet_spec import N_SUBJECTS_TOTAL, UNILATERAL_IMAGERY_RUNS


class TestDefaults:
    def test_default_is_the_dev_subset(self):
        assert DatasetConfig().subjects == DEV_SUBJECTS == (1, 2, 3, 4, 5)

    def test_default_runs_are_the_imagery_runs(self):
        assert DatasetConfig().runs == UNILATERAL_IMAGERY_RUNS == (4, 8, 12)

    def test_default_passband_covers_mu_and_beta(self):
        pre = PreprocessingConfig()
        assert pre.l_freq == 8.0 and pre.h_freq == 30.0

    def test_baseline_correction_is_off_by_default(self):
        assert PreprocessingConfig().baseline is None

    def test_no_epochs_are_dropped_by_default(self):
        assert PreprocessingConfig().reject_peak_to_peak_v is None

    def test_expected_epoch_samples(self):
        assert PreprocessingConfig().expected_epoch_samples == 640
        assert replace(PreprocessingConfig(), include_tmax=True).expected_epoch_samples == 641
        assert replace(PreprocessingConfig(), epoch_tmax=2.0).expected_epoch_samples == 320


class TestScaleUp:
    def test_scaling_to_all_subjects_is_one_field(self):
        """The brief requires that scaling up touches only a config variable."""
        config = replace(DatasetConfig(), subjects=ALL_SUBJECTS)
        assert len(config.subjects) == N_SUBJECTS_TOTAL
        assert config.runs == DatasetConfig().runs
        assert config.preprocessing == DatasetConfig().preprocessing

    def test_full_config_helper(self):
        assert len(full_config().subjects) == N_SUBJECTS_TOTAL

    def test_dev_config_helper(self):
        assert dev_config().subjects == DEV_SUBJECTS


class TestValidation:
    def test_empty_subjects_rejected(self):
        with pytest.raises(ValueError, match="subjects"):
            replace(DatasetConfig(), subjects=())

    def test_empty_runs_rejected(self):
        with pytest.raises(ValueError, match="runs"):
            replace(DatasetConfig(), runs=())

    def test_unknown_source_rejected(self):
        with pytest.raises(ValueError, match="source"):
            replace(DatasetConfig(), source="magic")


class TestParsing:
    def test_parse_all(self):
        assert parse_subjects("all") == ALL_SUBJECTS

    def test_parse_dev(self):
        assert parse_subjects("dev") == DEV_SUBJECTS

    def test_parse_range(self):
        assert parse_subjects("1-5") == (1, 2, 3, 4, 5)

    def test_parse_mixed(self):
        assert parse_subjects("1-3,20,88") == (1, 2, 3, 20, 88)

    def test_parse_deduplicates_and_sorts(self):
        assert parse_subjects("5,1,5,3") == (1, 3, 5)

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            parse_subjects("110")

    def test_inverted_range_rejected(self):
        with pytest.raises(ValueError, match="invalid subject range"):
            parse_subjects("9-2")

    def test_parse_runs_imagery_keyword(self):
        assert parse_runs("imagery") == (4, 8, 12)

    def test_parse_runs_explicit(self):
        assert parse_runs("4,12") == (4, 12)


class TestFingerprint:
    def test_is_hex_and_stable(self):
        fingerprint = DatasetConfig().fingerprint
        assert len(fingerprint) == 16
        int(fingerprint, 16)
        assert fingerprint == DatasetConfig().fingerprint

    def test_differs_between_subject_sets(self):
        assert full_config().fingerprint != dev_config().fingerprint

    def test_to_dict_is_json_safe(self):
        import json

        json.dumps(DatasetConfig().to_dict())


class TestSourceScopedPaths:
    """Real and synthetic artifacts must never share a location.

    Both sources produce a file called ``S001R04.edf``. Without this split a
    synthetic fixture left in ``data/raw`` would be picked up by a
    ``--source physionet`` build and reported as real data.
    """

    def test_physionet_uses_canonical_locations(self):
        paths = PathsConfig().for_source("physionet")
        assert paths.raw == Path("data/raw")
        assert paths.processed == Path("data/processed")
        assert paths.metadata == Path("data/metadata")
        assert paths.figures == Path("reports/figures")

    def test_synthetic_is_diverted_to_its_own_tree(self):
        paths = PathsConfig().for_source("synthetic")
        assert paths.raw == Path("data/synthetic/raw")
        assert paths.processed == Path("data/synthetic/processed")
        assert paths.metadata == Path("data/synthetic/metadata")
        assert paths.figures == Path("reports/figures_synthetic")

    def test_the_two_trees_never_overlap(self):
        real = PathsConfig().for_source("physionet")
        fake = PathsConfig().for_source("synthetic")
        for attr in ("raw", "processed", "metadata", "figures"):
            assert getattr(real, attr) != getattr(fake, attr)

    def test_resolved_paths_applies_the_split(self, tmp_path):
        real = replace(DatasetConfig(), source="physionet", paths=PathsConfig(root=tmp_path))
        fake = replace(DatasetConfig(), source="synthetic", paths=PathsConfig(root=tmp_path))
        assert real.resolved_paths.raw == tmp_path / "data/raw"
        assert fake.resolved_paths.raw == tmp_path / "data/synthetic/raw"

    def test_resolved_paths_are_absolute(self, tmp_path):
        config = replace(DatasetConfig(), paths=PathsConfig(root=tmp_path))
        for attr in ("raw", "processed", "metadata", "figures"):
            assert getattr(config.resolved_paths, attr).is_absolute()

    def test_source_changes_the_fingerprint(self):
        real = replace(DatasetConfig(), source="physionet")
        fake = replace(DatasetConfig(), source="synthetic")
        assert real.fingerprint != fake.fingerprint
