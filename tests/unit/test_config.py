"""Tests for configuration management."""

import os
from pathlib import Path

import pytest

from minerva.config import MinervaSettings, RoutingConfig, TaxonomySimilarityThresholds


class TestTaxonomySimilarityThresholds:
    def test_defaults(self):
        t = TaxonomySimilarityThresholds()
        assert t.critical == 0.75
        assert t.high == 0.78
        assert t.medium == 0.80


class TestRoutingConfig:
    def test_defaults(self):
        r = RoutingConfig()
        assert r.auto_approve_min_confidence == 0.90
        assert r.auto_block_min_confidence == 0.90


class TestMinervaSettings:
    def test_defaults(self):
        s = MinervaSettings()
        assert s.config_dir == Path("./config")
        assert s.db_dsn is None
        assert s.log_level == "INFO"
        assert s.embedding_model == "intfloat/e5-small-v2"
        assert s.nli_model == "cross-encoder/nli-deberta-v3-small"
        assert s.active_classifier == "nli"
        assert s.batch_size == 256

    def test_taxonomy_path_property(self):
        s = MinervaSettings(config_dir=Path("/custom/config"))
        assert s.taxonomy_path == Path("/custom/config/taxonomy.json")

    def test_get_similarity_threshold(self):
        s = MinervaSettings()
        assert s.get_similarity_threshold("critical") == 0.75
        assert s.get_similarity_threshold("high") == 0.78
        assert s.get_similarity_threshold("medium") == 0.80

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MINERVA_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("MINERVA_ACTIVE_CLASSIFIER", "distilbert")
        s = MinervaSettings()
        assert s.log_level == "DEBUG"
        assert s.active_classifier == "distilbert"

    def test_nested_config_defaults(self):
        s = MinervaSettings()
        assert s.taxonomy_thresholds.critical == 0.75
        assert s.routing.auto_approve_min_confidence == 0.90
        assert s.nli_hypotheses.allowed == "This shipment is allowed for export without restrictions."
