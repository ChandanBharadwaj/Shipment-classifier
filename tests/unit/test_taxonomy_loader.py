"""Tests for taxonomy JSON loader."""

import json
import tempfile
from pathlib import Path

import pytest

from minerva.schema import RiskLevel
from minerva.taxonomy.loader import load_taxonomy


class TestLoadTaxonomy:
    def test_load_valid_taxonomy(self, sample_taxonomy_path):
        groups = load_taxonomy(sample_taxonomy_path)
        assert len(groups) == 3
        assert groups[0].group_id == "hazardous_materials"
        assert groups[0].risk_level == RiskLevel.CRITICAL
        assert len(groups[0].semantic_phrases) == 3

    def test_load_with_hard_keywords(self, sample_taxonomy_path):
        groups = load_taxonomy(sample_taxonomy_path)
        mil = next(g for g in groups if g.group_id == "military_and_dual_use")
        assert "ak47" in mil.hard_keywords

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_taxonomy(Path("/nonexistent/taxonomy.json"))

    def test_missing_groups_key(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"version": "1.0"}))
        with pytest.raises(ValueError, match="'groups' key"):
            load_taxonomy(path)

    def test_missing_group_id(self, tmp_path):
        path = tmp_path / "bad.json"
        data = {
            "groups": [
                {"name": "Test", "risk_level": "critical", "semantic_phrases": ["test"]}
            ]
        }
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="group_id"):
            load_taxonomy(path)

    def test_duplicate_group_id(self, tmp_path):
        path = tmp_path / "dup.json"
        data = {
            "groups": [
                {"group_id": "g1", "name": "A", "risk_level": "critical", "semantic_phrases": ["a"]},
                {"group_id": "g1", "name": "B", "risk_level": "high", "semantic_phrases": ["b"]},
            ]
        }
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="Duplicate"):
            load_taxonomy(path)

    def test_empty_semantic_phrases(self, tmp_path):
        path = tmp_path / "empty.json"
        data = {
            "groups": [
                {"group_id": "g1", "name": "A", "risk_level": "critical", "semantic_phrases": []}
            ]
        }
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="at least one semantic phrase"):
            load_taxonomy(path)

    def test_valid_all_risk_levels(self, tmp_path):
        path = tmp_path / "levels.json"
        data = {
            "groups": [
                {"group_id": "c", "name": "C", "risk_level": "critical", "semantic_phrases": ["c"]},
                {"group_id": "h", "name": "H", "risk_level": "high", "semantic_phrases": ["h"]},
                {"group_id": "m", "name": "M", "risk_level": "medium", "semantic_phrases": ["m"]},
            ]
        }
        path.write_text(json.dumps(data))
        groups = load_taxonomy(path)
        assert len(groups) == 3
        assert groups[0].risk_level == RiskLevel.CRITICAL
        assert groups[1].risk_level == RiskLevel.HIGH
        assert groups[2].risk_level == RiskLevel.MEDIUM
