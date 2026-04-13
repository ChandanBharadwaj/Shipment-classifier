"""Tests for domain types in schema.py."""

import pytest
from pydantic import ValidationError

from minerva.schema import (
    Action,
    ClassificationResult,
    ClassifierLabel,
    RiskLevel,
    ScreeningDecision,
    Shipment,
    TaxonomyGroup,
    TaxonomyHit,
)


class TestEnums:
    def test_risk_level_values(self):
        assert RiskLevel.CRITICAL == "critical"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.MEDIUM == "medium"

    def test_action_values(self):
        assert Action.BLOCK == "block"
        assert Action.APPROVE == "approve"
        assert Action.MANUAL_REVIEW == "manual_review"

    def test_classifier_label_values(self):
        assert ClassifierLabel.ALLOWED == "allowed"
        assert ClassifierLabel.RESTRICTED == "restricted"
        assert ClassifierLabel.NEEDS_REVIEW == "needs_review"


class TestShipment:
    def test_create_shipment(self):
        s = Shipment(id="S1", description="test item")
        assert s.id == "S1"
        assert s.description == "test item"
        assert s.metadata == {}

    def test_shipment_with_metadata(self):
        s = Shipment(id="S1", description="test", metadata={"origin": "US"})
        assert s.metadata["origin"] == "US"

    def test_shipment_missing_fields(self):
        with pytest.raises(ValidationError):
            Shipment(id="S1")  # missing description


class TestTaxonomyGroup:
    def test_create_group(self):
        g = TaxonomyGroup(
            group_id="test",
            name="Test Group",
            risk_level=RiskLevel.CRITICAL,
            semantic_phrases=["test phrase"],
        )
        assert g.group_id == "test"
        assert g.hard_keywords == []
        assert g.allow_auto_clear is False

    def test_group_is_frozen(self):
        g = TaxonomyGroup(
            group_id="test",
            name="Test",
            risk_level=RiskLevel.HIGH,
            semantic_phrases=["phrase"],
        )
        with pytest.raises(ValidationError):
            g.group_id = "changed"


class TestTaxonomyHit:
    def test_keyword_hit(self):
        h = TaxonomyHit(
            group_id="mil",
            group_name="Military",
            risk_level=RiskLevel.CRITICAL,
            matched_keyword="ak47",
        )
        assert h.matched_keyword == "ak47"
        assert h.matched_phrase is None
        assert h.similarity_score is None

    def test_semantic_hit(self):
        h = TaxonomyHit(
            group_id="haz",
            group_name="Hazardous",
            risk_level=RiskLevel.CRITICAL,
            matched_phrase="explosive materials",
            similarity_score=0.85,
        )
        assert h.matched_phrase == "explosive materials"
        assert h.similarity_score == 0.85


class TestClassificationResult:
    def test_valid_result(self):
        r = ClassificationResult(label=ClassifierLabel.ALLOWED, confidence=0.95)
        assert r.label == ClassifierLabel.ALLOWED
        assert r.confidence == 0.95

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ClassificationResult(label=ClassifierLabel.ALLOWED, confidence=1.5)
        with pytest.raises(ValidationError):
            ClassificationResult(label=ClassifierLabel.ALLOWED, confidence=-0.1)

    def test_boundary_values(self):
        r1 = ClassificationResult(label=ClassifierLabel.ALLOWED, confidence=0.0)
        r2 = ClassificationResult(label=ClassifierLabel.ALLOWED, confidence=1.0)
        assert r1.confidence == 0.0
        assert r2.confidence == 1.0


class TestScreeningDecision:
    def test_approve_decision(self):
        d = ScreeningDecision(
            shipment_id="S1",
            action=Action.APPROVE,
            reason="AI classified as allowed",
        )
        assert d.action == Action.APPROVE
        assert d.taxonomy_hits == []
        assert d.classification is None

    def test_block_with_taxonomy(self):
        hit = TaxonomyHit(
            group_id="mil",
            group_name="Military",
            risk_level=RiskLevel.CRITICAL,
            matched_keyword="ak47",
        )
        d = ScreeningDecision(
            shipment_id="S1",
            action=Action.BLOCK,
            taxonomy_hits=[hit],
            reason="Taxonomy hit",
        )
        assert d.action == Action.BLOCK
        assert len(d.taxonomy_hits) == 1

    def test_serialization_roundtrip(self):
        cls = ClassificationResult(
            label=ClassifierLabel.RESTRICTED, confidence=0.92
        )
        d = ScreeningDecision(
            shipment_id="S1",
            action=Action.BLOCK,
            classification=cls,
            reason="test",
        )
        data = d.model_dump()
        assert data["shipment_id"] == "S1"
        assert data["action"] == "block"
        assert data["classification"]["label"] == "restricted"
