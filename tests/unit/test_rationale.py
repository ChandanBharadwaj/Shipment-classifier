"""Tests for rationale generation."""

import pytest

from minerva.explain.rationale import TemplatedRationaleGenerator, attach_rationale
from minerva.schema import (
    Action,
    ClassificationResult,
    ClassifierLabel,
    EntityMatch,
    RiskAssessment,
    RiskLevel,
    RiskScore,
    RiskSignal,
    ScreeningDecision,
    Shipment,
    TaxonomyHit,
)


@pytest.fixture
def generator():
    return TemplatedRationaleGenerator()


@pytest.fixture
def shipment():
    return Shipment(id="S1", description="test shipment with some goods")


class TestTemplatedRationaleGenerator:
    def test_approve_rationale(self, generator, shipment):
        decision = ScreeningDecision(
            shipment_id="S1",
            action=Action.APPROVE,
            classification=ClassificationResult(
                label=ClassifierLabel.ALLOWED, confidence=0.95
            ),
            reason="AI allowed",
        )
        rationale = generator.generate(shipment, decision)
        assert "APPROVE" in rationale
        assert "S1" in rationale
        assert "ALLOWED" in rationale
        assert "0.950" in rationale

    def test_block_rationale_with_taxonomy(self, generator, shipment):
        hit = TaxonomyHit(
            group_id="mil",
            group_name="Military",
            risk_level=RiskLevel.CRITICAL,
            matched_keyword="ak47",
        )
        decision = ScreeningDecision(
            shipment_id="S1",
            action=Action.BLOCK,
            taxonomy_hits=[hit],
            classification=ClassificationResult(
                label=ClassifierLabel.ALLOWED, confidence=0.99
            ),
            reason="Taxonomy hit",
        )
        rationale = generator.generate(shipment, decision)
        assert "Military" in rationale
        assert "ak47" in rationale
        assert "non-negotiable" in rationale
        assert "BLOCK" in rationale

    def test_rationale_with_entity_match(self, generator, shipment):
        match = EntityMatch(
            matched_party="Blocked Industries",
            input_party="blocked industries ltd",
            role="consignee",
            score=0.95,
            list_name="ofac",
            exact_match=False,
        )
        decision = ScreeningDecision(
            shipment_id="S1",
            action=Action.BLOCK,
            classification=ClassificationResult(
                label=ClassifierLabel.ALLOWED, confidence=0.9
            ),
            entity_matches=[match],
            reason="Denied party",
        )
        rationale = generator.generate(shipment, decision)
        assert "consignee" in rationale
        assert "Blocked Industries" in rationale
        assert "fuzzy match" in rationale

    def test_rationale_with_risk_assessment(self, generator, shipment):
        signals = [
            RiskSignal(
                name="origin_country",
                weight=0.25,
                value=0.75,
                description="High-risk origin",
            ),
        ]
        risk = RiskAssessment(
            score=RiskScore.HIGH,
            raw_score=0.65,
            signals=signals,
        )
        decision = ScreeningDecision(
            shipment_id="S1",
            action=Action.MANUAL_REVIEW,
            classification=ClassificationResult(
                label=ClassifierLabel.ALLOWED, confidence=0.92
            ),
            risk_assessment=risk,
            reason="High risk",
        )
        rationale = generator.generate(shipment, decision)
        assert "risk assessment" in rationale.lower()
        assert "tier 4" in rationale
        assert "origin_country" in rationale

    def test_long_description_truncation(self, generator):
        long_desc = "a" * 500
        shipment = Shipment(id="S1", description=long_desc)
        decision = ScreeningDecision(
            shipment_id="S1",
            action=Action.APPROVE,
            classification=ClassificationResult(
                label=ClassifierLabel.ALLOWED, confidence=0.95
            ),
            reason="test",
        )
        rationale = generator.generate(shipment, decision)
        assert "..." in rationale


class TestAttachRationale:
    def test_attach_multiple(self, generator):
        shipments = [
            Shipment(id="S1", description="d1"),
            Shipment(id="S2", description="d2"),
        ]
        decisions = [
            ScreeningDecision(
                shipment_id="S1",
                action=Action.APPROVE,
                classification=ClassificationResult(
                    label=ClassifierLabel.ALLOWED, confidence=0.95
                ),
                reason="r1",
            ),
            ScreeningDecision(
                shipment_id="S2",
                action=Action.BLOCK,
                classification=ClassificationResult(
                    label=ClassifierLabel.RESTRICTED, confidence=0.99
                ),
                reason="r2",
            ),
        ]
        updated = attach_rationale(generator, shipments, decisions)
        assert len(updated) == 2
        assert updated[0].rationale
        assert updated[1].rationale
        assert "APPROVE" in updated[0].rationale
        assert "BLOCK" in updated[1].rationale
        # Original decisions unchanged (frozen)
        assert decisions[0].rationale == ""
