"""Tests for Layer 3 confidence-based routing logic.

Exhaustive testing of all routing branches — this is the most
critical unit test file for compliance correctness.
"""

import pytest

from minerva.config import RoutingConfig
from minerva.router.decision import route, route_batch
from minerva.schema import (
    Action,
    ClassificationResult,
    ClassifierLabel,
    EntityMatch,
    RiskAssessment,
    RiskLevel,
    RiskScore,
    RiskSignal,
    Shipment,
    TaxonomyHit,
)


@pytest.fixture
def routing_config():
    return RoutingConfig(
        auto_approve_min_confidence=0.90,
        auto_block_min_confidence=0.90,
    )


@pytest.fixture
def shipment():
    return Shipment(id="TEST-001", description="test shipment")


def _make_classification(label: ClassifierLabel, confidence: float) -> ClassificationResult:
    return ClassificationResult(label=label, confidence=confidence)


def _make_taxonomy_hit(
    group_id: str = "mil",
    keyword: str | None = None,
    phrase: str | None = None,
) -> TaxonomyHit:
    return TaxonomyHit(
        group_id=group_id,
        group_name="Military",
        risk_level=RiskLevel.CRITICAL,
        matched_keyword=keyword,
        matched_phrase=phrase,
        similarity_score=0.85 if phrase else None,
    )


class TestTaxonomyBlocksAlways:
    """Taxonomy hits are non-negotiable — always BLOCK regardless of AI output."""

    def test_taxonomy_hit_with_ai_allowed(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[_make_taxonomy_hit(keyword="ak47")],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.99),
            routing_config=routing_config,
        )
        assert decision.action == Action.BLOCK
        assert "Taxonomy hit" in decision.reason

    def test_taxonomy_hit_with_ai_restricted(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[_make_taxonomy_hit(phrase="military equipment")],
            classification=_make_classification(ClassifierLabel.RESTRICTED, 0.95),
            routing_config=routing_config,
        )
        assert decision.action == Action.BLOCK

    def test_taxonomy_hit_with_ai_needs_review(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[_make_taxonomy_hit(keyword="ak47")],
            classification=_make_classification(ClassifierLabel.NEEDS_REVIEW, 0.50),
            routing_config=routing_config,
        )
        assert decision.action == Action.BLOCK

    def test_taxonomy_hit_with_low_ai_confidence(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[_make_taxonomy_hit(keyword="ivory")],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.30),
            routing_config=routing_config,
        )
        assert decision.action == Action.BLOCK

    def test_multiple_taxonomy_hits(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[
                _make_taxonomy_hit("mil", keyword="ak47"),
                _make_taxonomy_hit("haz", phrase="explosive materials"),
            ],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.99),
            routing_config=routing_config,
        )
        assert decision.action == Action.BLOCK
        assert len(decision.taxonomy_hits) == 2


class TestAutoApprove:
    """High-confidence ALLOWED → APPROVE."""

    def test_allowed_high_confidence(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.95),
            routing_config=routing_config,
        )
        assert decision.action == Action.APPROVE
        assert "allowed" in decision.reason.lower()

    def test_allowed_at_threshold(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.90),
            routing_config=routing_config,
        )
        assert decision.action == Action.APPROVE

    def test_allowed_below_threshold(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.89),
            routing_config=routing_config,
        )
        assert decision.action == Action.MANUAL_REVIEW


class TestAutoBlock:
    """High-confidence RESTRICTED → BLOCK."""

    def test_restricted_high_confidence(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.RESTRICTED, 0.95),
            routing_config=routing_config,
        )
        assert decision.action == Action.BLOCK
        assert "restricted" in decision.reason.lower()

    def test_restricted_at_threshold(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.RESTRICTED, 0.90),
            routing_config=routing_config,
        )
        assert decision.action == Action.BLOCK

    def test_restricted_below_threshold(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.RESTRICTED, 0.89),
            routing_config=routing_config,
        )
        assert decision.action == Action.MANUAL_REVIEW


class TestManualReview:
    """All uncertain outcomes → MANUAL_REVIEW."""

    def test_needs_review_high_confidence(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.NEEDS_REVIEW, 0.95),
            routing_config=routing_config,
        )
        assert decision.action == Action.MANUAL_REVIEW

    def test_needs_review_low_confidence(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.NEEDS_REVIEW, 0.50),
            routing_config=routing_config,
        )
        assert decision.action == Action.MANUAL_REVIEW

    def test_allowed_low_confidence(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.60),
            routing_config=routing_config,
        )
        assert decision.action == Action.MANUAL_REVIEW

    def test_restricted_low_confidence(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.RESTRICTED, 0.50),
            routing_config=routing_config,
        )
        assert decision.action == Action.MANUAL_REVIEW


class TestDecisionReason:
    """Verify reason strings contain meaningful information."""

    def test_taxonomy_reason_includes_group(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[_make_taxonomy_hit(keyword="ak47")],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.99),
            routing_config=routing_config,
        )
        assert "Military" in decision.reason
        assert "ak47" in decision.reason

    def test_approve_reason_includes_confidence(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.95),
            routing_config=routing_config,
        )
        assert "0.950" in decision.reason

    def test_review_reason_includes_below_threshold(self, shipment, routing_config):
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.70),
            routing_config=routing_config,
        )
        assert "below threshold" in decision.reason.lower()


class TestRouteBatch:
    def test_batch_routing(self, routing_config):
        shipments = [
            Shipment(id="S1", description="safe item"),
            Shipment(id="S2", description="weapon"),
            Shipment(id="S3", description="uncertain item"),
        ]
        taxonomy_results = [
            [],
            [_make_taxonomy_hit(keyword="ak47")],
            [],
        ]
        classification_results = [
            _make_classification(ClassifierLabel.ALLOWED, 0.95),
            _make_classification(ClassifierLabel.RESTRICTED, 0.99),
            _make_classification(ClassifierLabel.NEEDS_REVIEW, 0.60),
        ]

        decisions = route_batch(
            shipments, taxonomy_results, classification_results, routing_config
        )
        assert len(decisions) == 3
        assert decisions[0].action == Action.APPROVE
        assert decisions[1].action == Action.BLOCK
        assert decisions[2].action == Action.MANUAL_REVIEW

    def test_batch_length_mismatch_raises(self, routing_config):
        with pytest.raises(ValueError):
            route_batch(
                [Shipment(id="S1", description="test")],
                [[], []],  # mismatched length
                [_make_classification(ClassifierLabel.ALLOWED, 0.95)],
                routing_config,
            )


class TestEntityResolutionRouting:
    def test_entity_auto_block_takes_precedence_over_ai(self, shipment, routing_config):
        entity = EntityMatch(
            matched_party="Blocked Industries",
            input_party="blocked industries ltd",
            role="consignee",
            score=0.95,
            list_name="ofac",
            exact_match=False,
        )
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.99),
            routing_config=routing_config,
            entity_matches=[entity],
        )
        assert decision.action == Action.BLOCK
        assert "Denied party" in decision.reason

    def test_exact_entity_match_blocks(self, shipment, routing_config):
        entity = EntityMatch(
            matched_party="X", input_party="x", role="consignee",
            score=1.0, list_name="t", exact_match=True,
        )
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.99),
            routing_config=routing_config,
            entity_matches=[entity],
        )
        assert decision.action == Action.BLOCK

    def test_moderate_entity_match_manual_review(self, shipment, routing_config):
        entity = EntityMatch(
            matched_party="X", input_party="x", role="consignee",
            score=0.85, list_name="t", exact_match=False,
        )
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.99),
            routing_config=routing_config,
            entity_matches=[entity],
        )
        assert decision.action == Action.MANUAL_REVIEW
        assert "denied-party" in decision.reason.lower()

    def test_taxonomy_still_trumps_entity(self, shipment, routing_config):
        # Taxonomy hit takes absolute precedence
        entity = EntityMatch(
            matched_party="X", input_party="x", role="consignee",
            score=0.95, list_name="t", exact_match=False,
        )
        decision = route(
            shipment,
            taxonomy_hits=[_make_taxonomy_hit(keyword="ak47")],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.99),
            routing_config=routing_config,
            entity_matches=[entity],
        )
        assert decision.action == Action.BLOCK
        assert "Taxonomy" in decision.reason


class TestRiskBasedRouting:
    def test_critical_risk_forces_review(self, shipment, routing_config):
        risk = RiskAssessment(
            score=RiskScore.CRITICAL,
            raw_score=0.85,
            signals=[RiskSignal(name="origin", weight=0.25, value=1.0)],
        )
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.99),
            routing_config=routing_config,
            risk_assessment=risk,
        )
        assert decision.action == Action.MANUAL_REVIEW
        assert "Critical risk" in decision.reason

    def test_high_risk_with_allowed_ai_escalates(self, shipment, routing_config):
        risk = RiskAssessment(
            score=RiskScore.HIGH,
            raw_score=0.65,
            signals=[],
        )
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.95),
            routing_config=routing_config,
            risk_assessment=risk,
        )
        assert decision.action == Action.MANUAL_REVIEW

    def test_low_risk_allows_auto_approve(self, shipment, routing_config):
        risk = RiskAssessment(
            score=RiskScore.LOW,
            raw_score=0.20,
            signals=[],
        )
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.95),
            routing_config=routing_config,
            risk_assessment=risk,
        )
        assert decision.action == Action.APPROVE

    def test_critical_risk_still_blocks_if_taxonomy(self, shipment, routing_config):
        risk = RiskAssessment(score=RiskScore.CRITICAL, raw_score=0.85)
        decision = route(
            shipment,
            taxonomy_hits=[_make_taxonomy_hit(keyword="ak47")],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.99),
            routing_config=routing_config,
            risk_assessment=risk,
        )
        # Taxonomy always wins
        assert decision.action == Action.BLOCK


class TestCustomThresholds:
    def test_lower_approve_threshold(self, shipment):
        config = RoutingConfig(
            auto_approve_min_confidence=0.80,
            auto_block_min_confidence=0.90,
        )
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.ALLOWED, 0.85),
            routing_config=config,
        )
        assert decision.action == Action.APPROVE

    def test_higher_block_threshold(self, shipment):
        config = RoutingConfig(
            auto_approve_min_confidence=0.90,
            auto_block_min_confidence=0.95,
        )
        decision = route(
            shipment,
            taxonomy_hits=[],
            classification=_make_classification(ClassifierLabel.RESTRICTED, 0.92),
            routing_config=config,
        )
        assert decision.action == Action.MANUAL_REVIEW  # Below 0.95 threshold
