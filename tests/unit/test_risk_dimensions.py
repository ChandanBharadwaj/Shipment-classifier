"""Tests for individual dimension assessors."""

import pytest

from minerva.risk.dimensions import (
    DataQualityAssessor,
    DimensionContext,
    DualUseAssessor,
    DualUseConfig,
    GeographyAssessor,
    GeographyConfig,
    GoodsAssessor,
    HsCodeAssessor,
    HsCodeConfig,
    ModelUncertaintyAssessor,
    PartyAssessor,
    ValuationAssessor,
    ValuationConfig,
)
from minerva.schema import (
    ClassificationResult,
    ClassifierLabel,
    EntityMatch,
    RiskDimension,
    RiskLevel,
    Severity,
    Shipment,
    TaxonomyHit,
)


def _ctx(**kwargs):
    shipment = kwargs.pop("shipment", Shipment(id="S", description="test shipment"))
    return DimensionContext(shipment=shipment, **kwargs)


class TestGoodsAssessor:
    def test_no_signals_no_severity(self):
        result = GoodsAssessor().assess(_ctx())
        assert result.dimension == RiskDimension.GOODS
        assert result.severity == Severity.NONE
        assert result.score == 0.0

    def test_keyword_hit_is_critical(self):
        hit = TaxonomyHit(
            group_id="mil", group_name="Military",
            risk_level=RiskLevel.CRITICAL, matched_keyword="ak47",
        )
        result = GoodsAssessor().assess(_ctx(taxonomy_hits=[hit]))
        assert result.severity == Severity.CRITICAL
        assert any(s.name == "taxonomy_keyword" for s in result.signals)

    def test_semantic_hit_elevates_severity(self):
        hit = TaxonomyHit(
            group_id="mil", group_name="Military",
            risk_level=RiskLevel.CRITICAL,
            matched_phrase="missile system", similarity_score=0.82,
        )
        result = GoodsAssessor().assess(_ctx(taxonomy_hits=[hit]))
        assert result.severity.rank >= Severity.MEDIUM.rank

    def test_restricted_classification_adds_risk(self):
        cls = ClassificationResult(label=ClassifierLabel.RESTRICTED, confidence=0.92)
        result = GoodsAssessor().assess(_ctx(classification=cls))
        assert result.score > 0
        assert any(s.name == "classifier_restricted" for s in result.signals)

    def test_allowed_classification_no_signal(self):
        cls = ClassificationResult(label=ClassifierLabel.ALLOWED, confidence=0.95)
        result = GoodsAssessor().assess(_ctx(classification=cls))
        assert not any("classifier" in s.name for s in result.signals)


class TestPartyAssessor:
    def test_no_matches_no_severity(self):
        shipment = Shipment(id="S", description="t", consignee="ACME")
        result = PartyAssessor().assess(_ctx(shipment=shipment))
        assert result.severity == Severity.NONE

    def test_exact_match_critical(self):
        match = EntityMatch(
            matched_party="X", input_party="x", role="consignee",
            score=1.0, list_name="t", exact_match=True,
        )
        result = PartyAssessor().assess(_ctx(entity_matches=[match]))
        assert result.severity == Severity.CRITICAL

    def test_high_fuzzy_raises_severity(self):
        match = EntityMatch(
            matched_party="X", input_party="x", role="consignee",
            score=0.93, list_name="t", exact_match=False,
        )
        result = PartyAssessor().assess(_ctx(entity_matches=[match]))
        assert result.severity.rank >= Severity.MEDIUM.rank

    def test_no_party_info_yields_mild_signal(self):
        # no consignee, no shipper
        shipment = Shipment(id="S", description="test")
        result = PartyAssessor().assess(_ctx(shipment=shipment))
        assert any(s.name == "no_party_information" for s in result.signals)


class TestGeographyAssessor:
    def test_tier_4_destination_high(self):
        config = GeographyConfig(country_tiers={"XX": 4})
        shipment = Shipment(
            id="S", description="t",
            origin_country="US", destination_country="XX",
        )
        result = GeographyAssessor(config).assess(_ctx(shipment=shipment))
        assert result.severity.rank >= Severity.MEDIUM.rank

    def test_unknown_country_no_signal(self):
        config = GeographyConfig(country_tiers={})
        shipment = Shipment(
            id="S", description="t",
            origin_country="US", destination_country="CA",
        )
        result = GeographyAssessor(config).assess(_ctx(shipment=shipment))
        assert result.severity == Severity.NONE

    def test_both_tier_4_critical_ish(self):
        config = GeographyConfig(country_tiers={"XX": 4, "YY": 4})
        shipment = Shipment(
            id="S", description="t",
            origin_country="XX", destination_country="YY",
        )
        result = GeographyAssessor(config).assess(_ctx(shipment=shipment))
        # Both origin and dest at 1.0 each → mean 1.0 → CRITICAL
        assert result.severity == Severity.CRITICAL

    def test_missing_countries_signal(self):
        shipment = Shipment(id="S", description="t")
        result = GeographyAssessor().assess(_ctx(shipment=shipment))
        assert any("missing" in s.name for s in result.signals)


class TestValuationAssessor:
    def test_missing_value(self):
        shipment = Shipment(id="S", description="t")
        result = ValuationAssessor().assess(_ctx(shipment=shipment))
        assert any(s.name == "missing_declared_value" for s in result.signals)

    def test_high_value_signal(self):
        config = ValuationConfig(high_value_threshold=100_000.0)
        shipment = Shipment(
            id="S", description="t", declared_value=500_000.0,
            declared_currency="USD",
        )
        result = ValuationAssessor(config).assess(_ctx(shipment=shipment))
        assert any(s.name == "high_value" for s in result.signals)

    def test_very_high_value_critical(self):
        config = ValuationConfig(
            high_value_threshold=100_000.0, very_high_value_threshold=1_000_000.0
        )
        shipment = Shipment(
            id="S", description="t", declared_value=5_000_000.0,
            declared_currency="USD",
        )
        result = ValuationAssessor(config).assess(_ctx(shipment=shipment))
        assert any(s.name == "very_high_value" for s in result.signals)

    def test_round_value_flagged(self):
        shipment = Shipment(
            id="S", description="t", declared_value=50_000.0,
            declared_currency="USD",
        )
        result = ValuationAssessor().assess(_ctx(shipment=shipment))
        assert any(s.name == "suspiciously_round_value" for s in result.signals)

    def test_nonpositive_value(self):
        shipment = Shipment(
            id="S", description="t", declared_value=0.0,
            declared_currency="USD",
        )
        result = ValuationAssessor().assess(_ctx(shipment=shipment))
        assert any(s.name == "nonpositive_value" for s in result.signals)


class TestHsCodeAssessor:
    def test_missing_hs_code(self):
        shipment = Shipment(id="S", description="t")
        result = HsCodeAssessor().assess(_ctx(shipment=shipment))
        assert any(s.name == "missing_hs_code" for s in result.signals)

    def test_sensitive_prefix_match(self):
        config = HsCodeConfig(sensitive_prefixes=["93"])
        shipment = Shipment(id="S", description="t", hs_code="9301.10")
        result = HsCodeAssessor(config).assess(_ctx(shipment=shipment))
        assert result.severity.rank >= Severity.MEDIUM.rank
        assert any("sensitive_hs_prefix" in s.name for s in result.signals)

    def test_non_sensitive_prefix_no_signal(self):
        config = HsCodeConfig(sensitive_prefixes=["93"])
        shipment = Shipment(id="S", description="t", hs_code="6109.10")
        result = HsCodeAssessor(config).assess(_ctx(shipment=shipment))
        assert not any("sensitive_hs_prefix" in s.name for s in result.signals)

    def test_malformed_hs_code(self):
        shipment = Shipment(id="S", description="t", hs_code="abc")
        result = HsCodeAssessor().assess(_ctx(shipment=shipment))
        assert any(s.name == "malformed_hs_code" for s in result.signals)


class TestDataQualityAssessor:
    def test_empty_shipment_has_missing_fields(self):
        shipment = Shipment(id="S", description="test shipment item")
        result = DataQualityAssessor().assess(_ctx(shipment=shipment))
        assert any(s.name == "missing_fields" for s in result.signals)

    def test_complete_shipment_clean(self):
        shipment = Shipment(
            id="S",
            description="A full well-described shipment",
            origin_country="US",
            destination_country="CA",
            consignee="Buyer",
            shipper="Seller",
            declared_value=100.0,
            hs_code="1234",
        )
        result = DataQualityAssessor().assess(_ctx(shipment=shipment))
        assert result.severity == Severity.NONE

    def test_short_description_flagged(self):
        shipment = Shipment(
            id="S", description="x",
            origin_country="US", destination_country="CA",
            consignee="Buyer", shipper="Seller",
            declared_value=100.0, hs_code="1234",
        )
        result = DataQualityAssessor().assess(_ctx(shipment=shipment))
        assert any(s.name == "very_short_description" for s in result.signals)

    def test_same_origin_destination_flagged(self):
        shipment = Shipment(
            id="S", description="A complete description",
            origin_country="US", destination_country="us",
            consignee="Buyer", shipper="Seller",
            declared_value=100.0, hs_code="1234",
        )
        result = DataQualityAssessor().assess(_ctx(shipment=shipment))
        assert any(s.name == "origin_equals_destination" for s in result.signals)


class TestModelUncertaintyAssessor:
    def test_no_classifier_high_uncertainty(self):
        result = ModelUncertaintyAssessor().assess(_ctx())
        assert result.severity.rank >= Severity.HIGH.rank

    def test_high_confidence_low_uncertainty(self):
        cls = ClassificationResult(label=ClassifierLabel.ALLOWED, confidence=0.98)
        result = ModelUncertaintyAssessor().assess(_ctx(classification=cls))
        assert result.severity.rank <= Severity.LOW.rank

    def test_low_confidence_high_uncertainty(self):
        cls = ClassificationResult(label=ClassifierLabel.ALLOWED, confidence=0.40)
        result = ModelUncertaintyAssessor().assess(_ctx(classification=cls))
        assert result.severity.rank >= Severity.MEDIUM.rank

    def test_needs_review_label_adds_signal(self):
        cls = ClassificationResult(label=ClassifierLabel.NEEDS_REVIEW, confidence=0.7)
        result = ModelUncertaintyAssessor().assess(_ctx(classification=cls))
        assert any(s.name == "classifier_needs_review" for s in result.signals)


class TestDualUseAssessor:
    def test_no_dual_use_keywords(self):
        shipment = Shipment(id="S", description="Plain cotton t-shirts")
        result = DualUseAssessor().assess(_ctx(shipment=shipment))
        assert result.severity == Severity.NONE

    def test_dual_use_keyword_detected(self):
        shipment = Shipment(
            id="S", description="High-performance computing cluster for research",
        )
        result = DualUseAssessor().assess(_ctx(shipment=shipment))
        assert any(s.name == "dual_use_keyword" for s in result.signals)
        assert result.severity.rank >= Severity.MEDIUM.rank

    def test_multiple_keywords_escalate(self):
        shipment = Shipment(
            id="S",
            description="Encryption hardware with surveillance drone capability",
        )
        result = DualUseAssessor().assess(_ctx(shipment=shipment))
        signal = next(s for s in result.signals if s.name == "dual_use_keyword")
        assert signal.value >= 0.6

    def test_hs_chapter_84_dual_use(self):
        shipment = Shipment(
            id="S", description="industrial widget",
            hs_code="8471.30",  # computer/data processing
        )
        result = DualUseAssessor().assess(_ctx(shipment=shipment))
        assert any(s.name == "dual_use_hs_chapter" for s in result.signals)

    def test_non_dual_use_hs_chapter(self):
        shipment = Shipment(
            id="S", description="cotton garments",
            hs_code="6109.10",
        )
        result = DualUseAssessor().assess(_ctx(shipment=shipment))
        assert not any(s.name == "dual_use_hs_chapter" for s in result.signals)
