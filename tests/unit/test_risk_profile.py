"""Tests for the multi-dimensional RiskProfileBuilder."""

import pytest

from minerva.risk.dimensions import (
    DimensionContext,
    GeographyConfig,
    HsCodeConfig,
    ValuationConfig,
)
from minerva.risk.profile import (
    ProfileBuilderConfig,
    RiskProfileBuilder,
    default_assessors,
)
from minerva.schema import (
    Action,
    ClassificationResult,
    ClassifierLabel,
    EntityMatch,
    RiskDimension,
    RiskLevel,
    Severity,
    Shipment,
    TaxonomyHit,
)


@pytest.fixture
def builder():
    assessors = default_assessors(
        geography_config=GeographyConfig(country_tiers={"XX": 4, "YY": 4}),
        valuation_config=ValuationConfig(high_value_threshold=100_000.0),
        hs_code_config=HsCodeConfig(sensitive_prefixes=["93"]),
    )
    return RiskProfileBuilder(assessors=assessors)


class TestRiskProfileBuilder:
    def test_produces_all_dimensions(self, builder):
        ctx = DimensionContext(
            shipment=Shipment(id="S1", description="test"),
        )
        profile = builder.build(ctx)
        dims = {d.dimension for d in profile.dimensions}
        assert dims == set(RiskDimension)

    def test_clean_shipment_low_overall(self, builder):
        ctx = DimensionContext(
            shipment=Shipment(
                id="S1",
                description="Organic cotton t-shirts bulk order for retailer",
                origin_country="US",
                destination_country="CA",
                consignee="ACME Retail",
                shipper="Cotton Co",
                declared_value=5_000.0,
                declared_currency="USD",
                hs_code="6109.10",
                weight_kg=500.0,
            ),
            classification=ClassificationResult(
                label=ClassifierLabel.ALLOWED, confidence=0.95
            ),
        )
        profile = builder.build(ctx)
        assert profile.overall_severity.rank <= Severity.LOW.rank
        assert profile.advisory_action == Action.APPROVE

    def test_taxonomy_keyword_produces_critical(self, builder):
        ctx = DimensionContext(
            shipment=Shipment(id="S1", description="ak47 parts"),
            taxonomy_hits=[
                TaxonomyHit(
                    group_id="mil", group_name="Military",
                    risk_level=RiskLevel.CRITICAL, matched_keyword="ak47",
                )
            ],
            classification=ClassificationResult(
                label=ClassifierLabel.RESTRICTED, confidence=0.9
            ),
        )
        profile = builder.build(ctx)
        assert profile.overall_severity == Severity.CRITICAL
        assert profile.advisory_action == Action.BLOCK
        # Has a hard flag
        assert any(f.name == "taxonomy_keyword_hit" for f in profile.flags)

    def test_exact_denied_party_produces_critical(self, builder):
        ctx = DimensionContext(
            shipment=Shipment(
                id="S1",
                description="legitimate-looking item",
                consignee="Blocked Industries",
            ),
            classification=ClassificationResult(
                label=ClassifierLabel.ALLOWED, confidence=0.95
            ),
            entity_matches=[
                EntityMatch(
                    matched_party="Blocked Industries",
                    input_party="Blocked Industries",
                    role="consignee",
                    score=1.0,
                    list_name="ofac",
                    exact_match=True,
                )
            ],
        )
        profile = builder.build(ctx)
        assert profile.overall_severity == Severity.CRITICAL
        assert any(f.name == "denied_party_exact_match" for f in profile.flags)
        assert profile.advisory_action == Action.BLOCK

    def test_high_risk_geography_escalates(self, builder):
        ctx = DimensionContext(
            shipment=Shipment(
                id="S1",
                description="generic product",
                origin_country="XX",
                destination_country="YY",
                consignee="ACME",
                shipper="BETA",
                declared_value=5000.0,
                declared_currency="USD",
                hs_code="6109",
            ),
            classification=ClassificationResult(
                label=ClassifierLabel.ALLOWED, confidence=0.95
            ),
        )
        profile = builder.build(ctx)
        # Both countries tier 4 → CRITICAL geography → overall at least HIGH
        geography = profile.get_dimension(RiskDimension.GEOGRAPHY)
        assert geography is not None
        assert geography.severity.rank >= Severity.HIGH.rank
        assert profile.overall_severity.rank >= Severity.HIGH.rank

    def test_dual_use_surfaced(self, builder):
        ctx = DimensionContext(
            shipment=Shipment(
                id="S1",
                description="High-performance computing cluster for scientific use",
                origin_country="US",
                destination_country="CA",
                consignee="University",
                shipper="Vendor",
                declared_value=50_000.0,
                declared_currency="USD",
                hs_code="8471.30",
            ),
            classification=ClassificationResult(
                label=ClassifierLabel.ALLOWED, confidence=0.88
            ),
        )
        profile = builder.build(ctx)
        dual_use = profile.get_dimension(RiskDimension.DUAL_USE)
        assert dual_use.severity.rank >= Severity.MEDIUM.rank

    def test_narrative_mentions_flags(self, builder):
        ctx = DimensionContext(
            shipment=Shipment(id="S1", description="contains ak47 parts"),
            taxonomy_hits=[
                TaxonomyHit(
                    group_id="mil", group_name="Military",
                    risk_level=RiskLevel.CRITICAL, matched_keyword="ak47",
                )
            ],
            classification=ClassificationResult(
                label=ClassifierLabel.RESTRICTED, confidence=0.95
            ),
        )
        profile = builder.build(ctx)
        assert "CRITICAL" in profile.narrative
        assert "ak47" in profile.narrative
        assert "officer" in profile.narrative.lower()

    def test_advisory_action_never_authoritative_in_narrative(self, builder):
        ctx = DimensionContext(
            shipment=Shipment(
                id="S1",
                description="Organic cotton t-shirts",
                origin_country="US",
                destination_country="CA",
                consignee="Retailer",
                shipper="Mill",
                declared_value=1000.0,
                declared_currency="USD",
                hs_code="6109",
            ),
            classification=ClassificationResult(
                label=ClassifierLabel.ALLOWED, confidence=0.96
            ),
        )
        profile = builder.build(ctx)
        assert "advisory" in profile.narrative.lower()

    def test_batch(self, builder):
        contexts = [
            DimensionContext(shipment=Shipment(id=f"S{i}", description=f"item {i}"))
            for i in range(3)
        ]
        profiles = builder.build_batch(contexts)
        assert len(profiles) == 3


class TestProfileBuilderConfig:
    def test_from_json(self, tmp_path):
        import json
        data = {
            "dimension_weights": {
                "goods": 3.0,
                "party": 3.0,
                "geography": 1.0,
            }
        }
        path = tmp_path / "weights.json"
        path.write_text(json.dumps(data))
        config = ProfileBuilderConfig.from_json(path)
        assert config.dimension_weights[RiskDimension.GOODS] == 3.0

    def test_missing_config_returns_defaults(self, tmp_path):
        config = ProfileBuilderConfig.from_json(tmp_path / "nothing.json")
        # Defaults include all 8 dimensions
        assert len(config.dimension_weights) == 8
