"""Tests for multi-feature risk scorer."""

import json
from pathlib import Path

import pytest

from minerva.risk.scorer import RiskConfig, RiskScorer
from minerva.schema import RiskScore, Shipment


@pytest.fixture
def config():
    return RiskConfig(
        country_tiers={"XX": 4, "YY": 3, "ZZ": 2, "AA": 1},
        high_value_threshold=100_000.0,
        sensitive_hs_prefixes=["93", "36"],
    )


@pytest.fixture
def scorer(config):
    return RiskScorer(config=config)


class TestRiskScorer:
    def test_empty_shipment_low_score(self, scorer):
        # Missing all fields → only missing_fields signal contributes
        shipment = Shipment(id="S1", description="something")
        result = scorer.score(shipment)
        # Missing all 4 critical fields → missing_fields signal value 1.0 × weight 0.1 = 0.1
        assert result.raw_score == pytest.approx(0.1, abs=0.01)
        assert result.score == RiskScore.VERY_LOW

    def test_high_risk_origin(self, scorer):
        shipment = Shipment(
            id="S1",
            description="test",
            origin_country="XX",
            destination_country="US",
            consignee="ACME",
            declared_value=1000.0,
        )
        result = scorer.score(shipment)
        # Origin XX = tier 4, value = 4/4 = 1.0, weight 0.25 = 0.25
        signal = next(s for s in result.signals if s.name == "origin_country")
        assert signal.value == 1.0

    def test_high_value_triggers_signal(self, scorer):
        shipment = Shipment(
            id="S1",
            description="test",
            origin_country="US",
            destination_country="UK",
            consignee="ACME",
            declared_value=500_000.0,
        )
        result = scorer.score(shipment)
        signal = next(s for s in result.signals if s.name == "high_value")
        assert signal is not None
        assert signal.value > 0

    def test_low_value_no_signal(self, scorer):
        shipment = Shipment(
            id="S1",
            description="test",
            origin_country="US",
            destination_country="UK",
            consignee="ACME",
            declared_value=100.0,
        )
        result = scorer.score(shipment)
        assert not any(s.name == "high_value" for s in result.signals)

    def test_sensitive_hs_code(self, scorer):
        shipment = Shipment(
            id="S1",
            description="test",
            origin_country="US",
            destination_country="UK",
            consignee="ACME",
            declared_value=1000.0,
            hs_code="9301.10",
        )
        result = scorer.score(shipment)
        signal = next(s for s in result.signals if s.name == "sensitive_hs")
        assert signal.value == 1.0

    def test_non_sensitive_hs_code(self, scorer):
        shipment = Shipment(
            id="S1",
            description="test",
            origin_country="US",
            destination_country="UK",
            consignee="ACME",
            declared_value=1000.0,
            hs_code="6109.10",
        )
        result = scorer.score(shipment)
        assert not any(s.name == "sensitive_hs" for s in result.signals)

    def test_critical_risk_tier(self, scorer):
        # Stack multiple high-risk signals
        shipment = Shipment(
            id="S1",
            description="test",
            origin_country="XX",       # tier 4 → 0.25
            destination_country="XX",  # tier 4 → 0.25
            consignee="ACME",
            declared_value=1_000_000.0,  # high value → ~0.15
            hs_code="9301.10",         # sensitive → 0.15
        )
        result = scorer.score(shipment)
        assert result.score.value >= RiskScore.HIGH.value

    def test_tier_mapping(self):
        assert RiskScorer._to_tier(0.0) == RiskScore.VERY_LOW
        assert RiskScorer._to_tier(0.25) == RiskScore.LOW
        assert RiskScorer._to_tier(0.45) == RiskScore.MEDIUM
        assert RiskScorer._to_tier(0.65) == RiskScore.HIGH
        assert RiskScorer._to_tier(0.85) == RiskScore.CRITICAL

    def test_score_batch(self, scorer):
        shipments = [
            Shipment(id="S1", description="d1", origin_country="AA",
                     destination_country="US", consignee="ACME", declared_value=100),
            Shipment(id="S2", description="d2", origin_country="XX",
                     destination_country="XX", consignee="BAD", declared_value=1_000_000),
        ]
        results = scorer.score_batch(shipments)
        assert len(results) == 2
        assert results[1].score.value > results[0].score.value


class TestRiskConfig:
    def test_from_json(self, tmp_path):
        config_data = {
            "country_tiers": {"XX": 4},
            "high_value_threshold": 50_000.0,
            "sensitive_hs_prefixes": ["93"],
        }
        path = tmp_path / "risk.json"
        path.write_text(json.dumps(config_data))

        config = RiskConfig.from_json(path)
        assert config.country_tiers == {"XX": 4}
        assert config.high_value_threshold == 50_000.0
        assert "93" in config.sensitive_hs_prefixes

    def test_from_missing_json_returns_defaults(self, tmp_path):
        config = RiskConfig.from_json(tmp_path / "nonexistent.json")
        assert config.high_value_threshold == 100_000.0
