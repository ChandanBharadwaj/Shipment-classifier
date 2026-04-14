"""Tests for CrossBorderAssessor and GeographyAssessor when backed
by a LexisNexis-style CountryRiskDatabase."""

import pytest

from minerva.risk.country_risk import (
    CountryRiskDatabase,
    CountryRiskEntry,
    CountryRiskLevel,
    FatfStatus,
    SanctionsStatus,
)
from minerva.risk.dimensions import (
    CrossBorderAssessor,
    CrossBorderConfig,
    DimensionContext,
    GeographyAssessor,
    GeographyConfig,
)
from minerva.schema import (
    ClassificationResult,
    ClassifierLabel,
    RiskDimension,
    Severity,
    Shipment,
)


@pytest.fixture
def country_db():
    return CountryRiskDatabase([
        CountryRiskEntry(
            country_code="US",
            country_name="United States",
            overall_risk=CountryRiskLevel.LOW,
            overall_score=0.15,
        ),
        CountryRiskEntry(
            country_code="CA",
            country_name="Canada",
            overall_risk=CountryRiskLevel.LOW,
            overall_score=0.10,
        ),
        CountryRiskEntry(
            country_code="IR",
            country_name="Iran",
            overall_risk=CountryRiskLevel.CRITICAL,
            overall_score=1.0,
            sanctions_status=SanctionsStatus.COMPREHENSIVE,
            fatf_status=FatfStatus.BLACK,
        ),
        CountryRiskEntry(
            country_code="TR",
            country_name="Türkiye",
            overall_risk=CountryRiskLevel.MEDIUM,
            overall_score=0.55,
            fatf_status=FatfStatus.GREY,
            sanctioned_neighbors=["IR", "SY"],
        ),
        CountryRiskEntry(
            country_code="SG",
            country_name="Singapore",
            overall_risk=CountryRiskLevel.LOW,
            overall_score=0.22,
            transshipment_hub=True,
        ),
        CountryRiskEntry(
            country_code="AE",
            country_name="United Arab Emirates",
            overall_risk=CountryRiskLevel.MEDIUM,
            overall_score=0.55,
            fatf_status=FatfStatus.GREY,
            transshipment_hub=True,
            sanctioned_neighbors=["IR"],
        ),
        CountryRiskEntry(
            country_code="SY",
            country_name="Syria",
            overall_risk=CountryRiskLevel.CRITICAL,
            overall_score=1.0,
            sanctions_status=SanctionsStatus.COMPREHENSIVE,
        ),
    ])


def _ctx(ship, **kwargs):
    return DimensionContext(shipment=ship, **kwargs)


# -------------- GeographyAssessor with CountryRiskDatabase --------------

class TestGeographyAssessorWithDB:
    def test_low_risk_route_no_signals(self, country_db):
        ship = Shipment(
            id="S", description="test",
            origin_country="US", destination_country="CA",
        )
        result = GeographyAssessor(country_db=country_db).assess(_ctx(ship))
        # US->CA both LOW → one LOW signal per country still generated
        # but overall severity should be LOW at most
        assert result.severity.rank <= Severity.LOW.rank

    def test_sanctioned_destination_critical(self, country_db):
        ship = Shipment(
            id="S", description="test",
            origin_country="US", destination_country="IR",
        )
        result = GeographyAssessor(country_db=country_db).assess(_ctx(ship))
        assert result.severity == Severity.CRITICAL
        assert any(s.name == "destination_country_sanctioned" for s in result.signals)

    def test_fatf_grey_signal(self, country_db):
        ship = Shipment(
            id="S", description="test",
            origin_country="US", destination_country="TR",
        )
        result = GeographyAssessor(country_db=country_db).assess(_ctx(ship))
        assert any("fatf" in s.name for s in result.signals)

    def test_fallback_tier_map_when_no_db_entry(self, country_db):
        config = GeographyConfig(country_tiers={"ZZ": 4})
        ship = Shipment(
            id="S", description="test",
            origin_country="US", destination_country="ZZ",
        )
        result = GeographyAssessor(config=config, country_db=country_db).assess(_ctx(ship))
        # ZZ not in db; falls back to tier map → tier 4 signal
        assert any("destination_country_tier_4" in s.name for s in result.signals)

    def test_works_without_db(self):
        config = GeographyConfig(country_tiers={"XX": 4})
        ship = Shipment(
            id="S", description="test",
            origin_country="US", destination_country="XX",
        )
        result = GeographyAssessor(config=config, country_db=None).assess(_ctx(ship))
        assert any("destination_country_tier_4" in s.name for s in result.signals)


# -------------- CrossBorderAssessor ----------------------------------

class TestCrossBorderAssessor:
    def test_no_routing_info_no_signals(self, country_db):
        ship = Shipment(id="S", description="test")
        result = CrossBorderAssessor(country_db=country_db).assess(_ctx(ship))
        assert result.severity == Severity.NONE
        assert len(result.signals) == 0

    def test_sanctioned_transit_forces_critical(self, country_db):
        ship = Shipment(
            id="S", description="test",
            origin_country="US",
            destination_country="CA",
            transit_countries=["IR"],
        )
        result = CrossBorderAssessor(country_db=country_db).assess(_ctx(ship))
        assert result.severity == Severity.CRITICAL
        assert any(s.name == "sanctioned_transit" for s in result.signals)

    def test_sanctioned_destination_forces_critical(self, country_db):
        ship = Shipment(
            id="S", description="test",
            origin_country="US", destination_country="IR",
        )
        result = CrossBorderAssessor(country_db=country_db).assess(_ctx(ship))
        assert result.severity == Severity.CRITICAL
        assert any(s.name == "sanctioned_destination" for s in result.signals)

    def test_sanctioned_final_destination(self, country_db):
        ship = Shipment(
            id="S", description="test",
            origin_country="US",
            destination_country="AE",
            final_destination="IR",  # declared final destination is sanctioned
        )
        result = CrossBorderAssessor(country_db=country_db).assess(_ctx(ship))
        assert result.severity == Severity.CRITICAL
        # Both "final destination mismatch" and "final destination sanctioned" fire
        names = {s.name for s in result.signals}
        assert "final_destination_mismatch" in names
        assert "final_destination_sanctioned" in names

    def test_transshipment_hub_signal(self, country_db):
        ship = Shipment(
            id="S", description="test",
            origin_country="US",
            destination_country="CA",
            transit_countries=["SG"],
        )
        result = CrossBorderAssessor(country_db=country_db).assess(_ctx(ship))
        assert any("transshipment_hub" in s.name for s in result.signals)

    def test_diversion_neighbor_risk(self, country_db):
        # Shipping to Türkiye which borders sanctioned Iran → diversion risk
        ship = Shipment(
            id="S", description="test",
            origin_country="US", destination_country="TR",
        )
        result = CrossBorderAssessor(country_db=country_db).assess(_ctx(ship))
        assert any(s.name == "diversion_risk_sanctioned_neighbor" for s in result.signals)

    def test_manufacture_origin_mismatch(self, country_db):
        ship = Shipment(
            id="S", description="test",
            origin_country="AE",
            destination_country="US",
            country_of_manufacture="IR",  # manufactured in sanctioned country
        )
        result = CrossBorderAssessor(country_db=country_db).assess(_ctx(ship))
        names = {s.name for s in result.signals}
        assert "manufacture_origin_mismatch" in names
        assert "sanctioned_manufacture" in names
        assert result.severity == Severity.CRITICAL

    def test_fatf_grey_transit(self, country_db):
        ship = Shipment(
            id="S", description="test",
            origin_country="US",
            destination_country="CA",
            transit_countries=["TR"],  # grey-listed transit
        )
        result = CrossBorderAssessor(country_db=country_db).assess(_ctx(ship))
        assert any("fatf_listed" in s.name for s in result.signals)

    def test_many_transit_opacity(self, country_db):
        ship = Shipment(
            id="S", description="test",
            origin_country="US", destination_country="CA",
            transit_countries=["SG", "AE", "HK"],
        )
        result = CrossBorderAssessor(country_db=country_db).assess(_ctx(ship))
        assert any(s.name == "many_transit_stops" for s in result.signals)

    def test_final_destination_mismatch(self, country_db):
        ship = Shipment(
            id="S", description="test",
            origin_country="US",
            destination_country="SG",
            final_destination="AU",
        )
        result = CrossBorderAssessor(country_db=country_db).assess(_ctx(ship))
        assert any(s.name == "final_destination_mismatch" for s in result.signals)

    def test_works_without_db_limited_signals(self):
        # Without a country DB, only heuristic signals fire (mismatches, many transit)
        ship = Shipment(
            id="S", description="test",
            origin_country="US", destination_country="CA",
            country_of_manufacture="CN",
            transit_countries=["X1", "X2", "X3"],
        )
        result = CrossBorderAssessor(country_db=None).assess(_ctx(ship))
        names = {s.name for s in result.signals}
        assert "manufacture_origin_mismatch" in names
        assert "many_transit_stops" in names
        # No sanctioned/FATF signals because no DB
        assert not any(s.name.startswith("sanctioned_") for s in result.signals)

    def test_summary_shows_route(self, country_db):
        ship = Shipment(
            id="S", description="test",
            origin_country="US",
            destination_country="AE",
            transit_countries=["SG"],
            final_destination="IR",
        )
        result = CrossBorderAssessor(country_db=country_db).assess(_ctx(ship))
        assert "US" in result.summary
        assert "AE" in result.summary
        assert "SG" in result.summary
        assert "IR" in result.summary

    def test_dimension_value(self, country_db):
        ship = Shipment(id="S", description="test")
        result = CrossBorderAssessor(country_db=country_db).assess(_ctx(ship))
        assert result.dimension == RiskDimension.CROSS_BORDER


class TestCrossBorderConfig:
    def test_custom_transit_threshold(self, country_db):
        cfg = CrossBorderConfig(many_transit_threshold=1)
        ship = Shipment(
            id="S", description="test",
            origin_country="US", destination_country="CA",
            transit_countries=["SG"],
        )
        result = CrossBorderAssessor(country_db=country_db, config=cfg).assess(_ctx(ship))
        assert any(s.name == "many_transit_stops" for s in result.signals)
