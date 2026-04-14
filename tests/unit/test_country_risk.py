"""Tests for country risk data loading (LexisNexis-style)."""

import json

import pytest

from minerva.risk.country_risk import (
    CountryRiskDatabase,
    CountryRiskEntry,
    CountryRiskLevel,
    CsvColumnMapping,
    FatfStatus,
    SanctionsStatus,
    load_country_risk,
    load_csv,
    load_json,
)


class TestCountryRiskEntry:
    def test_normalizes_country_code(self):
        entry = CountryRiskEntry(country_code="us")
        assert entry.country_code == "US"

    def test_is_sanctioned_property(self):
        entry = CountryRiskEntry(
            country_code="IR",
            sanctions_status=SanctionsStatus.COMPREHENSIVE,
        )
        assert entry.is_sanctioned is True

    def test_not_sanctioned(self):
        entry = CountryRiskEntry(country_code="US")
        assert entry.is_sanctioned is False

    def test_fatf_listed_property(self):
        entry = CountryRiskEntry(country_code="IR", fatf_status=FatfStatus.BLACK)
        assert entry.is_fatf_listed is True
        entry2 = CountryRiskEntry(country_code="AE", fatf_status=FatfStatus.GREY)
        assert entry2.is_fatf_listed is True
        entry3 = CountryRiskEntry(country_code="US", fatf_status=FatfStatus.COMPLIANT)
        assert entry3.is_fatf_listed is False

    def test_sanctioned_neighbors_normalized(self):
        entry = CountryRiskEntry(
            country_code="TR",
            sanctioned_neighbors=["ir", "SY", " kp "],
        )
        assert entry.sanctioned_neighbors == ["IR", "SY", "KP"]


class TestCountryRiskDatabase:
    def _db(self):
        entries = [
            CountryRiskEntry(
                country_code="US",
                overall_risk=CountryRiskLevel.LOW,
            ),
            CountryRiskEntry(
                country_code="IR",
                overall_risk=CountryRiskLevel.CRITICAL,
                sanctions_status=SanctionsStatus.COMPREHENSIVE,
            ),
            CountryRiskEntry(
                country_code="SG",
                overall_risk=CountryRiskLevel.LOW,
                transshipment_hub=True,
            ),
            CountryRiskEntry(
                country_code="TR",
                overall_risk=CountryRiskLevel.MEDIUM,
                sanctioned_neighbors=["IR"],
            ),
        ]
        return CountryRiskDatabase(entries)

    def test_get_normalizes_case(self):
        db = self._db()
        assert db.get("us") is not None
        assert db.get("US") is not None
        assert db.get(None) is None

    def test_is_high_risk(self):
        db = self._db()
        assert db.is_high_risk("IR")
        assert not db.is_high_risk("US")
        assert not db.is_high_risk("ZZ")  # unknown country

    def test_is_sanctioned(self):
        db = self._db()
        assert db.is_sanctioned("IR")
        assert not db.is_sanctioned("US")

    def test_is_transshipment_hub(self):
        db = self._db()
        assert db.is_transshipment_hub("SG")
        assert not db.is_transshipment_hub("US")

    def test_sanctioned_neighbors(self):
        db = self._db()
        assert db.sanctioned_neighbors("TR") == ["IR"]
        assert db.sanctioned_neighbors("US") == []


class TestLoadJSON:
    def test_load_with_countries_wrapper(self, tmp_path):
        data = {
            "source": "lexisnexis",
            "as_of": "2026-01-01",
            "countries": [
                {"country_code": "US", "overall_risk": "low"},
                {"country_code": "IR", "overall_risk": "critical",
                 "sanctions_status": "comprehensive"},
            ],
        }
        path = tmp_path / "cr.json"
        path.write_text(json.dumps(data))

        db = load_json(path)
        assert db.size == 2
        assert db.is_sanctioned("IR")
        assert db.get("US").source == "lexisnexis"
        assert db.get("US").as_of == "2026-01-01"

    def test_load_bare_list(self, tmp_path):
        data = [
            {"country_code": "US", "overall_risk": "low"},
            {"country_code": "IR", "overall_risk": "critical"},
        ]
        path = tmp_path / "cr.json"
        path.write_text(json.dumps(data))

        db = load_json(path)
        assert db.size == 2

    def test_missing_file_returns_empty_db(self, tmp_path):
        db = load_json(tmp_path / "nothing.json")
        assert db.size == 0
        assert db.get("US") is None

    def test_skips_entries_without_country_code(self, tmp_path):
        data = {
            "countries": [
                {"country_code": "US"},
                {"country_name": "Missing Code"},
            ]
        }
        path = tmp_path / "cr.json"
        path.write_text(json.dumps(data))
        db = load_json(path)
        assert db.size == 1


class TestLoadCSV:
    def test_load_default_mapping(self, tmp_path):
        csv_text = (
            "country_code,country_name,risk_tier,risk_score,"
            "sanctions_status,fatf_status,transshipment_hub,sanctioned_neighbors\n"
            "US,United States,low,15,none,compliant,false,\n"
            "IR,Iran,critical,100,comprehensive,black,false,\n"
            "TR,Türkiye,medium,55,none,grey,false,IR|SY\n"
            "SG,Singapore,low,22,none,compliant,true,\n"
        )
        path = tmp_path / "cr.csv"
        path.write_text(csv_text)

        db = load_csv(path)
        assert db.size == 4
        assert db.is_sanctioned("IR")
        assert db.get("IR").fatf_status == FatfStatus.BLACK
        assert db.get("TR").sanctioned_neighbors == ["IR", "SY"]
        assert db.get("SG").transshipment_hub is True

    def test_load_with_custom_mapping(self, tmp_path):
        csv_text = (
            "iso2,name,tier,score100,sanct\n"
            "US,United States,low,15,none\n"
            "IR,Iran,critical,100,comprehensive\n"
        )
        path = tmp_path / "custom.csv"
        path.write_text(csv_text)

        mapping = CsvColumnMapping(
            country_code="iso2",
            country_name="name",
            overall_risk="tier",
            overall_score="score100",
            sanctions_status="sanct",
        )
        db = load_csv(path, mapping=mapping)
        assert db.size == 2
        assert db.is_sanctioned("IR")
        assert db.get("US").overall_risk == CountryRiskLevel.LOW

    def test_0_to_100_score_normalized(self, tmp_path):
        csv_text = "country_code,risk_score\nUS,15\nIR,100\n"
        path = tmp_path / "c.csv"
        path.write_text(csv_text)
        db = load_csv(path)
        assert db.get("US").overall_score == 0.15
        assert db.get("IR").overall_score == 1.0

    def test_missing_columns_tolerated(self, tmp_path):
        csv_text = "country_code\nUS\nIR\n"
        path = tmp_path / "c.csv"
        path.write_text(csv_text)
        db = load_csv(path)
        assert db.size == 2
        assert db.get("US").overall_risk == CountryRiskLevel.UNKNOWN

    def test_vendor_aliases_parsed(self, tmp_path):
        csv_text = (
            "country_code,risk_tier,fatf_status,sanctions_status\n"
            "XX,severe,enhanced monitoring,sanctioned\n"
        )
        path = tmp_path / "c.csv"
        path.write_text(csv_text)
        db = load_csv(path)
        entry = db.get("XX")
        assert entry.overall_risk == CountryRiskLevel.CRITICAL
        assert entry.fatf_status == FatfStatus.GREY
        assert entry.sanctions_status == SanctionsStatus.COMPREHENSIVE

    def test_missing_file_returns_empty(self, tmp_path):
        db = load_csv(tmp_path / "nothing.csv")
        assert db.size == 0


class TestLoadCountryRisk:
    def test_auto_dispatches_json(self, tmp_path):
        path = tmp_path / "cr.json"
        path.write_text(json.dumps({"countries": [{"country_code": "US"}]}))
        db = load_country_risk(path)
        assert db.size == 1

    def test_auto_dispatches_csv(self, tmp_path):
        path = tmp_path / "cr.csv"
        path.write_text("country_code\nUS\n")
        db = load_country_risk(path)
        assert db.size == 1

    def test_missing_returns_empty(self, tmp_path):
        db = load_country_risk(tmp_path / "nothing.txt")
        assert db.size == 0
