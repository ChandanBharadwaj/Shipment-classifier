"""Country risk database.

Holds per-country risk intelligence (e.g. from LexisNexis WorldCompliance /
CLEAR exports, OFAC data, FATF lists, Basel AML Index). Provides a stable
interface for assessors regardless of upstream data source.

Supports two ingestion formats:
    1. Native Minerva JSON format (config/country_risk.json)
    2. CSV exports — column names are configurable so any vendor format fits

Country codes are normalized to ISO 3166-1 alpha-2 uppercase internally.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CountryRiskLevel(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SanctionsStatus(str, Enum):
    NONE = "none"
    SECTORAL = "sectoral"
    COMPREHENSIVE = "comprehensive"


class FatfStatus(str, Enum):
    COMPLIANT = "compliant"
    GREY = "grey"
    BLACK = "black"
    UNKNOWN = "unknown"


class CountryRiskEntry(BaseModel):
    """Risk intelligence for a single country.

    Designed to hold the subset of fields most compliance teams extract from
    LexisNexis / WorldCompliance country risk intelligence. Extra vendor
    fields can be stashed in `extra`.
    """

    country_code: str                                    # ISO 3166-1 alpha-2
    country_name: str | None = None
    overall_risk: CountryRiskLevel = CountryRiskLevel.UNKNOWN
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sanctions_status: SanctionsStatus = SanctionsStatus.NONE
    fatf_status: FatfStatus = FatfStatus.UNKNOWN
    pep_exposure: CountryRiskLevel = CountryRiskLevel.UNKNOWN
    corruption_risk: CountryRiskLevel = CountryRiskLevel.UNKNOWN
    financial_crime_risk: CountryRiskLevel = CountryRiskLevel.UNKNOWN
    transshipment_hub: bool = False
    sanctioned_neighbors: list[str] = Field(default_factory=list)
    source: str = "lexisnexis"
    as_of: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}

    @field_validator("country_code", mode="before")
    @classmethod
    def _normalize_code(cls, v: str) -> str:
        return str(v).strip().upper()

    @field_validator("sanctioned_neighbors", mode="before")
    @classmethod
    def _normalize_neighbors(cls, v: list[str]) -> list[str]:
        return [str(c).strip().upper() for c in v if c]

    @property
    def is_sanctioned(self) -> bool:
        return self.sanctions_status != SanctionsStatus.NONE

    @property
    def is_fatf_listed(self) -> bool:
        return self.fatf_status in (FatfStatus.GREY, FatfStatus.BLACK)


class CountryRiskDatabase:
    """Lookup interface over a set of country risk entries."""

    def __init__(self, entries: list[CountryRiskEntry]) -> None:
        self._by_code: dict[str, CountryRiskEntry] = {}
        for entry in entries:
            self._by_code[entry.country_code] = entry

    @property
    def size(self) -> int:
        return len(self._by_code)

    def get(self, country_code: str | None) -> CountryRiskEntry | None:
        if not country_code:
            return None
        return self._by_code.get(country_code.strip().upper())

    def is_high_risk(self, country_code: str | None) -> bool:
        entry = self.get(country_code)
        if entry is None:
            return False
        return entry.overall_risk in (
            CountryRiskLevel.HIGH,
            CountryRiskLevel.CRITICAL,
        )

    def is_sanctioned(self, country_code: str | None) -> bool:
        entry = self.get(country_code)
        return entry is not None and entry.is_sanctioned

    def is_transshipment_hub(self, country_code: str | None) -> bool:
        entry = self.get(country_code)
        return entry is not None and entry.transshipment_hub

    def is_fatf_listed(self, country_code: str | None) -> bool:
        entry = self.get(country_code)
        return entry is not None and entry.is_fatf_listed

    def sanctioned_neighbors(self, country_code: str | None) -> list[str]:
        entry = self.get(country_code)
        return list(entry.sanctioned_neighbors) if entry else []


# --------- Loaders ---------

def load_json(path: Path) -> CountryRiskDatabase:
    """Load country risk data from the native Minerva JSON format.

    Expected format:
        {
            "source": "lexisnexis",
            "as_of": "2026-04-01",
            "countries": [
                {"country_code": "US", "overall_risk": "low", ...},
                ...
            ]
        }
    A bare list of country objects is also accepted.
    """
    if not path.exists():
        return CountryRiskDatabase(entries=[])

    data = json.loads(path.read_text())
    if isinstance(data, list):
        countries = data
        default_source = "unknown"
        default_as_of = None
    else:
        countries = data.get("countries", [])
        default_source = data.get("source", "unknown")
        default_as_of = data.get("as_of")

    entries = []
    for entry in countries:
        if "country_code" not in entry:
            continue
        entry_data = dict(entry)
        entry_data.setdefault("source", default_source)
        if default_as_of and "as_of" not in entry_data:
            entry_data["as_of"] = default_as_of
        entries.append(CountryRiskEntry(**entry_data))
    return CountryRiskDatabase(entries=entries)


@dataclass
class CsvColumnMapping:
    """Column name mapping for CSV ingestion.

    Defaults match a common LexisNexis-style export. Override any field
    to match a different vendor format.
    """

    country_code: str = "country_code"
    country_name: str = "country_name"
    overall_risk: str = "risk_tier"
    overall_score: str = "risk_score"
    sanctions_status: str = "sanctions_status"
    fatf_status: str = "fatf_status"
    pep_exposure: str = "pep_exposure"
    corruption_risk: str = "corruption_risk"
    financial_crime_risk: str = "financial_crime_risk"
    transshipment_hub: str = "transshipment_hub"
    sanctioned_neighbors: str = "sanctioned_neighbors"
    # Delimiter for list-type fields encoded in a single cell
    list_delimiter: str = "|"


def _parse_risk_level(value: str | None) -> CountryRiskLevel:
    if value is None:
        return CountryRiskLevel.UNKNOWN
    normalized = str(value).strip().lower()
    for level in CountryRiskLevel:
        if level.value == normalized:
            return level
    # Common vendor aliases
    if normalized in ("severe", "very high", "very_high"):
        return CountryRiskLevel.CRITICAL
    if normalized in ("moderate",):
        return CountryRiskLevel.MEDIUM
    if normalized in ("minimal", "very low", "very_low"):
        return CountryRiskLevel.LOW
    return CountryRiskLevel.UNKNOWN


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t")


def _parse_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    # If looks like a 0-100 scale, normalize
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _parse_sanctions(value: str | None) -> SanctionsStatus:
    if value is None:
        return SanctionsStatus.NONE
    normalized = str(value).strip().lower()
    for status in SanctionsStatus:
        if status.value == normalized:
            return status
    if normalized in ("sanctioned", "full", "total"):
        return SanctionsStatus.COMPREHENSIVE
    if normalized in ("partial", "sector"):
        return SanctionsStatus.SECTORAL
    return SanctionsStatus.NONE


def _parse_fatf(value: str | None) -> FatfStatus:
    if value is None:
        return FatfStatus.UNKNOWN
    normalized = str(value).strip().lower()
    for status in FatfStatus:
        if status.value == normalized:
            return status
    if normalized in ("blacklist", "increased monitoring"):
        return FatfStatus.BLACK
    if normalized in ("greylist", "enhanced monitoring"):
        return FatfStatus.GREY
    if normalized in ("clear", "ok", ""):
        return FatfStatus.COMPLIANT
    return FatfStatus.UNKNOWN


def load_csv(
    path: Path,
    mapping: CsvColumnMapping | None = None,
    source: str = "lexisnexis",
    as_of: str | None = None,
) -> CountryRiskDatabase:
    """Load country risk data from a CSV file.

    Column names are configurable via `mapping` so any vendor-specific
    LexisNexis-style export (WorldCompliance, CLEAR, Accuity, Dow Jones,
    Refinitiv) can be ingested without code changes.

    Missing columns are tolerated — fields default to UNKNOWN or 0.
    """
    if not path.exists():
        return CountryRiskDatabase(entries=[])

    mapping = mapping or CsvColumnMapping()
    entries: list[CountryRiskEntry] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get(mapping.country_code, "").strip()
            if not code:
                continue

            neighbors_raw = row.get(mapping.sanctioned_neighbors, "") or ""
            neighbors = [
                n.strip()
                for n in neighbors_raw.split(mapping.list_delimiter)
                if n.strip()
            ]

            entry = CountryRiskEntry(
                country_code=code,
                country_name=row.get(mapping.country_name) or None,
                overall_risk=_parse_risk_level(row.get(mapping.overall_risk)),
                overall_score=_parse_float(row.get(mapping.overall_score)),
                sanctions_status=_parse_sanctions(row.get(mapping.sanctions_status)),
                fatf_status=_parse_fatf(row.get(mapping.fatf_status)),
                pep_exposure=_parse_risk_level(row.get(mapping.pep_exposure)),
                corruption_risk=_parse_risk_level(row.get(mapping.corruption_risk)),
                financial_crime_risk=_parse_risk_level(
                    row.get(mapping.financial_crime_risk)
                ),
                transshipment_hub=_parse_bool(row.get(mapping.transshipment_hub)),
                sanctioned_neighbors=neighbors,
                source=source,
                as_of=as_of,
            )
            entries.append(entry)

    return CountryRiskDatabase(entries=entries)


def load_country_risk(
    path: Path,
    csv_mapping: CsvColumnMapping | None = None,
) -> CountryRiskDatabase:
    """Auto-dispatch loader based on file extension."""
    if not path.exists():
        return CountryRiskDatabase(entries=[])
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path, mapping=csv_mapping)
    return load_json(path)
