"""Domain types for the Minerva Hybrid Screening Framework."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"


class Action(str, Enum):
    BLOCK = "block"
    APPROVE = "approve"
    MANUAL_REVIEW = "manual_review"


class ClassifierLabel(str, Enum):
    ALLOWED = "allowed"
    RESTRICTED = "restricted"
    NEEDS_REVIEW = "needs_review"


class RiskScore(int, Enum):
    """Legacy single-tier risk score (1-5)."""

    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class Severity(str, Enum):
    """Severity level used throughout the multi-dimensional risk profile."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANKS[self]


_SEVERITY_RANKS = {
    Severity.NONE: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class RiskDimension(str, Enum):
    """Distinct aspects of risk surfaced in the risk profile."""

    GOODS = "goods"                        # What's being shipped
    PARTY = "party"                        # Who is involved
    GEOGRAPHY = "geography"                # Where it's going / from
    VALUATION = "valuation"                # Declared value anomalies
    HS_CODE = "hs_code"                    # Tariff classification risk
    DATA_QUALITY = "data_quality"          # Completeness & coherence
    MODEL_UNCERTAINTY = "model_uncertainty"  # How confident the AI is
    DUAL_USE = "dual_use"                  # End-use / catch-all concerns
    CROSS_BORDER = "cross_border"          # Routing, transit, transshipment, diversion


class Shipment(BaseModel):
    """Shipment to be screened."""

    id: str
    description: str

    # Structured features
    origin_country: str | None = None
    destination_country: str | None = None
    consignee: str | None = None
    shipper: str | None = None
    declared_value: float | None = None
    declared_currency: str | None = None
    hs_code: str | None = None
    weight_kg: float | None = None

    # Routing / cross-border fields
    transit_countries: list[str] = Field(default_factory=list)
    country_of_manufacture: str | None = None
    final_destination: str | None = None
    incoterms: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class TaxonomyGroup(BaseModel):
    group_id: str
    name: str
    risk_level: RiskLevel
    semantic_phrases: list[str]
    hard_keywords: list[str] = Field(default_factory=list)
    allow_auto_clear: bool = False

    model_config = {"frozen": True}


class TaxonomyHit(BaseModel):
    group_id: str
    group_name: str
    risk_level: RiskLevel
    matched_phrase: str | None = None
    matched_keyword: str | None = None
    similarity_score: float | None = None

    model_config = {"frozen": True}


class ClassificationResult(BaseModel):
    label: ClassifierLabel
    confidence: float = Field(ge=0.0, le=1.0)

    model_config = {"frozen": True}


class EntityMatch(BaseModel):
    matched_party: str
    input_party: str
    role: str
    score: float
    list_name: str
    exact_match: bool = False

    model_config = {"frozen": True}


class RiskSignal(BaseModel):
    """Legacy signal type used by the single-tier RiskScorer."""

    name: str
    weight: float
    value: float
    description: str = ""

    model_config = {"frozen": True}


class RiskAssessment(BaseModel):
    """Legacy single-tier risk assessment."""

    score: RiskScore
    raw_score: float = Field(ge=0.0, le=1.0)
    signals: list[RiskSignal] = Field(default_factory=list)

    model_config = {"frozen": True}


# --- Multi-dimensional risk profile types ---


class DimensionSignal(BaseModel):
    """A single piece of evidence contributing to a dimension assessment."""

    name: str
    value: float = Field(ge=0.0, le=1.0)
    weight: float = 1.0
    evidence: str                   # human-readable evidence

    model_config = {"frozen": True}


class DimensionAssessment(BaseModel):
    """Assessment of one risk dimension for a shipment."""

    dimension: RiskDimension
    severity: Severity
    score: float = Field(ge=0.0, le=1.0)
    signals: list[DimensionSignal] = Field(default_factory=list)
    summary: str = ""               # one-line summary

    model_config = {"frozen": True}


class Flag(BaseModel):
    """A hard red-flag finding independent of dimension scores.

    Flags surface the officer's attention to specific, non-negotiable evidence
    (e.g. a taxonomy keyword hit or an exact denied-party match).
    """

    name: str                       # e.g. "taxonomy_keyword_hit"
    severity: Severity
    dimension: RiskDimension
    description: str

    model_config = {"frozen": True}


class RiskProfile(BaseModel):
    """Multi-dimensional risk profile for a shipment.

    This is the primary output surfaced to compliance officers.
    The system does not take action — the officer decides. The
    `advisory_action` is a non-binding suggestion.
    """

    shipment_id: str
    overall_severity: Severity
    overall_score: float = Field(ge=0.0, le=1.0)
    dimensions: list[DimensionAssessment] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    narrative: str = ""

    # Advisory (non-authoritative) - officer owns the decision
    advisory_action: Action = Action.MANUAL_REVIEW
    advisory_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = {"frozen": True}

    def get_dimension(
        self, dimension: RiskDimension
    ) -> DimensionAssessment | None:
        for d in self.dimensions:
            if d.dimension == dimension:
                return d
        return None


class ScreeningDecision(BaseModel):
    """Screening output including routing action and full risk profile.

    NOTE: `action` is produced by the router as an automated decision.
    In advisory deployments, the `risk_profile.advisory_action` is the
    officer-facing suggestion and the officer makes the final call.
    """

    shipment_id: str
    action: Action
    taxonomy_hits: list[TaxonomyHit] = Field(default_factory=list)
    classification: ClassificationResult | None = None
    entity_matches: list[EntityMatch] = Field(default_factory=list)
    risk_assessment: RiskAssessment | None = None
    risk_profile: RiskProfile | None = None
    reason: str = ""
    rationale: str = ""

    # Audit trail
    model_version: str | None = None
    classifier_type: str | None = None
    thresholds_snapshot: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}
