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
    """Risk tier from 1 (low) to 5 (critical)."""

    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class Shipment(BaseModel):
    """Shipment to be screened.

    `description` is the required free-text field.
    All other fields are optional structured signals used by the risk scorer
    and entity resolution layer.
    """

    id: str
    description: str

    # Structured features (optional, used by risk scorer and entity resolution)
    origin_country: str | None = None
    destination_country: str | None = None
    consignee: str | None = None
    shipper: str | None = None
    declared_value: float | None = None
    declared_currency: str | None = None
    hs_code: str | None = None
    weight_kg: float | None = None

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
    """A denied-party / watchlist match against a shipment party."""

    matched_party: str            # name of the denied party matched
    input_party: str              # name as appeared on the shipment
    role: str                     # "consignee" | "shipper"
    score: float                  # fuzzy match score 0.0 - 1.0
    list_name: str                # source list identifier
    exact_match: bool = False

    model_config = {"frozen": True}


class RiskSignal(BaseModel):
    """A single risk signal contributing to the overall score."""

    name: str                     # e.g. "high_risk_origin_country"
    weight: float                 # contribution weight
    value: float                  # signal value (0.0 - 1.0)
    description: str = ""         # human-readable explanation

    model_config = {"frozen": True}


class RiskAssessment(BaseModel):
    """Multi-feature risk assessment for a shipment."""

    score: RiskScore
    raw_score: float = Field(ge=0.0, le=1.0)
    signals: list[RiskSignal] = Field(default_factory=list)

    model_config = {"frozen": True}


class ScreeningDecision(BaseModel):
    shipment_id: str
    action: Action
    taxonomy_hits: list[TaxonomyHit] = Field(default_factory=list)
    classification: ClassificationResult | None = None
    entity_matches: list[EntityMatch] = Field(default_factory=list)
    risk_assessment: RiskAssessment | None = None
    reason: str = ""
    rationale: str = ""           # Detailed human-readable explanation for auditors

    # Audit trail fields
    model_version: str | None = None
    classifier_type: str | None = None
    thresholds_snapshot: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}
