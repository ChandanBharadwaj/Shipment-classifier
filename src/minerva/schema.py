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


class Shipment(BaseModel):
    id: str
    description: str
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


class ScreeningDecision(BaseModel):
    shipment_id: str
    action: Action
    taxonomy_hits: list[TaxonomyHit] = Field(default_factory=list)
    classification: ClassificationResult | None = None
    reason: str = ""

    model_config = {"frozen": True}
