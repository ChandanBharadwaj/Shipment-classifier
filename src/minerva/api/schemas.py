"""API request/response Pydantic models (separate from domain schema.py)."""

from typing import Any

from pydantic import BaseModel, Field

from minerva.schema import Action, ClassifierLabel, RiskLevel


# --- Request models ---


class ScreenRequest(BaseModel):
    id: str
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchScreenRequest(BaseModel):
    shipments: list[ScreenRequest]


class ReviewerFeedbackRequest(BaseModel):
    shipment_id: str
    reviewer_id: str
    final_decision: str
    notes: str | None = None


class TaxonomyGroupRequest(BaseModel):
    group_id: str
    name: str
    risk_level: RiskLevel
    semantic_phrases: list[str]
    hard_keywords: list[str] = Field(default_factory=list)
    allow_auto_clear: bool = False


class TaxonomyUpdateRequest(BaseModel):
    version: str
    groups: list[TaxonomyGroupRequest]


# --- Response models ---


class TaxonomyHitResponse(BaseModel):
    group_id: str
    group_name: str
    risk_level: RiskLevel
    matched_phrase: str | None = None
    matched_keyword: str | None = None
    similarity_score: float | None = None


class ClassificationResponse(BaseModel):
    label: ClassifierLabel
    confidence: float


class ScreeningDecisionResponse(BaseModel):
    shipment_id: str
    action: Action
    taxonomy_hits: list[TaxonomyHitResponse] = Field(default_factory=list)
    classification: ClassificationResponse | None = None
    reason: str = ""


class BatchScreenResponse(BaseModel):
    decisions: list[ScreeningDecisionResponse]
    total: int
    summary: dict[str, int]


class TaxonomyGroupResponse(BaseModel):
    group_id: str
    name: str
    risk_level: RiskLevel
    semantic_phrases: list[str]
    hard_keywords: list[str]
    allow_auto_clear: bool


class TaxonomyResponse(BaseModel):
    version: str
    groups: list[TaxonomyGroupResponse]


class DisagreementResponse(BaseModel):
    records: list[dict[str, Any]]
    total: int


class HealthResponse(BaseModel):
    status: str
    version: str
    active_classifier: str
    taxonomy_groups: int
    taxonomy_phrases: int
