"""API request/response Pydantic models (separate from domain schema.py)."""

from typing import Any

from pydantic import BaseModel, Field

from minerva.schema import Action, ClassifierLabel, RiskDimension, RiskLevel, Severity


# --- Request models ---


class ScreenRequest(BaseModel):
    id: str
    description: str
    origin_country: str | None = None
    destination_country: str | None = None
    consignee: str | None = None
    shipper: str | None = None
    declared_value: float | None = None
    declared_currency: str | None = None
    hs_code: str | None = None
    weight_kg: float | None = None
    # Cross-border / routing fields
    transit_countries: list[str] = Field(default_factory=list)
    country_of_manufacture: str | None = None
    final_destination: str | None = None
    incoterms: str | None = None
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


class EntityMatchResponse(BaseModel):
    matched_party: str
    input_party: str
    role: str
    score: float
    list_name: str
    exact_match: bool


class RiskSignalResponse(BaseModel):
    name: str
    weight: float
    value: float
    description: str


class RiskAssessmentResponse(BaseModel):
    score: int
    raw_score: float
    signals: list[RiskSignalResponse] = Field(default_factory=list)


class DimensionSignalResponse(BaseModel):
    name: str
    value: float
    weight: float
    evidence: str


class DimensionAssessmentResponse(BaseModel):
    dimension: RiskDimension
    severity: Severity
    score: float
    signals: list[DimensionSignalResponse] = Field(default_factory=list)
    summary: str = ""


class FlagResponse(BaseModel):
    name: str
    severity: Severity
    dimension: RiskDimension
    description: str


class RiskProfileResponse(BaseModel):
    """Multi-dimensional risk profile — the primary officer-facing output.

    The officer owns the decision; `advisory_action` is a non-binding suggestion.
    """

    shipment_id: str
    overall_severity: Severity
    overall_score: float
    dimensions: list[DimensionAssessmentResponse] = Field(default_factory=list)
    flags: list[FlagResponse] = Field(default_factory=list)
    narrative: str = ""
    advisory_action: Action
    advisory_confidence: float


class ScreeningDecisionResponse(BaseModel):
    shipment_id: str
    action: Action
    taxonomy_hits: list[TaxonomyHitResponse] = Field(default_factory=list)
    classification: ClassificationResponse | None = None
    entity_matches: list[EntityMatchResponse] = Field(default_factory=list)
    risk_assessment: RiskAssessmentResponse | None = None
    risk_profile: RiskProfileResponse | None = None
    reason: str = ""
    rationale: str = ""
    model_version: str | None = None
    classifier_type: str | None = None
    thresholds_snapshot: dict[str, Any] = Field(default_factory=dict)


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
