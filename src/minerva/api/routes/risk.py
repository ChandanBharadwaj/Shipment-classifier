"""Risk scoring and profile endpoints.

The profile endpoint surfaces risk across 8 dimensions for the officer.
The system does not take action — it shows evidence; the officer decides.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from minerva.api.dependencies import get_pipeline
from minerva.api.schemas import RiskProfileResponse
from minerva.pipeline import ScreeningPipeline
from minerva.schema import Shipment

router = APIRouter()


@router.post("/score")
def score_shipment(
    shipment_data: dict[str, Any],
    pipeline: ScreeningPipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    """Compute the legacy single-tier risk assessment for a shipment."""
    if pipeline.risk_scorer is None:
        raise HTTPException(
            status_code=503,
            detail="Risk scoring not enabled. Set risk_scoring.enabled=true.",
        )
    shipment = Shipment(**shipment_data)
    assessment = pipeline.risk_scorer.score(shipment)
    return assessment.model_dump()


@router.post("/profile", response_model=RiskProfileResponse)
def build_risk_profile(
    shipment_data: dict[str, Any],
    pipeline: ScreeningPipeline = Depends(get_pipeline),
) -> RiskProfileResponse:
    """Return the multi-dimensional risk profile for a shipment.

    Runs the full pipeline up to Layer 3, then emits every dimension
    (goods, party, geography, valuation, hs_code, data_quality,
    model_uncertainty, dual_use) with evidence and severity. The
    advisory_action is a non-binding suggestion; the officer decides.
    """
    shipment = Shipment(**shipment_data)
    profile = pipeline.build_profile(shipment)
    return RiskProfileResponse(**profile.model_dump())
