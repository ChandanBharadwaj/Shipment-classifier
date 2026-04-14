"""Risk scoring endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from minerva.api.dependencies import get_pipeline
from minerva.pipeline import ScreeningPipeline
from minerva.schema import Shipment

router = APIRouter()


@router.post("/score")
def score_shipment(
    shipment_data: dict[str, Any],
    pipeline: ScreeningPipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    """Compute a risk assessment for a shipment without running full screening."""
    if pipeline.risk_scorer is None:
        raise HTTPException(
            status_code=503,
            detail="Risk scoring not enabled. Set risk_scoring.enabled=true.",
        )
    shipment = Shipment(**shipment_data)
    assessment = pipeline.risk_scorer.score(shipment)
    return assessment.model_dump()
