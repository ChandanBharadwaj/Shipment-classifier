"""Entity resolution endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from minerva.api.dependencies import get_pipeline
from minerva.pipeline import ScreeningPipeline
from minerva.schema import Shipment

router = APIRouter()


@router.post("/resolve")
def resolve_entity(
    shipment_data: dict[str, Any],
    pipeline: ScreeningPipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    """Return denied-party matches for the shipment parties (consignee / shipper)."""
    if pipeline.entity_resolver is None:
        raise HTTPException(
            status_code=503,
            detail="Entity resolution not enabled or no denied parties loaded.",
        )
    shipment = Shipment(**shipment_data)
    matches = pipeline.entity_resolver.resolve_shipment(shipment)
    return {
        "shipment_id": shipment.id,
        "matches": [m.model_dump() for m in matches],
        "match_count": len(matches),
    }
