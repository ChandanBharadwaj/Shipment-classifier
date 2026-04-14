"""Screening endpoints: single and batch shipment screening."""

from collections import Counter

from fastapi import APIRouter, Depends

from minerva.api.dependencies import get_feedback_store, get_pipeline
from minerva.api.schemas import (
    BatchScreenRequest,
    BatchScreenResponse,
    ScreeningDecisionResponse,
    ScreenRequest,
)
from minerva.feedback.store import FeedbackStore
from minerva.pipeline import ScreeningPipeline
from minerva.schema import Shipment

router = APIRouter()


def _to_shipment(request: ScreenRequest) -> Shipment:
    return Shipment(
        id=request.id,
        description=request.description,
        origin_country=request.origin_country,
        destination_country=request.destination_country,
        consignee=request.consignee,
        shipper=request.shipper,
        declared_value=request.declared_value,
        declared_currency=request.declared_currency,
        hs_code=request.hs_code,
        weight_kg=request.weight_kg,
        transit_countries=request.transit_countries,
        country_of_manufacture=request.country_of_manufacture,
        final_destination=request.final_destination,
        incoterms=request.incoterms,
        metadata=request.metadata,
    )


@router.post("", response_model=ScreeningDecisionResponse)
def screen_shipment(
    request: ScreenRequest,
    pipeline: ScreeningPipeline = Depends(get_pipeline),
    store: FeedbackStore | None = Depends(get_feedback_store),
) -> ScreeningDecisionResponse:
    """Screen a single shipment."""
    shipment = _to_shipment(request)
    decision = pipeline.screen(shipment)

    if store:
        store.record_decisions([decision], [shipment])

    return ScreeningDecisionResponse(**decision.model_dump())


@router.post("/batch", response_model=BatchScreenResponse)
def screen_batch(
    request: BatchScreenRequest,
    pipeline: ScreeningPipeline = Depends(get_pipeline),
    store: FeedbackStore | None = Depends(get_feedback_store),
) -> BatchScreenResponse:
    """Screen a batch of shipments."""
    shipments = [_to_shipment(s) for s in request.shipments]
    decisions = pipeline.screen_batch(shipments)

    if store:
        store.record_decisions(decisions, shipments)

    summary = Counter(d.action.value for d in decisions)

    return BatchScreenResponse(
        decisions=[
            ScreeningDecisionResponse(**d.model_dump()) for d in decisions
        ],
        total=len(decisions),
        summary=dict(summary),
    )
