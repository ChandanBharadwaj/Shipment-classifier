"""Feedback endpoints for reviewer decisions and disagreement queries."""

from fastapi import APIRouter, Depends, HTTPException

from minerva.api.dependencies import get_feedback_store
from minerva.api.schemas import DisagreementResponse, ReviewerFeedbackRequest
from minerva.feedback.store import FeedbackStore

router = APIRouter()


def _require_store(
    store: FeedbackStore | None = Depends(get_feedback_store),
) -> FeedbackStore:
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Feedback store not configured. Set MINERVA_DB_DSN.",
        )
    return store


@router.post("")
def submit_feedback(
    request: ReviewerFeedbackRequest,
    store: FeedbackStore = Depends(_require_store),
) -> dict[str, str]:
    """Submit reviewer feedback for a screened shipment."""
    store.record_reviewer_feedback(
        shipment_id=request.shipment_id,
        reviewer_id=request.reviewer_id,
        final_decision=request.final_decision,
        notes=request.notes,
    )
    return {"status": "recorded", "shipment_id": request.shipment_id}


@router.get("/disagreements", response_model=DisagreementResponse)
def get_disagreements(
    limit: int = 100,
    offset: int = 0,
    store: FeedbackStore = Depends(_require_store),
) -> DisagreementResponse:
    """List screening records where AI and keyword decisions disagree."""
    records = store.get_disagreements(limit=limit, offset=offset)
    return DisagreementResponse(records=records, total=len(records))
