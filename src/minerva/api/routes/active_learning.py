"""Active learning endpoints — select most informative items for reviewer labeling."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from minerva.active_learning.sampler import select_for_review

router = APIRouter()


class ActiveLearningRequest(BaseModel):
    """Inputs for selecting review targets.

    embeddings: list of per-item embedding vectors (2D).
    probabilities: list of per-item class probability vectors (2D).
    budget: number of items to select.
    """

    embeddings: list[list[float]]
    probabilities: list[list[float]]
    budget: int = Field(gt=0, default=10)
    margin_weight: float = 0.5
    entropy_weight: float = 0.5


class ActiveLearningResponse(BaseModel):
    selected_indices: list[int]
    count: int


@router.post("/select", response_model=ActiveLearningResponse)
def select_for_labeling(request: ActiveLearningRequest) -> ActiveLearningResponse:
    """Select the most informative items for reviewer labeling."""
    emb = np.array(request.embeddings, dtype=np.float32)
    probs = np.array(request.probabilities, dtype=np.float32)

    if emb.shape[0] != probs.shape[0]:
        raise HTTPException(
            status_code=400,
            detail="embeddings and probabilities must have the same number of rows",
        )
    if emb.shape[0] == 0:
        return ActiveLearningResponse(selected_indices=[], count=0)

    indices = select_for_review(
        embeddings=emb,
        probabilities=probs,
        budget=request.budget,
        margin_weight=request.margin_weight,
        entropy_weight=request.entropy_weight,
    )
    return ActiveLearningResponse(selected_indices=indices, count=len(indices))
