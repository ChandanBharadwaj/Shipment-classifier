"""Taxonomy management endpoints."""

import json

from fastapi import APIRouter, Depends, HTTPException

from minerva.api.dependencies import get_pipeline
from minerva.api.schemas import (
    TaxonomyGroupResponse,
    TaxonomyResponse,
    TaxonomyUpdateRequest,
)
from minerva.pipeline import ScreeningPipeline
from minerva.taxonomy.embeddings import TaxonomyEmbeddingIndex
from minerva.taxonomy.loader import load_taxonomy
from minerva.taxonomy.matcher import TaxonomyMatcher

router = APIRouter()


@router.get("", response_model=TaxonomyResponse)
def get_taxonomy(
    pipeline: ScreeningPipeline = Depends(get_pipeline),
) -> TaxonomyResponse:
    """Return the current taxonomy configuration."""
    groups = pipeline._taxonomy_index.groups
    return TaxonomyResponse(
        version="1.0",
        groups=[
            TaxonomyGroupResponse(
                group_id=g.group_id,
                name=g.name,
                risk_level=g.risk_level,
                semantic_phrases=g.semantic_phrases,
                hard_keywords=g.hard_keywords,
                allow_auto_clear=g.allow_auto_clear,
            )
            for g in groups
        ],
    )


@router.put("")
def update_taxonomy(
    request: TaxonomyUpdateRequest,
    pipeline: ScreeningPipeline = Depends(get_pipeline),
) -> dict[str, str]:
    """Update taxonomy and re-compute embeddings.

    Writes the new taxonomy to the config file and rebuilds the
    embedding index and matcher in-place.
    """
    # Write updated taxonomy to config
    taxonomy_data = {
        "version": request.version,
        "groups": [g.model_dump() for g in request.groups],
    }

    taxonomy_path = pipeline._settings.taxonomy_path
    with open(taxonomy_path, "w") as f:
        json.dump(taxonomy_data, f, indent=2)

    # Reload taxonomy
    try:
        groups = load_taxonomy(taxonomy_path)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Rebuild index and matcher
    new_index = TaxonomyEmbeddingIndex(
        groups=groups, model=pipeline._embedding_model
    )
    new_matcher = TaxonomyMatcher(
        index=new_index,
        model=pipeline._embedding_model,
        settings=pipeline._settings,
    )

    # Swap in-place
    pipeline._taxonomy_index = new_index
    pipeline._matcher = new_matcher

    return {
        "status": "updated",
        "groups": str(len(groups)),
        "phrases": str(new_index.num_phrases),
    }
