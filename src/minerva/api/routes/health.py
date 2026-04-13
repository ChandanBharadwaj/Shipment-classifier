"""Health check endpoint."""

from fastapi import APIRouter, Depends

from minerva import __version__
from minerva.api.dependencies import get_pipeline
from minerva.api.schemas import HealthResponse
from minerva.pipeline import ScreeningPipeline

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(
    pipeline: ScreeningPipeline = Depends(get_pipeline),
) -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version=__version__,
        active_classifier=pipeline._settings.active_classifier,
        taxonomy_groups=pipeline._taxonomy_index.num_groups,
        taxonomy_phrases=pipeline._taxonomy_index.num_phrases,
    )
