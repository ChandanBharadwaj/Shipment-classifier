"""FastAPI dependency injection for pipeline and feedback store singletons."""

from minerva.feedback.store import FeedbackStore
from minerva.pipeline import ScreeningPipeline

# Module-level singletons, set during app lifespan
_pipeline: ScreeningPipeline | None = None
_feedback_store: FeedbackStore | None = None


def set_pipeline(pipeline: ScreeningPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def set_feedback_store(store: FeedbackStore) -> None:
    global _feedback_store
    _feedback_store = store


def get_pipeline() -> ScreeningPipeline:
    if _pipeline is None:
        raise RuntimeError("Screening pipeline not initialized")
    return _pipeline


def get_feedback_store() -> FeedbackStore | None:
    return _feedback_store
