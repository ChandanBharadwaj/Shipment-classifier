"""FastAPI application factory with lifespan management."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from minerva import __version__
from minerva.api.dependencies import set_feedback_store, set_pipeline
from minerva.api.routes import feedback, health, screening, taxonomy
from minerva.config import MinervaSettings
from minerva.feedback.store import FeedbackStore
from minerva.logging_utils import get_logger, setup_logging
from minerva.pipeline import ScreeningPipeline

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize pipeline and feedback store on startup, clean up on shutdown."""
    settings = MinervaSettings()
    setup_logging(settings.log_level)

    logger.info("Starting Minerva API v%s", __version__)

    # Initialize screening pipeline
    pipeline = ScreeningPipeline(settings)
    set_pipeline(pipeline)

    # Initialize feedback store if DB is configured
    store = None
    if settings.db_dsn:
        store = FeedbackStore(settings.db_dsn)
        store.init_schema()
        set_feedback_store(store)
        logger.info("Feedback store connected")
    else:
        logger.info("No DB configured — feedback store disabled")

    yield

    # Cleanup
    if store:
        store.close()
    logger.info("Minerva API shut down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Minerva Hybrid Screening API",
        description="Trade compliance shipment screening with hybrid taxonomy + AI classification",
        version=__version__,
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(screening.router, prefix="/screen", tags=["screening"])
    app.include_router(taxonomy.router, prefix="/taxonomy", tags=["taxonomy"])
    app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])

    return app
