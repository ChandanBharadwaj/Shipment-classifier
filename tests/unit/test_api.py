"""Tests for FastAPI routes using TestClient with mocked pipeline."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from minerva.api.dependencies import set_feedback_store, set_pipeline
from minerva.api.routes import feedback, health, screening, taxonomy
from minerva.schema import (
    Action,
    ClassificationResult,
    ClassifierLabel,
    ScreeningDecision,
)


@pytest.fixture
def mock_pipeline():
    """Create a mock screening pipeline."""
    pipeline = MagicMock()
    pipeline._settings = MagicMock()
    pipeline._settings.active_classifier = "nli"
    pipeline._taxonomy_index = MagicMock()
    pipeline._taxonomy_index.num_groups = 3
    pipeline._taxonomy_index.num_phrases = 8
    pipeline._taxonomy_index.groups = []
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """Create a test client with mocked dependencies, bypassing real lifespan."""

    @asynccontextmanager
    async def mock_lifespan(app: FastAPI) -> AsyncIterator[None]:
        set_pipeline(mock_pipeline)
        set_feedback_store(None)
        yield

    app = FastAPI(lifespan=mock_lifespan)
    app.include_router(health.router)
    app.include_router(screening.router, prefix="/screen", tags=["screening"])
    app.include_router(taxonomy.router, prefix="/taxonomy", tags=["taxonomy"])
    app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])

    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health(self, client, mock_pipeline):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["active_classifier"] == "nli"


class TestScreeningEndpoint:
    def test_screen_single(self, client, mock_pipeline):
        mock_pipeline.screen.return_value = ScreeningDecision(
            shipment_id="S1",
            action=Action.APPROVE,
            classification=ClassificationResult(
                label=ClassifierLabel.ALLOWED, confidence=0.95
            ),
            reason="AI classified as allowed",
        )

        response = client.post(
            "/screen",
            json={"id": "S1", "description": "laptop computers"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "approve"
        assert data["shipment_id"] == "S1"

    def test_screen_batch(self, client, mock_pipeline):
        mock_pipeline.screen_batch.return_value = [
            ScreeningDecision(
                shipment_id="S1",
                action=Action.APPROVE,
                classification=ClassificationResult(
                    label=ClassifierLabel.ALLOWED, confidence=0.95
                ),
                reason="allowed",
            ),
            ScreeningDecision(
                shipment_id="S2",
                action=Action.BLOCK,
                reason="restricted",
            ),
        ]

        response = client.post(
            "/screen/batch",
            json={
                "shipments": [
                    {"id": "S1", "description": "t-shirts"},
                    {"id": "S2", "description": "weapons"},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["decisions"]) == 2

    def test_screen_missing_description(self, client):
        response = client.post("/screen", json={"id": "S1"})
        assert response.status_code == 422  # Validation error


class TestFeedbackEndpoint:
    def test_feedback_without_store(self, client):
        """Feedback endpoint returns 503 when store is not configured."""
        response = client.post(
            "/feedback",
            json={
                "shipment_id": "S1",
                "reviewer_id": "R1",
                "final_decision": "approve",
            },
        )
        assert response.status_code == 503

    def test_disagreements_without_store(self, client):
        response = client.get("/feedback/disagreements")
        assert response.status_code == 503
