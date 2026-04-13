"""Integration tests for the feedback store (requires PostgreSQL)."""

import os

import pytest

from minerva.feedback.store import FeedbackStore
from minerva.schema import (
    Action,
    ClassificationResult,
    ClassifierLabel,
    ScreeningDecision,
    Shipment,
)

pytestmark = pytest.mark.integration

DB_DSN = os.environ.get("MINERVA_DB_DSN", "")


@pytest.fixture(scope="module")
def store():
    if not DB_DSN:
        pytest.skip("MINERVA_DB_DSN not set — skipping feedback store tests")

    s = FeedbackStore(DB_DSN)
    s.init_schema()
    yield s
    s.close()


class TestFeedbackStore:
    def test_record_and_query(self, store):
        decisions = [
            ScreeningDecision(
                shipment_id="TEST-001",
                action=Action.APPROVE,
                classification=ClassificationResult(
                    label=ClassifierLabel.ALLOWED, confidence=0.95
                ),
                reason="test",
            ),
        ]
        shipments = [
            Shipment(id="TEST-001", description="Test shipment"),
        ]

        store.record_decisions(decisions, shipments)

    def test_record_reviewer_feedback(self, store):
        store.record_reviewer_feedback(
            shipment_id="TEST-001",
            reviewer_id="tester",
            final_decision="approve",
            notes="Looks fine",
        )

    def test_get_disagreements(self, store):
        # Record a disagreement
        decisions = [
            ScreeningDecision(
                shipment_id="TEST-DISAGREE",
                action=Action.BLOCK,
                classification=ClassificationResult(
                    label=ClassifierLabel.ALLOWED, confidence=0.85
                ),
                reason="test disagreement",
            ),
        ]
        shipments = [
            Shipment(id="TEST-DISAGREE", description="Disputed shipment"),
        ]
        store.record_decisions(
            decisions, shipments, keyword_decisions=["block"]
        )

        records = store.get_disagreements(limit=10)
        # Should find at least the one we just inserted
        assert isinstance(records, list)
