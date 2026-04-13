"""Thin psycopg wrapper for the screening feedback store."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psycopg_pool import ConnectionPool

from minerva.logging_utils import get_logger
from minerva.schema import ScreeningDecision, Shipment

logger = get_logger("feedback.store")

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_INSERT_SQL = """
    INSERT INTO screening_feedback (
        shipment_id, description, ai_decision, ai_confidence,
        taxonomy_hit, taxonomy_group, final_decision, disagreement_type
    ) VALUES (
        %(shipment_id)s, %(description)s, %(ai_decision)s, %(ai_confidence)s,
        %(taxonomy_hit)s, %(taxonomy_group)s, %(final_decision)s,
        %(disagreement_type)s
    )
"""


class FeedbackStore:
    """PostgreSQL-backed store for screening decisions and reviewer feedback.

    Uses psycopg connection pooling for efficient batch operations.
    """

    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10) -> None:
        self._pool = ConnectionPool(
            dsn, min_size=min_size, max_size=max_size
        )
        logger.info("Feedback store connected")

    def close(self) -> None:
        self._pool.close()

    def init_schema(self) -> None:
        """Create tables if they don't exist."""
        sql = _SCHEMA_PATH.read_text()
        with self._pool.connection() as conn:
            conn.execute(sql)
            conn.commit()
        logger.info("Feedback store schema initialized")

    def record_decisions(
        self,
        decisions: list[ScreeningDecision],
        shipments: list[Shipment],
        keyword_decisions: list[str] | None = None,
    ) -> None:
        """Batch insert screening decisions.

        Args:
            decisions: AI screening decisions.
            shipments: Corresponding shipments.
            keyword_decisions: Optional keyword engine decisions for disagreement tracking.
        """
        rows: list[dict[str, Any]] = []
        for i, (decision, shipment) in enumerate(zip(decisions, shipments, strict=True)):
            has_taxonomy_hit = len(decision.taxonomy_hits) > 0
            taxonomy_group = (
                decision.taxonomy_hits[0].group_id
                if has_taxonomy_hit
                else None
            )

            ai_decision = (
                decision.classification.label.value
                if decision.classification
                else None
            )
            ai_confidence = (
                decision.classification.confidence
                if decision.classification
                else None
            )

            # Determine disagreement type
            disagreement_type = None
            if keyword_decisions and i < len(keyword_decisions):
                kw = keyword_decisions[i].lower()
                ai = ai_decision
                if kw == "block" and ai == "allowed":
                    disagreement_type = "KW_BLOCK_AI_ALLOW"
                elif kw == "allow" and ai == "restricted":
                    disagreement_type = "KW_ALLOW_AI_BLOCK"

            rows.append(
                {
                    "shipment_id": shipment.id,
                    "description": shipment.description,
                    "ai_decision": ai_decision,
                    "ai_confidence": ai_confidence,
                    "taxonomy_hit": has_taxonomy_hit,
                    "taxonomy_group": taxonomy_group,
                    "final_decision": decision.action.value,
                    "disagreement_type": disagreement_type,
                }
            )

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(_INSERT_SQL, rows)
            conn.commit()

        logger.info("Recorded %d screening decisions", len(rows))

    def record_reviewer_feedback(
        self,
        shipment_id: str,
        reviewer_id: str,
        final_decision: str,
        notes: str | None = None,
    ) -> None:
        """Record a reviewer's authoritative decision on a shipment."""
        sql = """
            UPDATE screening_feedback
            SET final_decision = %(final_decision)s,
                reviewer_id = %(reviewer_id)s,
                reviewer_notes = %(notes)s,
                reviewed_at = %(reviewed_at)s
            WHERE shipment_id = %(shipment_id)s
              AND reviewer_id IS NULL
            ORDER BY created_at DESC
            LIMIT 1
        """
        # Use a simpler update that works with PostgreSQL
        sql = """
            UPDATE screening_feedback
            SET final_decision = %(final_decision)s,
                reviewer_id = %(reviewer_id)s,
                reviewer_notes = %(notes)s,
                reviewed_at = %(reviewed_at)s
            WHERE id = (
                SELECT id FROM screening_feedback
                WHERE shipment_id = %(shipment_id)s
                  AND reviewer_id IS NULL
                ORDER BY created_at DESC
                LIMIT 1
            )
        """
        with self._pool.connection() as conn:
            conn.execute(
                sql,
                {
                    "shipment_id": shipment_id,
                    "reviewer_id": reviewer_id,
                    "final_decision": final_decision,
                    "notes": notes,
                    "reviewed_at": datetime.now(timezone.utc),
                },
            )
            conn.commit()

    def get_disagreements(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Query screening records where AI and keyword decisions disagree."""
        sql = """
            SELECT id, shipment_id, description, keyword_decision,
                   ai_decision, ai_confidence, taxonomy_hit, taxonomy_group,
                   final_decision, disagreement_type, created_at
            FROM screening_feedback
            WHERE disagreement_type IS NOT NULL
            ORDER BY created_at DESC
            LIMIT %(limit)s OFFSET %(offset)s
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"limit": limit, "offset": offset})
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    def export_labeled_data(
        self, min_confidence: float | None = None
    ) -> list[dict[str, Any]]:
        """Export reviewer-labeled data for model training.

        Returns records where a reviewer has provided a final decision.
        """
        sql = """
            SELECT shipment_id, description, ai_decision, ai_confidence,
                   final_decision, reviewer_id, taxonomy_hit
            FROM screening_feedback
            WHERE reviewer_id IS NOT NULL
              AND reviewed_at IS NOT NULL
        """
        params: dict[str, Any] = {}
        if min_confidence is not None:
            sql += " AND ai_confidence >= %(min_confidence)s"
            params["min_confidence"] = min_confidence

        sql += " ORDER BY reviewed_at DESC"

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()

        return [dict(zip(columns, row)) for row in rows]
