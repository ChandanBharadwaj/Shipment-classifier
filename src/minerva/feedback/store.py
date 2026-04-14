"""Thin psycopg wrapper for the screening feedback store."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from minerva.logging_utils import get_logger
from minerva.schema import ScreeningDecision, Shipment

logger = get_logger("feedback.store")

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_INSERT_SQL = """
    INSERT INTO screening_feedback (
        shipment_id, description, ai_decision, ai_confidence,
        taxonomy_hit, taxonomy_group, final_decision, disagreement_type,
        risk_score, risk_raw,
        entity_hit, entity_list_name, entity_match_score,
        rationale,
        model_version, classifier_type, thresholds_snapshot, input_snapshot
    ) VALUES (
        %(shipment_id)s, %(description)s, %(ai_decision)s, %(ai_confidence)s,
        %(taxonomy_hit)s, %(taxonomy_group)s, %(final_decision)s,
        %(disagreement_type)s,
        %(risk_score)s, %(risk_raw)s,
        %(entity_hit)s, %(entity_list_name)s, %(entity_match_score)s,
        %(rationale)s,
        %(model_version)s, %(classifier_type)s, %(thresholds_snapshot)s,
        %(input_snapshot)s
    )
"""


class FeedbackStore:
    """PostgreSQL-backed store for screening decisions and reviewer feedback."""

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
        """Batch insert screening decisions with full audit trail."""
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

            # Risk assessment
            risk_score = (
                decision.risk_assessment.score.value
                if decision.risk_assessment
                else None
            )
            risk_raw = (
                decision.risk_assessment.raw_score
                if decision.risk_assessment
                else None
            )

            # Entity match — record best match if any
            entity_hit = len(decision.entity_matches) > 0
            best_entity = (
                max(decision.entity_matches, key=lambda m: m.score)
                if decision.entity_matches
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

            # Input snapshot for audit trail
            input_snapshot = shipment.model_dump(exclude={"metadata"})

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
                    "risk_score": risk_score,
                    "risk_raw": risk_raw,
                    "entity_hit": entity_hit,
                    "entity_list_name": best_entity.list_name if best_entity else None,
                    "entity_match_score": best_entity.score if best_entity else None,
                    "rationale": decision.rationale or None,
                    "model_version": decision.model_version,
                    "classifier_type": decision.classifier_type,
                    "thresholds_snapshot": Jsonb(decision.thresholds_snapshot)
                        if decision.thresholds_snapshot else None,
                    "input_snapshot": Jsonb(input_snapshot),
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
        """Export reviewer-labeled data for model training."""
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

    # --- Drift baseline snapshots ---

    def save_drift_baseline(
        self,
        name: str,
        confidences: list[float],
        labels: list[str],
        hit_rates: dict[str, float],
        centroid: list[float] | None = None,
    ) -> None:
        """Persist a baseline distribution for later drift comparisons."""
        sql = """
            INSERT INTO drift_baselines (name, sample_size, confidences, labels, hit_rates, centroid)
            VALUES (%(name)s, %(sample_size)s, %(confidences)s, %(labels)s, %(hit_rates)s, %(centroid)s)
            ON CONFLICT (name) DO UPDATE SET
                sample_size = EXCLUDED.sample_size,
                confidences = EXCLUDED.confidences,
                labels = EXCLUDED.labels,
                hit_rates = EXCLUDED.hit_rates,
                centroid = EXCLUDED.centroid,
                created_at = NOW()
        """
        with self._pool.connection() as conn:
            conn.execute(
                sql,
                {
                    "name": name,
                    "sample_size": len(confidences),
                    "confidences": Jsonb(list(confidences)),
                    "labels": Jsonb(list(labels)),
                    "hit_rates": Jsonb(hit_rates),
                    "centroid": Jsonb(centroid) if centroid else None,
                },
            )
            conn.commit()
