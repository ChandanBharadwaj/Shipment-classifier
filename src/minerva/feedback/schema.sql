-- Minerva Hybrid Screening Framework — Feedback Store Schema

CREATE TABLE IF NOT EXISTS screening_feedback (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    shipment_id       VARCHAR(255) NOT NULL,
    description       TEXT NOT NULL,
    keyword_decision  VARCHAR(20),
    ai_decision       VARCHAR(20),
    ai_confidence     NUMERIC(4,3),
    taxonomy_hit      BOOLEAN DEFAULT FALSE,
    taxonomy_group    VARCHAR(100),
    final_decision    VARCHAR(20),
    reviewer_id       VARCHAR(100),
    reviewer_notes    TEXT,
    disagreement_type VARCHAR(50),
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_screening_feedback_shipment
    ON screening_feedback(shipment_id);

CREATE INDEX IF NOT EXISTS idx_screening_feedback_action
    ON screening_feedback(final_decision);

CREATE INDEX IF NOT EXISTS idx_screening_feedback_disagreement
    ON screening_feedback(disagreement_type, created_at);

CREATE INDEX IF NOT EXISTS idx_screening_feedback_created
    ON screening_feedback(created_at);

CREATE INDEX IF NOT EXISTS idx_screening_feedback_reviewer
    ON screening_feedback(reviewer_id)
    WHERE reviewer_id IS NOT NULL;
