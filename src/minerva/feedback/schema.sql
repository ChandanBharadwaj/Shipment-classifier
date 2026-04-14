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

    -- Risk scoring
    risk_score        INTEGER,
    risk_raw          NUMERIC(5,4),

    -- Entity resolution
    entity_hit        BOOLEAN DEFAULT FALSE,
    entity_list_name  VARCHAR(100),
    entity_match_score NUMERIC(4,3),

    -- Rationale
    rationale         TEXT,

    -- Audit trail (required for EU AI Act enforcement from Aug 2026)
    model_version     VARCHAR(100),
    classifier_type   VARCHAR(50),
    thresholds_snapshot JSONB,
    input_snapshot    JSONB,

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

CREATE INDEX IF NOT EXISTS idx_screening_feedback_risk
    ON screening_feedback(risk_score)
    WHERE risk_score IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_screening_feedback_entity_hit
    ON screening_feedback(entity_hit)
    WHERE entity_hit = TRUE;


-- Drift baseline snapshots for monitoring (Phase 3 automation prerequisite)
CREATE TABLE IF NOT EXISTS drift_baselines (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(100) NOT NULL UNIQUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    sample_size     INTEGER NOT NULL,
    confidences     JSONB NOT NULL,
    labels          JSONB NOT NULL,
    hit_rates       JSONB NOT NULL,
    centroid        JSONB
);
