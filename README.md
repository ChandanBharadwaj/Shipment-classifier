# Minerva Hybrid Screening Framework

Trade compliance shipment screening system combining deterministic taxonomy enforcement with AI-based semantic classification.

## Architecture

Six-stage hybrid screening pipeline:

1. **Layer 1 — Hard Taxonomy**: Deterministic enforcement via semantic similarity matching against configurable taxonomy groups. Taxonomy hits are non-negotiable — always blocked.
2. **Layer 2 — AI Classifier**: Zero-shot NLI (Phase 1) or fine-tuned DistilBERT (Phase 2+) producing 3-class output: Allowed / Restricted / Needs Review.
3. **Entity Resolution**: Cascade fuzzy-matches shipment consignee/shipper against a denied-party list (exact → auto-block, fuzzy → review).
4. **Multi-Feature Risk Scoring**: Combines origin/destination country tiers, declared value, HS code, and data quality into a 1–5 risk tier.
5. **Layer 3 — Confidence + Risk Router**: Routes decisions based on confidence thresholds and risk tier. Critical risk escalates to review even on high-confidence allowed.
6. **Rationale Generation + Audit Trail**: Every decision produces a human-readable rationale and a full audit record (model version, thresholds snapshot, input snapshot) — required for EU AI Act enforcement from August 2026.

### Supporting subsystems

- **Active Learning Sampler** — selects the *most informative* uncertain items for reviewer labeling (uncertainty + diversity via gATE-style core-set selection), for 5–10× labeling efficiency.
- **Drift Monitoring** — PSI on confidence distributions, label distributions, and per-taxonomy-group hit rates. Baselines stored for Phase 3 automation safety.

```
Shipment Description
        │
        ▼
┌───────────────────────────────┐
│  Layer 1: Hard Taxonomy       │  Cosine similarity against
│  (Deterministic Enforcement)  │  pre-computed phrase embeddings
└──────────────┬────────────────┘
               │ No hit
               ▼
┌───────────────────────────────────┐
│  Layer 2: AI Semantic Classifier  │  NLI cross-encoder or
│  3-class: Allow / Block / Review  │  fine-tuned DistilBERT
└──────────────┬────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Layer 3: Confidence-Based Router       │
│  High conf Allow    → Auto Approve      │
│  High conf Restrict → Auto Block        │
│  Uncertain          → Manual Review     │
│  Taxonomy hit       → Block (always)    │
└─────────────────────────────────────────┘
```

## Setup

```bash
# Install the package
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your settings
```

## Usage

### API Server

```bash
uvicorn minerva.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check with system info |
| `POST` | `/screen` | Screen a single shipment |
| `POST` | `/screen/batch` | Screen a batch of shipments |
| `GET` | `/taxonomy` | Get current taxonomy |
| `PUT` | `/taxonomy` | Update taxonomy (re-embeds) |
| `POST` | `/feedback` | Submit reviewer feedback |
| `GET` | `/feedback/disagreements` | List AI/keyword disagreements |
| `POST` | `/risk/score` | Compute risk assessment for a shipment |
| `POST` | `/entity/resolve` | Resolve shipment parties against denied-party list |
| `POST` | `/monitoring/drift` | Compare baseline vs current distribution (PSI, hit rates) |
| `POST` | `/active-learning/select` | Select most informative items for reviewer labeling |

### Single Shipment Screening

```bash
curl -X POST http://localhost:8000/screen \
  -H "Content-Type: application/json" \
  -d '{"id": "S001", "description": "12 plastic toy water pistols for kids summer camp"}'
```

### Batch Screening (CLI)

```bash
python scripts/run_batch.py \
  --input tests/fixtures/sample_shipments.json \
  --output /tmp/decisions.json
```

### Database Setup

```bash
python scripts/init_db.py --dsn postgresql://user:pass@localhost:5432/minerva
```

## Configuration

Configuration is loaded from `config/` directory:

- `taxonomy.json` — Taxonomy groups with risk levels, semantic phrases, and hard keywords
- `thresholds.json` — Similarity thresholds per risk level and routing confidence thresholds
- `models.json` — Model identifiers and NLI hypotheses

Environment variables (prefix `MINERVA_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MINERVA_DB_DSN` | None | PostgreSQL connection string |
| `MINERVA_CONFIG_DIR` | `./config` | Config directory path |
| `MINERVA_LOG_LEVEL` | `INFO` | Logging level |
| `MINERVA_ACTIVE_CLASSIFIER` | `nli` | `nli` or `distilbert` |
| `MINERVA_EMBEDDING_MODEL` | `intfloat/e5-small-v2` | Sentence embedding model |
| `MINERVA_NLI_MODEL` | `cross-encoder/nli-deberta-v3-small` | NLI cross-encoder model |

## Training Pipeline (Phase 2)

```bash
# Export labeled data from feedback store
python -c "
from minerva.feedback.store import FeedbackStore
from minerva.training.exporter import export_training_data
from pathlib import Path

store = FeedbackStore('postgresql://...')
export_training_data(store, Path('./data/training'))
"

# Train DistilBERT classifier
python -c "
from minerva.training.trainer import train_model, TrainingConfig
from pathlib import Path

train_model(Path('./data/training'), TrainingConfig(num_epochs=5))
"

# Evaluate model against quality gates
python -c "
from minerva.training.evaluator import evaluate_model
from pathlib import Path

result = evaluate_model(Path('./models/distilbert-finetuned'), Path('./data/training/test.jsonl'))
print('Passed:', result.passed)
print('Metrics:', result.metrics['classification_report'])
"
```

## Testing

```bash
# Run unit tests (no models required)
pytest tests/unit/ -v

# Run integration tests (requires models to download)
pytest tests/integration/ -v -m integration

# Run all tests
pytest -v -m ""
```

## Routing Rules

| Taxonomy Hit | AI Class | Confidence | Final Decision |
|-------------|----------|------------|----------------|
| Yes | Any | Any | **Block** |
| No | Allowed | ≥ 0.90 | Auto Approve |
| No | Restricted | ≥ 0.90 | Auto Block |
| No | Any | < 0.90 | Manual Review |

Thresholds are configurable per operator in `config/thresholds.json`.
