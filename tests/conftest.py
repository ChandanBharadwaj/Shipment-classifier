"""Shared test fixtures for Minerva tests."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from minerva.config import MinervaSettings
from minerva.schema import (
    ClassificationResult,
    ClassifierLabel,
    RiskLevel,
    Shipment,
    TaxonomyGroup,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def sample_taxonomy_path() -> Path:
    return FIXTURES_DIR / "sample_taxonomy.json"


@pytest.fixture
def sample_taxonomy() -> list[TaxonomyGroup]:
    """Three sample taxonomy groups covering different risk levels."""
    return [
        TaxonomyGroup(
            group_id="hazardous_materials",
            name="Hazardous Materials",
            risk_level=RiskLevel.CRITICAL,
            semantic_phrases=["explosive materials", "flammable liquids", "radioactive material"],
            hard_keywords=[],
            allow_auto_clear=False,
        ),
        TaxonomyGroup(
            group_id="military_and_dual_use",
            name="Military and Dual-Use Equipment",
            risk_level=RiskLevel.CRITICAL,
            semantic_phrases=["military equipment", "missile systems", "military firearms"],
            hard_keywords=["ak47"],
            allow_auto_clear=False,
        ),
        TaxonomyGroup(
            group_id="restricted_wildlife",
            name="Restricted Biology and Wildlife",
            risk_level=RiskLevel.HIGH,
            semantic_phrases=["protected wildlife", "live animal"],
            hard_keywords=["ivory"],
            allow_auto_clear=False,
        ),
    ]


@pytest.fixture
def sample_shipments() -> list[Shipment]:
    """Sample shipments for testing."""
    return [
        Shipment(id="SHIP-001", description="12 plastic toy water pistols for kids summer camp"),
        Shipment(id="SHIP-002", description="Industrial hydraulic fluid drums"),
        Shipment(id="SHIP-003", description="Military grade weapon systems for defense contractor"),
        Shipment(id="SHIP-004", description="Organic cotton t-shirts bulk shipment"),
        Shipment(id="SHIP-005", description="AK47 rifle parts and ammunition"),
        Shipment(id="SHIP-006", description="Fresh fruit and vegetables for supermarket"),
        Shipment(id="SHIP-007", description="Ivory carved decorative pieces"),
    ]


@pytest.fixture
def settings() -> MinervaSettings:
    """Test settings with defaults."""
    return MinervaSettings(
        config_dir=Path("./config"),
        log_level="DEBUG",
    )


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer that returns deterministic 384-dim embeddings."""
    mock = MagicMock()
    embedding_dim = 384

    def fake_encode(texts, **kwargs):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(len(texts), embedding_dim).astype(np.float32)
        # Normalize to unit vectors for meaningful cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    mock.encode = MagicMock(side_effect=fake_encode)
    return mock


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder with controllable prediction outputs."""
    mock = MagicMock()

    def fake_predict(pairs, **kwargs):
        # Return 3 NLI scores (contradiction, neutral, entailment) per pair
        n_pairs = len(pairs)
        rng = np.random.RandomState(42)
        return rng.randn(n_pairs, 3).astype(np.float32)

    mock.predict = MagicMock(side_effect=fake_predict)
    return mock
