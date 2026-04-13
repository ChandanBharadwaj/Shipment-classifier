"""Configuration management for Minerva using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class TaxonomySimilarityThresholds(BaseSettings):
    critical: float = 0.75
    high: float = 0.78
    medium: float = 0.80


class RoutingConfig(BaseSettings):
    auto_approve_min_confidence: float = 0.90
    auto_block_min_confidence: float = 0.90


class NLIHypotheses(BaseSettings):
    allowed: str = "This shipment is allowed for export without restrictions."
    restricted: str = "This shipment contains restricted or controlled items."
    needs_review: str = "This shipment requires further review by a compliance officer."


class MinervaSettings(BaseSettings):
    model_config = {"env_prefix": "MINERVA_"}

    config_dir: Path = Path("./config")
    db_dsn: str | None = None
    log_level: str = "INFO"

    embedding_model: str = "intfloat/e5-small-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    active_classifier: Literal["nli", "distilbert"] = "nli"
    distilbert_model_path: str = "./models/distilbert-finetuned"

    batch_size: int = Field(default=256, ge=1)

    taxonomy_thresholds: TaxonomySimilarityThresholds = Field(
        default_factory=TaxonomySimilarityThresholds
    )
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    nli_hypotheses: NLIHypotheses = Field(default_factory=NLIHypotheses)

    @property
    def taxonomy_path(self) -> Path:
        return self.config_dir / "taxonomy.json"

    def get_similarity_threshold(self, risk_level: str) -> float:
        return getattr(self.taxonomy_thresholds, risk_level.lower())
