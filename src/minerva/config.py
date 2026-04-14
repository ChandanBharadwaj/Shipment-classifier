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


class EntityResolutionConfig(BaseSettings):
    enabled: bool = True
    auto_block_threshold: float = 0.92
    review_threshold: float = 0.80


class RiskScoringConfig(BaseSettings):
    enabled: bool = True


class DriftMonitoringConfig(BaseSettings):
    enabled: bool = True
    psi_warning: float = 0.10
    psi_alert: float = 0.25


class AIEngineConfig(BaseSettings):
    """AI risk engine — HS prediction, semantic dual-use, coherence."""

    enabled: bool = True
    hs_top_k: int = 3
    dual_use_similarity_threshold: float = 0.55
    coherence_low_threshold: float = 0.35


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
    entity_resolution: EntityResolutionConfig = Field(
        default_factory=EntityResolutionConfig
    )
    risk_scoring: RiskScoringConfig = Field(default_factory=RiskScoringConfig)
    drift_monitoring: DriftMonitoringConfig = Field(
        default_factory=DriftMonitoringConfig
    )
    ai_engine: AIEngineConfig = Field(default_factory=AIEngineConfig)
    nli_hypotheses: NLIHypotheses = Field(default_factory=NLIHypotheses)

    @property
    def taxonomy_path(self) -> Path:
        return self.config_dir / "taxonomy.json"

    @property
    def denied_parties_path(self) -> Path:
        return self.config_dir / "denied_parties.json"

    @property
    def risk_config_path(self) -> Path:
        return self.config_dir / "risk_config.json"

    @property
    def country_risk_path(self) -> Path:
        """Country risk intelligence (LexisNexis-style). JSON or CSV."""
        json_path = self.config_dir / "country_risk.json"
        csv_path = self.config_dir / "country_risk.csv"
        # Prefer CSV if the user has dropped a vendor export there
        return csv_path if csv_path.exists() else json_path

    def get_similarity_threshold(self, risk_level: str) -> float:
        return getattr(self.taxonomy_thresholds, risk_level.lower())
