"""Screening pipeline orchestrating Layer 1 → Layer 2 → Layer 3 → Risk Profile.

The pipeline produces:
- Taxonomy matches (Layer 1)
- AI classification (Layer 2)
- Entity resolution matches (optional)
- Legacy single-tier risk assessment (optional)
- Multi-dimensional RiskProfile — the primary officer-facing output
- Routing action (advisory — officer decides)
- Rationale + full audit trail

The officer owns the decision. The system surfaces risk across dimensions.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from minerva import __version__
from minerva.classifier.base import BaseClassifier
from minerva.classifier.distilbert import DistilBERTClassifier
from minerva.classifier.nli import NLIClassifier
from minerva.config import MinervaSettings
from minerva.entity_resolution.denied_party_list import load_denied_parties
from minerva.entity_resolution.resolver import EntityResolver, EntityResolverConfig
from minerva.explain.rationale import TemplatedRationaleGenerator, attach_rationale
from minerva.logging_utils import get_logger, log_disagreement
from minerva.risk.dimensions import (
    DimensionContext,
    DualUseConfig,
    GeographyConfig,
    HsCodeConfig,
    ValuationConfig,
)
from minerva.risk.profile import (
    ProfileBuilderConfig,
    RiskProfileBuilder,
    default_assessors,
)
from minerva.risk.scorer import RiskConfig, RiskScorer
from minerva.router.decision import route_batch
from minerva.schema import Action, ScreeningDecision, Shipment
from minerva.taxonomy.embeddings import TaxonomyEmbeddingIndex
from minerva.taxonomy.loader import load_taxonomy
from minerva.taxonomy.matcher import TaxonomyMatcher

logger = get_logger("pipeline")


class ScreeningPipeline:
    """Orchestrates the hybrid screening pipeline + multi-dimensional risk profile.

    Layers:
        1. Taxonomy (deterministic, always enabled)
        2. AI Classifier (NLI or DistilBERT, always enabled)
        3. Entity Resolution (optional, via config)
        4. Legacy Risk Scoring (optional, via config — single tier view)
        5. Routing (advisory action)
        6. Risk Profile Builder (always enabled — 8-dimension view)
        7. Rationale Generation (always enabled - audit requirement)
    """

    def __init__(self, settings: MinervaSettings) -> None:
        self._settings = settings

        logger.info("Initializing screening pipeline...")

        # Load taxonomy
        groups = load_taxonomy(settings.taxonomy_path)
        logger.info("Loaded %d taxonomy groups", len(groups))

        # Embedding model (shared between taxonomy and matcher)
        logger.info("Loading embedding model: %s", settings.embedding_model)
        self._embedding_model = SentenceTransformer(settings.embedding_model)

        # Taxonomy index and matcher
        self._taxonomy_index = TaxonomyEmbeddingIndex(
            groups=groups, model=self._embedding_model
        )
        logger.info(
            "Taxonomy index built: %d phrases across %d groups",
            self._taxonomy_index.num_phrases,
            self._taxonomy_index.num_groups,
        )

        self._matcher = TaxonomyMatcher(
            index=self._taxonomy_index,
            model=self._embedding_model,
            settings=settings,
        )

        # AI classifier
        self._classifier: BaseClassifier
        if settings.active_classifier == "distilbert":
            logger.info("Loading DistilBERT classifier from: %s", settings.distilbert_model_path)
            self._classifier = DistilBERTClassifier(settings)
        else:
            logger.info("Loading NLI classifier: %s", settings.nli_model)
            self._classifier = NLIClassifier(settings)

        # Entity resolution (optional)
        self._entity_resolver: EntityResolver | None = None
        if settings.entity_resolution.enabled:
            parties = load_denied_parties(settings.denied_parties_path)
            if parties:
                entity_config = EntityResolverConfig(
                    auto_block_threshold=settings.entity_resolution.auto_block_threshold,
                    review_threshold=settings.entity_resolution.review_threshold,
                )
                self._entity_resolver = EntityResolver(
                    parties=parties, config=entity_config
                )
                logger.info(
                    "Entity resolver loaded with %d denied parties",
                    len(parties),
                )
            else:
                logger.info("No denied parties configured — entity resolution disabled")

        # Legacy single-tier risk scorer (optional)
        self._risk_scorer: RiskScorer | None = None
        risk_config: RiskConfig | None = None
        if settings.risk_scoring.enabled:
            risk_config = RiskConfig.from_json(settings.risk_config_path)
            self._risk_scorer = RiskScorer(config=risk_config)
            logger.info("Legacy risk scorer loaded")

        # Multi-dimensional risk profile builder (always enabled)
        self._profile_builder = self._build_profile_builder(
            risk_config, settings
        )
        logger.info("Risk profile builder ready (8 dimensions)")

        # Rationale generator (always on — audit requirement)
        self._rationale_generator = TemplatedRationaleGenerator()

        logger.info("Screening pipeline ready")

    @staticmethod
    def _build_profile_builder(
        risk_config: RiskConfig | None,
        settings: MinervaSettings,
    ) -> RiskProfileBuilder:
        """Wire up the 8 dimension assessors using shared config sources."""
        geography_cfg = GeographyConfig(
            country_tiers=(risk_config.country_tiers if risk_config else {}) or {},
        )
        valuation_cfg = ValuationConfig(
            high_value_threshold=(
                risk_config.high_value_threshold if risk_config else 100_000.0
            ),
        )
        hs_code_cfg = HsCodeConfig(
            sensitive_prefixes=(
                risk_config.sensitive_hs_prefixes if risk_config else []
            ),
        )
        dual_use_cfg = DualUseConfig()

        assessors = default_assessors(
            geography_config=geography_cfg,
            valuation_config=valuation_cfg,
            hs_code_config=hs_code_cfg,
            dual_use_config=dual_use_cfg,
        )
        profile_cfg = ProfileBuilderConfig.from_json(
            settings.config_dir / "profile_weights.json"
        )
        return RiskProfileBuilder(assessors=assessors, config=profile_cfg)

    # ---------------- accessors ----------------

    @property
    def matcher(self) -> TaxonomyMatcher:
        return self._matcher

    @property
    def classifier(self) -> BaseClassifier:
        return self._classifier

    @property
    def entity_resolver(self) -> EntityResolver | None:
        return self._entity_resolver

    @property
    def risk_scorer(self) -> RiskScorer | None:
        return self._risk_scorer

    @property
    def profile_builder(self) -> RiskProfileBuilder:
        return self._profile_builder

    # ---------------- public API ----------------

    def screen_batch(
        self, shipments: list[Shipment]
    ) -> list[ScreeningDecision]:
        if not shipments:
            return []

        all_decisions: list[ScreeningDecision] = []
        batch_size = self._settings.batch_size
        for start in range(0, len(shipments), batch_size):
            chunk = shipments[start : start + batch_size]
            chunk_decisions = self._screen_chunk(chunk)
            all_decisions.extend(chunk_decisions)
        return all_decisions

    def screen(self, shipment: Shipment) -> ScreeningDecision:
        return self.screen_batch([shipment])[0]

    def build_profile(self, shipment: Shipment):
        """Build a risk profile for a single shipment without routing.

        Useful when the consumer wants only the dimensional view.
        """
        descriptions = [shipment.description]
        tax_hits = self._matcher.match_batch(descriptions)[0]
        cls_result = self._classifier.classify_batch(descriptions)[0]
        entity_matches = (
            self._entity_resolver.resolve_shipment(shipment)
            if self._entity_resolver else []
        )
        ctx = DimensionContext(
            shipment=shipment,
            taxonomy_hits=tax_hits,
            classification=cls_result,
            entity_matches=entity_matches,
        )
        return self._profile_builder.build(ctx)

    # ---------------- internals ----------------

    def _screen_chunk(
        self, shipments: list[Shipment]
    ) -> list[ScreeningDecision]:
        descriptions = [s.description for s in shipments]

        # Layer 1: Taxonomy
        taxonomy_results = self._matcher.match_batch(descriptions)

        # Layer 2: AI classification
        classification_results = self._classifier.classify_batch(descriptions)

        # Entity resolution (optional)
        entity_results: list[list] | None = None
        if self._entity_resolver is not None:
            entity_results = self._entity_resolver.resolve_batch(shipments)

        # Legacy single-tier risk scoring (optional)
        risk_results: list | None = None
        if self._risk_scorer is not None:
            risk_results = self._risk_scorer.score_batch(shipments)

        # Layer 3: Routing (advisory action)
        decisions = route_batch(
            shipments=shipments,
            taxonomy_results=taxonomy_results,
            classification_results=classification_results,
            routing_config=self._settings.routing,
            entity_results=entity_results,
            risk_results=risk_results,
        )

        # Build multi-dimensional profiles for every shipment
        contexts = [
            DimensionContext(
                shipment=s,
                taxonomy_hits=t,
                classification=c,
                entity_matches=(e if entity_results else []),
            )
            for s, t, c, e in zip(
                shipments,
                taxonomy_results,
                classification_results,
                entity_results if entity_results else [[] for _ in shipments],
                strict=True,
            )
        ]
        profiles = self._profile_builder.build_batch(contexts)

        # Attach profiles and audit metadata
        decisions = [
            d.model_copy(update={"risk_profile": p})
            for d, p in zip(decisions, profiles, strict=True)
        ]
        decisions = self._attach_audit_metadata(decisions)

        # Generate rationales
        decisions = attach_rationale(
            self._rationale_generator, shipments, decisions
        )

        # Log disagreements
        self._log_disagreements(shipments, taxonomy_results, decisions)

        return decisions

    def _attach_audit_metadata(
        self, decisions: list[ScreeningDecision]
    ) -> list[ScreeningDecision]:
        thresholds = {
            "taxonomy": {
                "critical": self._settings.taxonomy_thresholds.critical,
                "high": self._settings.taxonomy_thresholds.high,
                "medium": self._settings.taxonomy_thresholds.medium,
            },
            "routing": {
                "auto_approve_min_confidence": self._settings.routing.auto_approve_min_confidence,
                "auto_block_min_confidence": self._settings.routing.auto_block_min_confidence,
            },
            "entity_resolution": {
                "enabled": self._settings.entity_resolution.enabled,
                "auto_block_threshold": self._settings.entity_resolution.auto_block_threshold,
                "review_threshold": self._settings.entity_resolution.review_threshold,
            },
        }
        model_version = (
            f"minerva-{__version__};classifier={self._settings.active_classifier}"
        )
        return [
            d.model_copy(
                update={
                    "model_version": model_version,
                    "classifier_type": self._settings.active_classifier,
                    "thresholds_snapshot": thresholds,
                }
            )
            for d in decisions
        ]

    def _log_disagreements(
        self,
        shipments: list[Shipment],
        taxonomy_results: list[list],
        decisions: list[ScreeningDecision],
    ) -> None:
        for shipment, tax_hits, decision in zip(
            shipments, taxonomy_results, decisions, strict=True
        ):
            has_taxonomy_hit = len(tax_hits) > 0
            ai_result = decision.classification
            if ai_result is None:
                continue
            if has_taxonomy_hit and ai_result.label.value == "allowed":
                log_disagreement(
                    logger=logger,
                    shipment_id=shipment.id,
                    taxonomy_action="block",
                    ai_action=ai_result.label.value,
                    ai_confidence=ai_result.confidence,
                    description=shipment.description,
                )
            if not has_taxonomy_hit and ai_result.label.value == "restricted":
                if decision.action == Action.APPROVE:
                    log_disagreement(
                        logger=logger,
                        shipment_id=shipment.id,
                        taxonomy_action="allow",
                        ai_action=ai_result.label.value,
                        ai_confidence=ai_result.confidence,
                        description=shipment.description,
                    )
