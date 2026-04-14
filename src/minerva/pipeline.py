"""Screening pipeline orchestrating Layer 1 → Layer 2 → Layer 3.

Enhanced with:
- Entity resolution / denied-party screening
- Multi-feature risk scoring
- Rationale generation
- Audit trail metadata

All enhancement layers are optional and toggleable via settings.
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
from minerva.risk.scorer import RiskConfig, RiskScorer
from minerva.router.decision import route_batch
from minerva.schema import Action, ScreeningDecision, Shipment
from minerva.taxonomy.embeddings import TaxonomyEmbeddingIndex
from minerva.taxonomy.loader import load_taxonomy
from minerva.taxonomy.matcher import TaxonomyMatcher

logger = get_logger("pipeline")


class ScreeningPipeline:
    """Orchestrates the full hybrid screening pipeline.

    Layers:
        1. Taxonomy (deterministic, always enabled)
        2. AI Classifier (NLI or DistilBERT, always enabled)
        3. Entity Resolution (optional, via config)
        4. Risk Scoring (optional, via config)
        5. Routing (always enabled)
        6. Rationale Generation (always enabled - audit requirement)
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

        # Risk scorer (optional)
        self._risk_scorer: RiskScorer | None = None
        if settings.risk_scoring.enabled:
            risk_config = RiskConfig.from_json(settings.risk_config_path)
            self._risk_scorer = RiskScorer(config=risk_config)
            logger.info("Risk scorer loaded")

        # Rationale generator (always on — audit requirement)
        self._rationale_generator = TemplatedRationaleGenerator()

        logger.info("Screening pipeline ready")

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

    def screen_batch(
        self, shipments: list[Shipment]
    ) -> list[ScreeningDecision]:
        """Screen a batch of shipments through all enabled layers."""
        if not shipments:
            return []

        all_decisions: list[ScreeningDecision] = []
        batch_size = self._settings.batch_size

        for start in range(0, len(shipments), batch_size):
            chunk = shipments[start : start + batch_size]
            chunk_decisions = self._screen_chunk(chunk)
            all_decisions.extend(chunk_decisions)

        return all_decisions

    def _screen_chunk(
        self, shipments: list[Shipment]
    ) -> list[ScreeningDecision]:
        """Process a single chunk through all enabled layers."""
        descriptions = [s.description for s in shipments]

        # Layer 1: Taxonomy
        taxonomy_results = self._matcher.match_batch(descriptions)

        # Layer 2: AI classification
        classification_results = self._classifier.classify_batch(descriptions)

        # Entity resolution (optional)
        entity_results: list[list] | None = None
        if self._entity_resolver is not None:
            entity_results = self._entity_resolver.resolve_batch(shipments)

        # Risk scoring (optional)
        risk_results: list | None = None
        if self._risk_scorer is not None:
            risk_results = self._risk_scorer.score_batch(shipments)

        # Layer 3: Routing
        decisions = route_batch(
            shipments=shipments,
            taxonomy_results=taxonomy_results,
            classification_results=classification_results,
            routing_config=self._settings.routing,
            entity_results=entity_results,
            risk_results=risk_results,
        )

        # Attach audit trail metadata
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
        """Attach model version, classifier type, and active thresholds to each decision."""
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
        """Log disagreements between taxonomy and AI classifier."""
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

    def screen(self, shipment: Shipment) -> ScreeningDecision:
        """Screen a single shipment (convenience wrapper)."""
        return self.screen_batch([shipment])[0]
