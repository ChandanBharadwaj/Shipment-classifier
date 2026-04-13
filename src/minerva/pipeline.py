"""Screening pipeline orchestrating Layer 1 → Layer 2 → Layer 3."""

from sentence_transformers import SentenceTransformer

from minerva.classifier.base import BaseClassifier
from minerva.classifier.distilbert import DistilBERTClassifier
from minerva.classifier.nli import NLIClassifier
from minerva.config import MinervaSettings
from minerva.logging_utils import get_logger, log_disagreement
from minerva.router.decision import route_batch
from minerva.schema import Action, ScreeningDecision, Shipment
from minerva.taxonomy.embeddings import TaxonomyEmbeddingIndex
from minerva.taxonomy.loader import load_taxonomy
from minerva.taxonomy.matcher import TaxonomyMatcher

logger = get_logger("pipeline")


class ScreeningPipeline:
    """Orchestrates the full 3-layer hybrid screening pipeline.

    Lifecycle:
        1. __init__: Loads models, builds taxonomy index
        2. screen_batch: Runs all three layers on a batch of shipments
        3. screen: Convenience wrapper for a single shipment
    """

    def __init__(self, settings: MinervaSettings) -> None:
        self._settings = settings

        logger.info("Initializing screening pipeline...")

        # Load taxonomy
        groups = load_taxonomy(settings.taxonomy_path)
        logger.info("Loaded %d taxonomy groups", len(groups))

        # Load embedding model (shared between taxonomy and matcher)
        logger.info("Loading embedding model: %s", settings.embedding_model)
        self._embedding_model = SentenceTransformer(settings.embedding_model)

        # Build taxonomy index
        self._taxonomy_index = TaxonomyEmbeddingIndex(
            groups=groups, model=self._embedding_model
        )
        logger.info(
            "Taxonomy index built: %d phrases across %d groups",
            self._taxonomy_index.num_phrases,
            self._taxonomy_index.num_groups,
        )

        # Create taxonomy matcher
        self._matcher = TaxonomyMatcher(
            index=self._taxonomy_index,
            model=self._embedding_model,
            settings=settings,
        )

        # Create AI classifier based on active phase
        self._classifier: BaseClassifier
        if settings.active_classifier == "distilbert":
            logger.info("Loading DistilBERT classifier from: %s", settings.distilbert_model_path)
            self._classifier = DistilBERTClassifier(settings)
        else:
            logger.info("Loading NLI classifier: %s", settings.nli_model)
            self._classifier = NLIClassifier(settings)

        logger.info("Screening pipeline ready")

    @property
    def matcher(self) -> TaxonomyMatcher:
        return self._matcher

    @property
    def classifier(self) -> BaseClassifier:
        return self._classifier

    def screen_batch(
        self, shipments: list[Shipment]
    ) -> list[ScreeningDecision]:
        """Screen a batch of shipments through all three layers.

        Internally chunks into sub-batches of settings.batch_size
        to control memory usage.

        Args:
            shipments: List of shipments to screen.

        Returns:
            List of ScreeningDecision objects.
        """
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
        """Process a single chunk through all three layers."""
        descriptions = [s.description for s in shipments]

        # Layer 1: Taxonomy matching
        taxonomy_results = self._matcher.match_batch(descriptions)

        # Layer 2: AI classification
        classification_results = self._classifier.classify_batch(descriptions)

        # Layer 3: Confidence-based routing
        decisions = route_batch(
            shipments=shipments,
            taxonomy_results=taxonomy_results,
            classification_results=classification_results,
            routing_config=self._settings.routing,
        )

        # Log disagreements for learning
        self._log_disagreements(shipments, taxonomy_results, decisions)

        return decisions

    def _log_disagreements(
        self,
        shipments: list[Shipment],
        taxonomy_results: list[list],
        decisions: list[ScreeningDecision],
    ) -> None:
        """Log cases where taxonomy and AI classifier disagree."""
        for shipment, tax_hits, decision in zip(
            shipments, taxonomy_results, decisions, strict=True
        ):
            has_taxonomy_hit = len(tax_hits) > 0
            ai_result = decision.classification

            if ai_result is None:
                continue

            # Disagreement: taxonomy blocks but AI says allowed
            if has_taxonomy_hit and ai_result.label.value == "allowed":
                log_disagreement(
                    logger=logger,
                    shipment_id=shipment.id,
                    taxonomy_action="block",
                    ai_action=ai_result.label.value,
                    ai_confidence=ai_result.confidence,
                    description=shipment.description,
                )

            # Disagreement: no taxonomy hit but AI says restricted
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
