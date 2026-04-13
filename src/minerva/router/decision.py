"""Layer 3: Confidence-based routing logic.

Pure functions — no ML models, no side effects. Takes pre-computed
Layer 1 and Layer 2 outputs and produces final screening decisions.

Routing rules:
  IF taxonomy hit                         → BLOCK  (non-negotiable)
  ELIF label=ALLOWED  & conf ≥ threshold  → APPROVE
  ELIF label=RESTRICTED & conf ≥ threshold → BLOCK
  ELSE                                    → MANUAL_REVIEW
"""

from minerva.config import RoutingConfig
from minerva.schema import (
    Action,
    ClassificationResult,
    ClassifierLabel,
    ScreeningDecision,
    Shipment,
    TaxonomyHit,
)


def route(
    shipment: Shipment,
    taxonomy_hits: list[TaxonomyHit],
    classification: ClassificationResult,
    routing_config: RoutingConfig,
) -> ScreeningDecision:
    """Produce a final screening decision for a single shipment.

    Args:
        shipment: The shipment being screened.
        taxonomy_hits: Layer 1 taxonomy matches (empty if no hits).
        classification: Layer 2 AI classification result.
        routing_config: Confidence thresholds for auto-approve/block.

    Returns:
        Final ScreeningDecision with action and reason.
    """
    # Taxonomy hits are non-negotiable — always block
    if taxonomy_hits:
        matched_groups = ", ".join(
            h.group_name for h in taxonomy_hits
        )
        match_details = []
        for h in taxonomy_hits:
            if h.matched_keyword:
                match_details.append(f"keyword '{h.matched_keyword}'")
            elif h.matched_phrase:
                match_details.append(
                    f"phrase '{h.matched_phrase}' (score={h.similarity_score:.3f})"
                )
        detail_str = "; ".join(match_details) if match_details else "taxonomy match"

        return ScreeningDecision(
            shipment_id=shipment.id,
            action=Action.BLOCK,
            taxonomy_hits=taxonomy_hits,
            classification=classification,
            reason=f"Taxonomy hit: {matched_groups} — {detail_str}",
        )

    # High-confidence allowed → auto approve
    if (
        classification.label == ClassifierLabel.ALLOWED
        and classification.confidence >= routing_config.auto_approve_min_confidence
    ):
        return ScreeningDecision(
            shipment_id=shipment.id,
            action=Action.APPROVE,
            taxonomy_hits=[],
            classification=classification,
            reason=(
                f"AI classified as allowed with confidence "
                f"{classification.confidence:.3f}"
            ),
        )

    # High-confidence restricted → auto block
    if (
        classification.label == ClassifierLabel.RESTRICTED
        and classification.confidence >= routing_config.auto_block_min_confidence
    ):
        return ScreeningDecision(
            shipment_id=shipment.id,
            action=Action.BLOCK,
            taxonomy_hits=[],
            classification=classification,
            reason=(
                f"AI classified as restricted with confidence "
                f"{classification.confidence:.3f}"
            ),
        )

    # Everything else → manual review
    return ScreeningDecision(
        shipment_id=shipment.id,
        action=Action.MANUAL_REVIEW,
        taxonomy_hits=[],
        classification=classification,
        reason=(
            f"Uncertain: AI classified as {classification.label.value} "
            f"with confidence {classification.confidence:.3f} "
            f"(below threshold)"
        ),
    )


def route_batch(
    shipments: list[Shipment],
    taxonomy_results: list[list[TaxonomyHit]],
    classification_results: list[ClassificationResult],
    routing_config: RoutingConfig,
) -> list[ScreeningDecision]:
    """Route a batch of shipments through confidence-based logic.

    Args:
        shipments: List of shipments.
        taxonomy_results: Layer 1 results (one hit list per shipment).
        classification_results: Layer 2 results (one per shipment).
        routing_config: Confidence thresholds.

    Returns:
        List of final ScreeningDecision objects.
    """
    return [
        route(shipment, tax_hits, cls_result, routing_config)
        for shipment, tax_hits, cls_result in zip(
            shipments, taxonomy_results, classification_results, strict=True
        )
    ]
