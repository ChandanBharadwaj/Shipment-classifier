"""Layer 3: Confidence-based routing logic.

Pure functions — no ML models, no side effects. Takes pre-computed
Layer 1 (taxonomy) and Layer 2 (AI classifier) outputs plus optional
entity-resolution and risk-scoring inputs, and produces final decisions.

Routing rules (in priority order):
  1. Taxonomy hit                          → BLOCK (non-negotiable)
  2. Entity auto-block match               → BLOCK
  3. Risk tier = CRITICAL                  → MANUAL_REVIEW (even if AI says allowed)
  4. Entity review-threshold match         → MANUAL_REVIEW
  5. AI=RESTRICTED & conf >= threshold     → BLOCK
  6. AI=ALLOWED   & conf >= threshold AND risk <= MEDIUM → APPROVE
  7. Otherwise                             → MANUAL_REVIEW
"""

from __future__ import annotations

from minerva.config import RoutingConfig
from minerva.schema import (
    Action,
    ClassificationResult,
    ClassifierLabel,
    EntityMatch,
    RiskAssessment,
    RiskScore,
    ScreeningDecision,
    Shipment,
    TaxonomyHit,
)


def route(
    shipment: Shipment,
    taxonomy_hits: list[TaxonomyHit],
    classification: ClassificationResult,
    routing_config: RoutingConfig,
    entity_matches: list[EntityMatch] | None = None,
    risk_assessment: RiskAssessment | None = None,
    entity_auto_block_threshold: float = 0.92,
) -> ScreeningDecision:
    """Produce a final screening decision for a single shipment."""
    entity_matches = entity_matches or []

    # 1. Taxonomy hits are non-negotiable
    if taxonomy_hits:
        matched_groups = ", ".join(h.group_name for h in taxonomy_hits)
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
            entity_matches=entity_matches,
            risk_assessment=risk_assessment,
            reason=f"Taxonomy hit: {matched_groups} — {detail_str}",
        )

    # 2. Denied-party auto-block
    auto_block_entities = [
        m for m in entity_matches
        if m.exact_match or m.score >= entity_auto_block_threshold
    ]
    if auto_block_entities:
        matched = ", ".join(
            f"{m.role} '{m.input_party}' -> '{m.matched_party}' (score {m.score:.3f})"
            for m in auto_block_entities
        )
        return ScreeningDecision(
            shipment_id=shipment.id,
            action=Action.BLOCK,
            taxonomy_hits=[],
            classification=classification,
            entity_matches=entity_matches,
            risk_assessment=risk_assessment,
            reason=f"Denied party match: {matched}",
        )

    # 3. Critical risk → always manual review
    if risk_assessment is not None and risk_assessment.score == RiskScore.CRITICAL:
        return ScreeningDecision(
            shipment_id=shipment.id,
            action=Action.MANUAL_REVIEW,
            taxonomy_hits=[],
            classification=classification,
            entity_matches=entity_matches,
            risk_assessment=risk_assessment,
            reason=(
                f"Critical risk tier ({risk_assessment.raw_score:.3f}); "
                f"escalated to manual review"
            ),
        )

    # 4. Entity match below auto-block but above review → manual review
    if entity_matches:
        best_entity = max(entity_matches, key=lambda m: m.score)
        return ScreeningDecision(
            shipment_id=shipment.id,
            action=Action.MANUAL_REVIEW,
            taxonomy_hits=[],
            classification=classification,
            entity_matches=entity_matches,
            risk_assessment=risk_assessment,
            reason=(
                f"Possible denied-party match: {best_entity.role} "
                f"'{best_entity.input_party}' ~ '{best_entity.matched_party}' "
                f"at score {best_entity.score:.3f}"
            ),
        )

    # 5. High-confidence restricted → auto block
    if (
        classification.label == ClassifierLabel.RESTRICTED
        and classification.confidence >= routing_config.auto_block_min_confidence
    ):
        return ScreeningDecision(
            shipment_id=shipment.id,
            action=Action.BLOCK,
            taxonomy_hits=[],
            classification=classification,
            entity_matches=entity_matches,
            risk_assessment=risk_assessment,
            reason=(
                f"AI classified as restricted with confidence "
                f"{classification.confidence:.3f}"
            ),
        )

    # 6. High-confidence allowed AND risk not high+ → auto approve
    if (
        classification.label == ClassifierLabel.ALLOWED
        and classification.confidence >= routing_config.auto_approve_min_confidence
        and (
            risk_assessment is None
            or risk_assessment.score.value <= RiskScore.MEDIUM.value
        )
    ):
        return ScreeningDecision(
            shipment_id=shipment.id,
            action=Action.APPROVE,
            taxonomy_hits=[],
            classification=classification,
            entity_matches=entity_matches,
            risk_assessment=risk_assessment,
            reason=(
                f"AI classified as allowed with confidence "
                f"{classification.confidence:.3f}"
            ),
        )

    # 7. If AI wants to approve but risk is HIGH, escalate
    if (
        classification.label == ClassifierLabel.ALLOWED
        and classification.confidence >= routing_config.auto_approve_min_confidence
        and risk_assessment is not None
        and risk_assessment.score == RiskScore.HIGH
    ):
        return ScreeningDecision(
            shipment_id=shipment.id,
            action=Action.MANUAL_REVIEW,
            taxonomy_hits=[],
            classification=classification,
            entity_matches=entity_matches,
            risk_assessment=risk_assessment,
            reason=(
                f"AI allows (conf {classification.confidence:.3f}) but risk tier "
                f"HIGH — escalated to manual review"
            ),
        )

    # 8. Everything else → manual review
    return ScreeningDecision(
        shipment_id=shipment.id,
        action=Action.MANUAL_REVIEW,
        taxonomy_hits=[],
        classification=classification,
        entity_matches=entity_matches,
        risk_assessment=risk_assessment,
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
    entity_results: list[list[EntityMatch]] | None = None,
    risk_results: list[RiskAssessment] | None = None,
) -> list[ScreeningDecision]:
    """Route a batch of shipments."""
    n = len(shipments)
    if entity_results is None:
        entity_results = [[] for _ in range(n)]
    if risk_results is None:
        risk_results = [None] * n  # type: ignore[list-item]

    return [
        route(
            shipment=s,
            taxonomy_hits=t,
            classification=c,
            routing_config=routing_config,
            entity_matches=e,
            risk_assessment=r,
        )
        for s, t, c, e, r in zip(
            shipments,
            taxonomy_results,
            classification_results,
            entity_results,
            risk_results,
            strict=True,
        )
    ]
