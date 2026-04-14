"""Rationale generation for screening decisions.

Produces human-readable explanations for each decision — required for
EU AI Act audit trails (enforcement from August 2026) and WCO guidance.

Default implementation uses templated generation; pluggable for future
LLM-based rationale if needed.
"""

from __future__ import annotations

from typing import Protocol

from minerva.schema import (
    Action,
    EntityMatch,
    RiskAssessment,
    ScreeningDecision,
    Shipment,
)


class RationaleGenerator(Protocol):
    """Interface for rationale generators."""

    def generate(
        self,
        shipment: Shipment,
        decision: ScreeningDecision,
    ) -> str:
        """Produce a human-readable rationale for a decision."""
        ...


class TemplatedRationaleGenerator:
    """Default rationale generator using structured templates.

    Produces deterministic, auditable rationales without external LLM calls.
    Suitable for initial deployment and regulatory audit trails.
    """

    def generate(
        self,
        shipment: Shipment,
        decision: ScreeningDecision,
    ) -> str:
        sections: list[str] = []

        # 1. Decision summary
        sections.append(self._summary(shipment, decision))

        # 2. Taxonomy evidence
        if decision.taxonomy_hits:
            sections.append(self._taxonomy_section(decision))

        # 3. AI classification
        if decision.classification is not None:
            sections.append(self._classification_section(decision))

        # 4. Entity resolution
        if decision.entity_matches:
            sections.append(self._entity_section(decision.entity_matches))

        # 5. Risk assessment
        if decision.risk_assessment is not None:
            sections.append(self._risk_section(decision.risk_assessment))

        # 6. Final reason
        sections.append(f"Final action: {decision.action.value.upper()}.")

        return " ".join(sections)

    @staticmethod
    def _summary(shipment: Shipment, decision: ScreeningDecision) -> str:
        desc_preview = shipment.description[:120]
        if len(shipment.description) > 120:
            desc_preview += "..."
        return (
            f"Shipment {shipment.id} (\"{desc_preview}\") was screened through "
            f"the hybrid pipeline."
        )

    @staticmethod
    def _taxonomy_section(decision: ScreeningDecision) -> str:
        hits = decision.taxonomy_hits
        parts = []
        for hit in hits:
            if hit.matched_keyword:
                parts.append(
                    f"matched hard keyword '{hit.matched_keyword}' in group "
                    f"'{hit.group_name}' (risk: {hit.risk_level.value})"
                )
            elif hit.matched_phrase:
                score = hit.similarity_score or 0.0
                parts.append(
                    f"matched semantic phrase '{hit.matched_phrase}' in group "
                    f"'{hit.group_name}' at similarity {score:.3f} "
                    f"(risk: {hit.risk_level.value})"
                )
        detail = "; ".join(parts)
        return (
            f"Layer 1 (Taxonomy) produced {len(hits)} hit(s): {detail}. "
            f"Taxonomy hits are non-negotiable and always result in BLOCK."
        )

    @staticmethod
    def _classification_section(decision: ScreeningDecision) -> str:
        cls = decision.classification
        assert cls is not None
        return (
            f"Layer 2 (AI Classifier) classified the shipment as "
            f"{cls.label.value.upper()} with confidence {cls.confidence:.3f}."
        )

    @staticmethod
    def _entity_section(matches: list[EntityMatch]) -> str:
        parts = []
        for m in matches:
            match_type = "exact match" if m.exact_match else f"fuzzy match at {m.score:.3f}"
            parts.append(
                f"{m.role} '{m.input_party}' matched denied party "
                f"'{m.matched_party}' ({match_type}, list: {m.list_name})"
            )
        return f"Entity resolution found {len(matches)} match(es): " + "; ".join(parts) + "."

    @staticmethod
    def _risk_section(risk: RiskAssessment) -> str:
        signal_summary = ", ".join(
            f"{s.name}={s.value:.2f}" for s in risk.signals
        ) or "none"
        return (
            f"Multi-feature risk assessment: tier {risk.score.value} "
            f"(raw score {risk.raw_score:.3f}); contributing signals: "
            f"{signal_summary}."
        )


def attach_rationale(
    generator: RationaleGenerator,
    shipments: list[Shipment],
    decisions: list[ScreeningDecision],
) -> list[ScreeningDecision]:
    """Produce a new decision list with rationales attached.

    Returns new ScreeningDecision instances since the model is frozen.
    """
    updated: list[ScreeningDecision] = []
    for shipment, decision in zip(shipments, decisions, strict=True):
        rationale = generator.generate(shipment, decision)
        updated.append(
            decision.model_copy(update={"rationale": rationale})
        )
    return updated
