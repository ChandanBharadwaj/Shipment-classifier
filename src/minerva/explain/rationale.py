"""Rationale generation for screening decisions.

Produces human-readable explanations for each decision — required for
EU AI Act audit trails (enforcement from August 2026) and WCO guidance.

When a RiskProfile is attached to the decision, the rationale surfaces
each dimension's severity, evidence, and contributing signals — helping
the compliance officer see every aspect of risk at a glance.
"""

from __future__ import annotations

from typing import Protocol

from minerva.schema import (
    DimensionAssessment,
    EntityMatch,
    Flag,
    RiskAssessment,
    RiskProfile,
    ScreeningDecision,
    Severity,
    Shipment,
)


class RationaleGenerator(Protocol):
    def generate(
        self,
        shipment: Shipment,
        decision: ScreeningDecision,
    ) -> str:
        ...


class TemplatedRationaleGenerator:
    """Default rationale generator using structured templates.

    Produces deterministic, auditable rationales. If a RiskProfile is
    present, the rationale leads with the multi-dimensional view.
    """

    def generate(
        self,
        shipment: Shipment,
        decision: ScreeningDecision,
    ) -> str:
        sections: list[str] = []
        sections.append(self._header(shipment, decision))

        profile = decision.risk_profile
        if profile is not None:
            sections.append(self._profile_section(profile))
        else:
            # Fall back to legacy rationale when no profile is present
            if decision.taxonomy_hits:
                sections.append(self._taxonomy_section(decision))
            if decision.classification is not None:
                sections.append(self._classification_section(decision))
            if decision.entity_matches:
                sections.append(self._entity_section(decision.entity_matches))
            if decision.risk_assessment is not None:
                sections.append(self._risk_section(decision.risk_assessment))

        sections.append(
            f"Advisory action (officer-owned decision): "
            f"{decision.action.value.upper()}. Reason: {decision.reason}"
        )
        return " ".join(sections)

    # ---------------- header ----------------

    @staticmethod
    def _header(shipment: Shipment, decision: ScreeningDecision) -> str:
        desc = shipment.description
        preview = desc[:120] + "..." if len(desc) > 120 else desc
        return (
            f"Shipment {shipment.id} (\"{preview}\") was screened through the "
            f"Minerva hybrid pipeline."
        )

    # ---------------- profile-driven narrative ----------------

    @staticmethod
    def _profile_section(profile: RiskProfile) -> str:
        parts: list[str] = []
        parts.append(
            f"OVERALL RISK: {profile.overall_severity.value.upper()} "
            f"(score {profile.overall_score:.3f})."
        )

        if profile.flags:
            parts.append(
                f"{len(profile.flags)} hard flag(s): "
                + "; ".join(
                    f"[{f.severity.value.upper()}] {f.description}"
                    for f in profile.flags
                )
                + "."
            )

        # Elevated dimensions first
        elevated = [
            d for d in profile.dimensions
            if d.severity.rank >= Severity.MEDIUM.rank
        ]
        if elevated:
            parts.append("Elevated dimensions:")
            for d in elevated:
                parts.append(
                    f"[{d.dimension.value} / {d.severity.value}] {d.summary}"
                )
                if d.signals:
                    parts.append(
                        "Evidence: "
                        + "; ".join(s.evidence for s in d.signals)
                        + "."
                    )

        # Low-severity dimensions summarized
        quiet = [
            d for d in profile.dimensions
            if d.severity.rank < Severity.MEDIUM.rank and d.signals
        ]
        if quiet:
            parts.append(
                "Low-severity dimensions with signals: "
                + ", ".join(d.dimension.value for d in quiet)
                + "."
            )

        clean = [
            d for d in profile.dimensions
            if d.severity.rank == 0 and not d.signals
        ]
        if clean:
            parts.append(
                "No concerns in: "
                + ", ".join(d.dimension.value for d in clean)
                + "."
            )

        parts.append(
            f"System's advisory suggestion: "
            f"{profile.advisory_action.value.upper()} "
            f"(confidence {profile.advisory_confidence:.2f}). "
            f"Compliance officer makes the final call."
        )
        return " ".join(parts)

    # ---------------- legacy sections (when no profile) ----------------

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
    """Produce a new decision list with rationales attached."""
    updated: list[ScreeningDecision] = []
    for shipment, decision in zip(shipments, decisions, strict=True):
        rationale = generator.generate(shipment, decision)
        updated.append(
            decision.model_copy(update={"rationale": rationale})
        )
    return updated
