"""Multi-dimensional risk profile builder.

Aggregates assessments from all enabled dimensions into a single
RiskProfile surfacing every aspect of risk for the compliance officer.

The system does not take action — it surfaces risk. An `advisory_action`
is emitted as a non-binding suggestion; the officer owns the decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from minerva.risk.country_risk import CountryRiskDatabase
from minerva.risk.dimensions import (
    BaseAssessor,
    CrossBorderAssessor,
    CrossBorderConfig,
    DataQualityAssessor,
    DimensionContext,
    DualUseAssessor,
    DualUseConfig,
    GeographyAssessor,
    GeographyConfig,
    GoodsAssessor,
    HsCodeAssessor,
    HsCodeConfig,
    ModelUncertaintyAssessor,
    PartyAssessor,
    ValuationAssessor,
    ValuationConfig,
)
from minerva.schema import (
    Action,
    DimensionAssessment,
    Flag,
    RiskDimension,
    RiskProfile,
    Severity,
    Shipment,
)


@dataclass
class ProfileBuilderConfig:
    """Weights controlling how dimensions combine into the overall score.

    Sum of weights doesn't need to equal 1 — they are normalized at aggregation.
    """

    dimension_weights: dict[RiskDimension, float] = field(
        default_factory=lambda: {
            RiskDimension.GOODS: 2.5,
            RiskDimension.PARTY: 2.5,
            RiskDimension.GEOGRAPHY: 1.2,
            RiskDimension.VALUATION: 1.0,
            RiskDimension.HS_CODE: 1.0,
            RiskDimension.DATA_QUALITY: 0.8,
            RiskDimension.MODEL_UNCERTAINTY: 1.0,
            RiskDimension.DUAL_USE: 1.3,
            RiskDimension.CROSS_BORDER: 2.0,
        }
    )

    @classmethod
    def from_json(cls, path: Path) -> "ProfileBuilderConfig":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        raw_weights = data.get("dimension_weights", {})
        weights = {
            RiskDimension(k): float(v)
            for k, v in raw_weights.items()
            if k in RiskDimension._value2member_map_
        }
        return cls(dimension_weights=weights or cls().dimension_weights)


class RiskProfileBuilder:
    """Runs all dimension assessors and aggregates into a RiskProfile."""

    def __init__(
        self,
        assessors: list[BaseAssessor] | None = None,
        config: ProfileBuilderConfig | None = None,
    ) -> None:
        self._assessors = assessors or default_assessors()
        self._config = config or ProfileBuilderConfig()

    def build(self, ctx: DimensionContext) -> RiskProfile:
        """Build a risk profile for the given context."""
        assessments = [a.assess(ctx) for a in self._assessors]
        flags = self._extract_flags(assessments, ctx)

        overall_score, overall_severity = self._aggregate(assessments, flags)
        advisory_action, advisory_confidence = self._advise(
            overall_severity, flags, assessments
        )
        narrative = self._build_narrative(assessments, flags, overall_severity)

        return RiskProfile(
            shipment_id=ctx.shipment.id,
            overall_severity=overall_severity,
            overall_score=overall_score,
            dimensions=assessments,
            flags=flags,
            narrative=narrative,
            advisory_action=advisory_action,
            advisory_confidence=advisory_confidence,
        )

    def build_batch(
        self, contexts: list[DimensionContext]
    ) -> list[RiskProfile]:
        return [self.build(c) for c in contexts]

    # ---------------- internals ----------------

    def _aggregate(
        self,
        assessments: list[DimensionAssessment],
        flags: list[Flag],
    ) -> tuple[float, Severity]:
        """Produce overall score and severity across dimensions.

        Overall severity = max of (dimension severities, flag severities).
        Overall score = weighted mean of dimension scores.
        """
        weights = self._config.dimension_weights

        total_weight = 0.0
        weighted_sum = 0.0
        max_severity_rank = 0

        for a in assessments:
            w = weights.get(a.dimension, 1.0)
            total_weight += w
            weighted_sum += w * a.score
            max_severity_rank = max(max_severity_rank, a.severity.rank)

        for f in flags:
            max_severity_rank = max(max_severity_rank, f.severity.rank)

        score = weighted_sum / total_weight if total_weight else 0.0
        severity = _severity_from_rank(max_severity_rank)
        return round(min(score, 1.0), 4), severity

    def _extract_flags(
        self,
        assessments: list[DimensionAssessment],
        ctx: DimensionContext,
    ) -> list[Flag]:
        """Surface high-importance individual findings independent of scoring."""
        flags: list[Flag] = []

        # Taxonomy keyword / high-similarity semantic match
        for hit in ctx.taxonomy_hits:
            if hit.matched_keyword:
                flags.append(
                    Flag(
                        name="taxonomy_keyword_hit",
                        severity=Severity.CRITICAL,
                        dimension=RiskDimension.GOODS,
                        description=(
                            f"Hard keyword '{hit.matched_keyword}' matched "
                            f"taxonomy group '{hit.group_name}'"
                        ),
                    )
                )
            elif hit.matched_phrase and (hit.similarity_score or 0) >= 0.85:
                flags.append(
                    Flag(
                        name="taxonomy_semantic_hit",
                        severity=Severity.HIGH,
                        dimension=RiskDimension.GOODS,
                        description=(
                            f"Strong semantic match to taxonomy phrase "
                            f"'{hit.matched_phrase}' (score "
                            f"{hit.similarity_score:.3f})"
                        ),
                    )
                )

        # Entity matches
        for match in ctx.entity_matches:
            if match.exact_match:
                flags.append(
                    Flag(
                        name="denied_party_exact_match",
                        severity=Severity.CRITICAL,
                        dimension=RiskDimension.PARTY,
                        description=(
                            f"Exact denied-party match on {match.role} "
                            f"('{match.matched_party}', list '{match.list_name}')"
                        ),
                    )
                )
            elif match.score >= 0.92:
                flags.append(
                    Flag(
                        name="denied_party_high_fuzzy_match",
                        severity=Severity.HIGH,
                        dimension=RiskDimension.PARTY,
                        description=(
                            f"High-confidence fuzzy match on {match.role} "
                            f"('{match.matched_party}', score {match.score:.3f})"
                        ),
                    )
                )

        return flags

    def _advise(
        self,
        overall_severity: Severity,
        flags: list[Flag],
        assessments: list[DimensionAssessment],
    ) -> tuple[Action, float]:
        """Produce an advisory (non-authoritative) action suggestion.

        The officer is the decision-maker; this is a starting point.
        """
        # CRITICAL severity (usually from a flag or top dimension) — BLOCK
        if overall_severity == Severity.CRITICAL:
            return Action.BLOCK, 0.90

        # HIGH — strongly consider blocking or reviewing
        if overall_severity == Severity.HIGH:
            return Action.MANUAL_REVIEW, 0.75

        # MEDIUM — manual review
        if overall_severity == Severity.MEDIUM:
            return Action.MANUAL_REVIEW, 0.60

        # LOW / NONE — approve with model-uncertainty-aware confidence
        uncertainty = next(
            (a for a in assessments
             if a.dimension == RiskDimension.MODEL_UNCERTAINTY),
            None,
        )
        base_conf = 0.90 if overall_severity == Severity.NONE else 0.80
        if uncertainty is not None and uncertainty.severity.rank >= Severity.HIGH.rank:
            # If AI is very uncertain, our suggestion is also weaker
            return Action.MANUAL_REVIEW, 0.55
        return Action.APPROVE, base_conf

    def _build_narrative(
        self,
        assessments: list[DimensionAssessment],
        flags: list[Flag],
        overall_severity: Severity,
    ) -> str:
        """Compose a narrative explaining the profile for the officer."""
        parts: list[str] = []

        parts.append(
            f"Overall risk: {overall_severity.value.upper()}."
        )

        if flags:
            flag_strs = [f"{f.name}: {f.description}" for f in flags]
            parts.append(
                f"{len(flags)} hard flag(s) — " + "; ".join(flag_strs) + "."
            )

        elevated = [
            a for a in assessments
            if a.severity.rank >= Severity.MEDIUM.rank
        ]
        for a in elevated:
            parts.append(f"[{a.dimension.value}] {a.summary}")

        quiet = [
            a for a in assessments
            if a.severity.rank < Severity.MEDIUM.rank and a.signals
        ]
        if quiet:
            names = ", ".join(a.dimension.value for a in quiet)
            parts.append(f"Low-severity signals in: {names}.")

        parts.append(
            "Action is advisory only — compliance officer makes the final call."
        )
        return " ".join(parts)


def default_assessors(
    geography_config: GeographyConfig | None = None,
    valuation_config: ValuationConfig | None = None,
    hs_code_config: HsCodeConfig | None = None,
    dual_use_config: DualUseConfig | None = None,
    cross_border_config: CrossBorderConfig | None = None,
    country_db: CountryRiskDatabase | None = None,
) -> list[BaseAssessor]:
    """Build the default set of 9 dimension assessors.

    If `country_db` is provided (e.g. loaded from LexisNexis), it is
    shared by the Geography and CrossBorder assessors for richer signals.
    """
    return [
        GoodsAssessor(),
        PartyAssessor(),
        GeographyAssessor(geography_config, country_db=country_db),
        ValuationAssessor(valuation_config),
        HsCodeAssessor(hs_code_config),
        DataQualityAssessor(),
        ModelUncertaintyAssessor(),
        DualUseAssessor(dual_use_config),
        CrossBorderAssessor(country_db=country_db, config=cross_border_config),
    ]


_RANK_TO_SEVERITY = {
    0: Severity.NONE,
    1: Severity.LOW,
    2: Severity.MEDIUM,
    3: Severity.HIGH,
    4: Severity.CRITICAL,
}


def _severity_from_rank(rank: int) -> Severity:
    return _RANK_TO_SEVERITY.get(rank, Severity.NONE)
