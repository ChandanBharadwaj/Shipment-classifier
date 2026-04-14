"""Eight risk-dimension assessors.

Each assessor examines one aspect of shipment risk and produces a
DimensionAssessment with severity, score, and human-readable signals.
The officer sees the full picture across all dimensions; the system
does not take action — it surfaces evidence.

Dimensions:
    GOODS             — what the item is (taxonomy + classifier)
    PARTY             — who is involved (entity resolution)
    GEOGRAPHY         — where it's going / from (country tiers)
    VALUATION         — declared value anomalies
    HS_CODE           — tariff classification risk
    DATA_QUALITY      — completeness & coherence of shipment metadata
    MODEL_UNCERTAINTY — how confident the AI classifier is
    DUAL_USE          — end-use / catch-all concerns (dual-use keywords)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from minerva.schema import (
    ClassificationResult,
    ClassifierLabel,
    DimensionAssessment,
    DimensionSignal,
    EntityMatch,
    RiskDimension,
    Severity,
    Shipment,
    TaxonomyHit,
)


@dataclass
class DimensionContext:
    """Everything an assessor needs, assembled once per shipment."""

    shipment: Shipment
    taxonomy_hits: list[TaxonomyHit] = field(default_factory=list)
    classification: ClassificationResult | None = None
    entity_matches: list[EntityMatch] = field(default_factory=list)


class BaseAssessor(ABC):
    """Each assessor examines one dimension and produces one assessment."""

    dimension: RiskDimension

    @abstractmethod
    def assess(self, ctx: DimensionContext) -> DimensionAssessment:
        ...


def _severity_from_score(score: float) -> Severity:
    """Default mapping from [0,1] raw score → severity tier."""
    if score >= 0.80:
        return Severity.CRITICAL
    if score >= 0.60:
        return Severity.HIGH
    if score >= 0.40:
        return Severity.MEDIUM
    if score >= 0.15:
        return Severity.LOW
    return Severity.NONE


def _combine(signals: list[DimensionSignal]) -> float:
    """Weighted mean of signal values, capped at 1.0."""
    if not signals:
        return 0.0
    total_weight = sum(max(s.weight, 0.0) for s in signals)
    if total_weight == 0:
        return 0.0
    weighted = sum(s.value * s.weight for s in signals)
    return min(weighted / total_weight, 1.0)


# --- 1. GOODS -----------------------------------------------------------

class GoodsAssessor(BaseAssessor):
    dimension = RiskDimension.GOODS

    def assess(self, ctx: DimensionContext) -> DimensionAssessment:
        signals: list[DimensionSignal] = []

        # Taxonomy hits dominate this dimension
        for hit in ctx.taxonomy_hits:
            if hit.matched_keyword:
                signals.append(
                    DimensionSignal(
                        name="taxonomy_keyword",
                        value=1.0,
                        weight=2.0,
                        evidence=(
                            f"Hard keyword '{hit.matched_keyword}' matched "
                            f"taxonomy group '{hit.group_name}'"
                        ),
                    )
                )
            elif hit.matched_phrase:
                score = hit.similarity_score or 0.0
                signals.append(
                    DimensionSignal(
                        name="taxonomy_semantic",
                        value=min(score + 0.1, 1.0),
                        weight=1.5,
                        evidence=(
                            f"Semantic phrase '{hit.matched_phrase}' matched "
                            f"taxonomy group '{hit.group_name}' "
                            f"at similarity {score:.3f}"
                        ),
                    )
                )

        # AI classifier verdict — restricted/needs_review raises goods risk
        if ctx.classification is not None:
            cls = ctx.classification
            if cls.label == ClassifierLabel.RESTRICTED:
                signals.append(
                    DimensionSignal(
                        name="classifier_restricted",
                        value=cls.confidence,
                        weight=1.0,
                        evidence=(
                            f"AI classifier labeled the item as 'restricted' "
                            f"(confidence {cls.confidence:.3f})"
                        ),
                    )
                )
            elif cls.label == ClassifierLabel.NEEDS_REVIEW:
                signals.append(
                    DimensionSignal(
                        name="classifier_needs_review",
                        value=cls.confidence * 0.6,
                        weight=1.0,
                        evidence=(
                            f"AI classifier flagged the item as 'needs review' "
                            f"(confidence {cls.confidence:.3f})"
                        ),
                    )
                )

        # Keyword-based taxonomy hit forces CRITICAL severity
        has_keyword_hit = any(s.name == "taxonomy_keyword" for s in signals)
        score = _combine(signals)
        severity = Severity.CRITICAL if has_keyword_hit else _severity_from_score(score)

        summary = self._summary(signals, severity)
        return DimensionAssessment(
            dimension=self.dimension,
            severity=severity,
            score=score,
            signals=signals,
            summary=summary,
        )

    @staticmethod
    def _summary(signals, severity) -> str:
        if not signals:
            return "No goods-related risk detected."
        if severity == Severity.CRITICAL:
            return "Goods classification indicates a prohibited or highly restricted item."
        if severity == Severity.HIGH:
            return "Goods classification strongly suggests a restricted item."
        if severity == Severity.MEDIUM:
            return "Goods classification raises moderate concerns."
        return "Minor goods-classification signals present."


# --- 2. PARTY -----------------------------------------------------------

class PartyAssessor(BaseAssessor):
    dimension = RiskDimension.PARTY

    def assess(self, ctx: DimensionContext) -> DimensionAssessment:
        signals: list[DimensionSignal] = []
        has_exact = False

        for match in ctx.entity_matches:
            if match.exact_match:
                has_exact = True
                signals.append(
                    DimensionSignal(
                        name="denied_party_exact",
                        value=1.0,
                        weight=2.0,
                        evidence=(
                            f"{match.role.capitalize()} '{match.input_party}' "
                            f"is an EXACT match to denied party "
                            f"'{match.matched_party}' on list '{match.list_name}'"
                        ),
                    )
                )
            elif match.score >= 0.92:
                signals.append(
                    DimensionSignal(
                        name="denied_party_high_fuzzy",
                        value=match.score,
                        weight=1.5,
                        evidence=(
                            f"{match.role.capitalize()} '{match.input_party}' "
                            f"closely resembles denied party "
                            f"'{match.matched_party}' (score {match.score:.3f})"
                        ),
                    )
                )
            else:
                signals.append(
                    DimensionSignal(
                        name="denied_party_fuzzy",
                        value=match.score * 0.8,
                        weight=1.0,
                        evidence=(
                            f"{match.role.capitalize()} '{match.input_party}' "
                            f"fuzzily matches denied party "
                            f"'{match.matched_party}' (score {match.score:.3f})"
                        ),
                    )
                )

        # No consignee / shipper information is itself a mild risk signal
        ship = ctx.shipment
        if ship.consignee is None and ship.shipper is None:
            signals.append(
                DimensionSignal(
                    name="no_party_information",
                    value=0.3,
                    weight=0.5,
                    evidence="No consignee or shipper information provided",
                )
            )

        score = _combine(signals)
        severity = Severity.CRITICAL if has_exact else _severity_from_score(score)

        summary = (
            "No party / denied-list concerns detected."
            if not signals else
            self._summary_from_severity(severity)
        )
        return DimensionAssessment(
            dimension=self.dimension,
            severity=severity,
            score=score,
            signals=signals,
            summary=summary,
        )

    @staticmethod
    def _summary_from_severity(severity) -> str:
        return {
            Severity.CRITICAL: "Exact denied-party match on a shipment party.",
            Severity.HIGH: "Close fuzzy match to a denied party.",
            Severity.MEDIUM: "Possible denied-party match requiring investigation.",
            Severity.LOW: "Low-confidence or secondary party concerns.",
            Severity.NONE: "No party concerns.",
        }[severity]


# --- 3. GEOGRAPHY -------------------------------------------------------

@dataclass
class GeographyConfig:
    """Country tier map (1 = low risk, 4 = very high risk) and weights."""
    country_tiers: dict[str, int] = field(default_factory=dict)
    origin_weight: float = 1.0
    destination_weight: float = 1.0


class GeographyAssessor(BaseAssessor):
    dimension = RiskDimension.GEOGRAPHY

    def __init__(self, config: GeographyConfig | None = None) -> None:
        self._config = config or GeographyConfig()

    def assess(self, ctx: DimensionContext) -> DimensionAssessment:
        signals: list[DimensionSignal] = []
        ship = ctx.shipment

        signals.extend(self._country_signal(ship.origin_country, "origin", self._config.origin_weight))
        signals.extend(self._country_signal(ship.destination_country, "destination", self._config.destination_weight))

        # Data completeness for geography
        if ship.origin_country is None:
            signals.append(
                DimensionSignal(
                    name="missing_origin",
                    value=0.3,
                    weight=0.3,
                    evidence="Origin country not provided",
                )
            )
        if ship.destination_country is None:
            signals.append(
                DimensionSignal(
                    name="missing_destination",
                    value=0.3,
                    weight=0.3,
                    evidence="Destination country not provided",
                )
            )

        score = _combine(signals)
        severity = _severity_from_score(score)
        summary = self._summary(signals, severity, ship)
        return DimensionAssessment(
            dimension=self.dimension,
            severity=severity,
            score=score,
            signals=signals,
            summary=summary,
        )

    def _country_signal(
        self, country: str | None, role: str, weight: float
    ) -> list[DimensionSignal]:
        if country is None:
            return []
        tier = self._config.country_tiers.get(country.upper(), 0)
        if tier == 0:
            return []
        # Tier 1..4 maps to 0.25, 0.50, 0.75, 1.0
        value = min(tier / 4.0, 1.0)
        return [
            DimensionSignal(
                name=f"{role}_country_tier_{tier}",
                value=value,
                weight=weight,
                evidence=(
                    f"{role.capitalize()} country {country.upper()} is "
                    f"configured as tier {tier} (of 4)"
                ),
            )
        ]

    @staticmethod
    def _summary(signals, severity, ship) -> str:
        if not signals:
            return "No geography information available or no elevated country risk."
        parts = []
        if ship.origin_country:
            parts.append(f"from {ship.origin_country.upper()}")
        if ship.destination_country:
            parts.append(f"to {ship.destination_country.upper()}")
        route = " ".join(parts) or "route unspecified"
        return f"Geography risk {severity.value} ({route})."


# --- 4. VALUATION -------------------------------------------------------

@dataclass
class ValuationConfig:
    high_value_threshold: float = 100_000.0
    very_high_value_threshold: float = 1_000_000.0
    round_value_tolerance: float = 0.01
    round_value_base: float = 10_000.0


class ValuationAssessor(BaseAssessor):
    dimension = RiskDimension.VALUATION

    def __init__(self, config: ValuationConfig | None = None) -> None:
        self._config = config or ValuationConfig()

    def assess(self, ctx: DimensionContext) -> DimensionAssessment:
        signals: list[DimensionSignal] = []
        ship = ctx.shipment
        value = ship.declared_value

        if value is None:
            signals.append(
                DimensionSignal(
                    name="missing_declared_value",
                    value=0.3,
                    weight=1.0,
                    evidence="Declared value not provided",
                )
            )
        else:
            if value <= 0:
                signals.append(
                    DimensionSignal(
                        name="nonpositive_value",
                        value=0.7,
                        weight=1.0,
                        evidence=f"Declared value is {value} (non-positive)",
                    )
                )

            if value >= self._config.very_high_value_threshold:
                signals.append(
                    DimensionSignal(
                        name="very_high_value",
                        value=1.0,
                        weight=1.5,
                        evidence=(
                            f"Declared value {value:,.2f} exceeds "
                            f"{self._config.very_high_value_threshold:,.2f}"
                        ),
                    )
                )
            elif value >= self._config.high_value_threshold:
                signals.append(
                    DimensionSignal(
                        name="high_value",
                        value=0.7,
                        weight=1.0,
                        evidence=(
                            f"Declared value {value:,.2f} exceeds "
                            f"{self._config.high_value_threshold:,.2f}"
                        ),
                    )
                )

            if self._is_round(value):
                signals.append(
                    DimensionSignal(
                        name="suspiciously_round_value",
                        value=0.5,
                        weight=0.8,
                        evidence=(
                            f"Declared value {value:,.2f} is suspiciously "
                            f"round (TBML indicator)"
                        ),
                    )
                )

            if ship.declared_currency is None:
                signals.append(
                    DimensionSignal(
                        name="missing_currency",
                        value=0.25,
                        weight=0.3,
                        evidence="Declared currency not provided",
                    )
                )

        score = _combine(signals)
        severity = _severity_from_score(score)
        summary = self._summary(signals, severity)
        return DimensionAssessment(
            dimension=self.dimension,
            severity=severity,
            score=score,
            signals=signals,
            summary=summary,
        )

    def _is_round(self, value: float) -> bool:
        if value <= 0:
            return False
        base = self._config.round_value_base
        tol = self._config.round_value_tolerance
        remainder = value % base
        return (
            remainder < tol * value
            or remainder > (1 - tol) * base
        )

    @staticmethod
    def _summary(signals, severity) -> str:
        if not signals:
            return "No valuation concerns."
        return f"Valuation risk {severity.value}."


# --- 5. HS CODE ---------------------------------------------------------

@dataclass
class HsCodeConfig:
    sensitive_prefixes: list[str] = field(default_factory=list)


class HsCodeAssessor(BaseAssessor):
    dimension = RiskDimension.HS_CODE

    def __init__(self, config: HsCodeConfig | None = None) -> None:
        self._config = config or HsCodeConfig()

    def assess(self, ctx: DimensionContext) -> DimensionAssessment:
        signals: list[DimensionSignal] = []
        hs = ctx.shipment.hs_code

        if hs is None:
            signals.append(
                DimensionSignal(
                    name="missing_hs_code",
                    value=0.3,
                    weight=1.0,
                    evidence="No HS code provided",
                )
            )
        else:
            cleaned = hs.strip().replace(".", "")
            if not cleaned.isdigit() or len(cleaned) < 4:
                signals.append(
                    DimensionSignal(
                        name="malformed_hs_code",
                        value=0.5,
                        weight=1.0,
                        evidence=f"HS code '{hs}' does not appear well-formed",
                    )
                )

            for prefix in self._config.sensitive_prefixes:
                if cleaned.startswith(prefix.replace(".", "")):
                    signals.append(
                        DimensionSignal(
                            name=f"sensitive_hs_prefix_{prefix}",
                            value=1.0,
                            weight=1.5,
                            evidence=(
                                f"HS code {hs} matches sensitive chapter '{prefix}'"
                            ),
                        )
                    )
                    break

        score = _combine(signals)
        severity = _severity_from_score(score)
        summary = self._summary(signals, severity, hs)
        return DimensionAssessment(
            dimension=self.dimension,
            severity=severity,
            score=score,
            signals=signals,
            summary=summary,
        )

    @staticmethod
    def _summary(signals, severity, hs) -> str:
        if not signals:
            return f"HS code {hs} presents no elevated concerns."
        return f"HS code risk {severity.value}."


# --- 6. DATA QUALITY ----------------------------------------------------

class DataQualityAssessor(BaseAssessor):
    dimension = RiskDimension.DATA_QUALITY

    # Fields considered essential for a complete shipment record
    _CRITICAL_FIELDS = (
        "origin_country",
        "destination_country",
        "consignee",
        "shipper",
        "declared_value",
        "hs_code",
    )

    def assess(self, ctx: DimensionContext) -> DimensionAssessment:
        signals: list[DimensionSignal] = []
        ship = ctx.shipment

        missing = [f for f in self._CRITICAL_FIELDS if getattr(ship, f, None) is None]
        if missing:
            value = len(missing) / len(self._CRITICAL_FIELDS)
            signals.append(
                DimensionSignal(
                    name="missing_fields",
                    value=value,
                    weight=1.0,
                    evidence=(
                        f"{len(missing)} of {len(self._CRITICAL_FIELDS)} "
                        f"critical fields missing: {', '.join(missing)}"
                    ),
                )
            )

        # Coherence: description too short
        desc = (ship.description or "").strip()
        if len(desc) < 10:
            signals.append(
                DimensionSignal(
                    name="very_short_description",
                    value=0.6,
                    weight=1.0,
                    evidence=f"Description is only {len(desc)} characters",
                )
            )

        # Weight coherence: 0 weight with non-zero value is suspicious
        if (
            ship.weight_kg is not None
            and ship.weight_kg <= 0
            and (ship.declared_value or 0) > 0
        ):
            signals.append(
                DimensionSignal(
                    name="zero_weight_positive_value",
                    value=0.5,
                    weight=1.0,
                    evidence=(
                        f"Weight is {ship.weight_kg} but declared value is "
                        f"{ship.declared_value}"
                    ),
                )
            )

        # Same-country origin/destination could be mislabeling
        if (
            ship.origin_country
            and ship.destination_country
            and ship.origin_country.upper() == ship.destination_country.upper()
        ):
            signals.append(
                DimensionSignal(
                    name="origin_equals_destination",
                    value=0.3,
                    weight=0.5,
                    evidence=(
                        f"Origin and destination are both "
                        f"{ship.origin_country.upper()} (possible mislabeling)"
                    ),
                )
            )

        score = _combine(signals)
        severity = _severity_from_score(score)
        summary = (
            "Shipment record is complete and coherent."
            if not signals
            else f"Data-quality risk {severity.value}."
        )
        return DimensionAssessment(
            dimension=self.dimension,
            severity=severity,
            score=score,
            signals=signals,
            summary=summary,
        )


# --- 7. MODEL UNCERTAINTY ----------------------------------------------

class ModelUncertaintyAssessor(BaseAssessor):
    dimension = RiskDimension.MODEL_UNCERTAINTY

    def assess(self, ctx: DimensionContext) -> DimensionAssessment:
        signals: list[DimensionSignal] = []
        cls = ctx.classification

        if cls is None:
            signals.append(
                DimensionSignal(
                    name="no_classifier_output",
                    value=0.8,
                    weight=1.0,
                    evidence="AI classifier produced no result",
                )
            )
            score = 0.8
        else:
            # Uncertainty = 1 - confidence, capped
            uncertainty = max(0.0, 1.0 - cls.confidence)
            signals.append(
                DimensionSignal(
                    name="classifier_uncertainty",
                    value=min(uncertainty * 1.5, 1.0),
                    weight=1.0,
                    evidence=(
                        f"Classifier confidence is {cls.confidence:.3f} "
                        f"(uncertainty {uncertainty:.3f})"
                    ),
                )
            )
            # If label is NEEDS_REVIEW it's explicitly uncertain
            if cls.label == ClassifierLabel.NEEDS_REVIEW:
                signals.append(
                    DimensionSignal(
                        name="classifier_needs_review",
                        value=0.7,
                        weight=1.0,
                        evidence="Classifier explicitly flagged for review",
                    )
                )
            score = _combine(signals)

        severity = _severity_from_score(score)
        summary = self._summary(severity)
        return DimensionAssessment(
            dimension=self.dimension,
            severity=severity,
            score=score,
            signals=signals,
            summary=summary,
        )

    @staticmethod
    def _summary(severity) -> str:
        return {
            Severity.CRITICAL: "AI model is highly uncertain about this shipment.",
            Severity.HIGH: "AI model confidence is low — human judgment recommended.",
            Severity.MEDIUM: "AI model confidence is moderate.",
            Severity.LOW: "AI model is reasonably confident.",
            Severity.NONE: "AI model is highly confident.",
        }[severity]


# --- 8. DUAL-USE --------------------------------------------------------

@dataclass
class DualUseConfig:
    """Catch-all keywords indicating potential dual-use / end-use concerns.

    These are not automatic blocks — they raise officer attention for
    items that may have legitimate civilian use but could also be used
    for restricted applications.
    """

    keywords: list[str] = field(
        default_factory=lambda: [
            "high-performance computing",
            "high performance computing",
            "gpu cluster",
            "encryption",
            "cryptographic",
            "centrifuge",
            "precursor",
            "surveillance",
            "drone",
            "uav",
            "lidar",
            "night vision",
            "thermal imaging",
            "nuclear",
            "biotech",
            "gene synthesis",
            "aerospace grade",
            "military grade",
            "export controlled",
        ]
    )


class DualUseAssessor(BaseAssessor):
    dimension = RiskDimension.DUAL_USE

    def __init__(self, config: DualUseConfig | None = None) -> None:
        self._config = config or DualUseConfig()
        pattern = "|".join(
            re.escape(k) for k in self._config.keywords if k
        )
        self._regex = re.compile(rf"\b({pattern})\b", re.IGNORECASE) if pattern else None

    def assess(self, ctx: DimensionContext) -> DimensionAssessment:
        signals: list[DimensionSignal] = []
        desc = ctx.shipment.description or ""

        if self._regex is not None:
            matches = {m.group(1).lower() for m in self._regex.finditer(desc)}
            if matches:
                # Each match adds weight; multiple matches escalate severity
                value = min(0.4 + 0.2 * len(matches), 1.0)
                signals.append(
                    DimensionSignal(
                        name="dual_use_keyword",
                        value=value,
                        weight=1.5,
                        evidence=(
                            f"Description contains dual-use keyword(s): "
                            f"{', '.join(sorted(matches))}"
                        ),
                    )
                )

        # HS code 84-90 chapters often relate to dual-use machinery/electronics
        hs = ctx.shipment.hs_code
        if hs:
            cleaned = hs.strip().replace(".", "")
            if cleaned[:2] in {"84", "85", "87", "88", "89", "90"}:
                signals.append(
                    DimensionSignal(
                        name="dual_use_hs_chapter",
                        value=0.4,
                        weight=0.8,
                        evidence=(
                            f"HS chapter {cleaned[:2]} commonly includes "
                            f"dual-use machinery/electronics"
                        ),
                    )
                )

        score = _combine(signals)
        severity = _severity_from_score(score)
        summary = (
            "No dual-use indicators detected."
            if not signals
            else f"Dual-use concern level {severity.value}."
        )
        return DimensionAssessment(
            dimension=self.dimension,
            severity=severity,
            score=score,
            signals=signals,
            summary=summary,
        )
