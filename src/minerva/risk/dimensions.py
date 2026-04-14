"""Nine risk-dimension assessors.

Each assessor examines one aspect of shipment risk and produces a
DimensionAssessment with severity, score, and human-readable signals.
The officer sees the full picture across all dimensions; the system
does not take action — it surfaces evidence.

Dimensions:
    GOODS             — what the item is (taxonomy + classifier)
    PARTY             — who is involved (entity resolution)
    GEOGRAPHY         — where it's going / from (country risk intel)
    VALUATION         — declared value anomalies
    HS_CODE           — tariff classification risk
    DATA_QUALITY      — completeness & coherence of shipment metadata
    MODEL_UNCERTAINTY — how confident the AI classifier is
    DUAL_USE          — end-use / catch-all concerns (dual-use keywords)
    CROSS_BORDER      — routing, transit, transshipment, diversion
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from minerva.risk.country_risk import (
    CountryRiskDatabase,
    CountryRiskLevel,
    FatfStatus,
)
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
    """Fallback country tier map and per-role weights.

    If a CountryRiskDatabase is supplied to the assessor, it is the
    primary source of truth. The `country_tiers` map is used as a
    fallback for countries not present in the risk database.
    """
    country_tiers: dict[str, int] = field(default_factory=dict)
    origin_weight: float = 1.0
    destination_weight: float = 1.0


_COUNTRY_RISK_VALUE = {
    CountryRiskLevel.UNKNOWN: 0.0,
    CountryRiskLevel.LOW: 0.25,
    CountryRiskLevel.MEDIUM: 0.50,
    CountryRiskLevel.HIGH: 0.80,
    CountryRiskLevel.CRITICAL: 1.00,
}


class GeographyAssessor(BaseAssessor):
    """Geography risk — uses LexisNexis-style country risk data when available,
    falls back to a simple tier map otherwise."""

    dimension = RiskDimension.GEOGRAPHY

    def __init__(
        self,
        config: GeographyConfig | None = None,
        country_db: CountryRiskDatabase | None = None,
    ) -> None:
        self._config = config or GeographyConfig()
        self._db = country_db

    def assess(self, ctx: DimensionContext) -> DimensionAssessment:
        signals: list[DimensionSignal] = []
        ship = ctx.shipment

        signals.extend(
            self._country_signals(ship.origin_country, "origin", self._config.origin_weight)
        )
        signals.extend(
            self._country_signals(
                ship.destination_country, "destination", self._config.destination_weight
            )
        )

        if ship.origin_country is None:
            signals.append(
                DimensionSignal(
                    name="missing_origin", value=0.3, weight=0.3,
                    evidence="Origin country not provided",
                )
            )
        if ship.destination_country is None:
            signals.append(
                DimensionSignal(
                    name="missing_destination", value=0.3, weight=0.3,
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

    def _country_signals(
        self, country: str | None, role: str, weight: float
    ) -> list[DimensionSignal]:
        if country is None:
            return []
        code = country.upper()

        # Prefer country risk database if available and has this country
        if self._db is not None:
            entry = self._db.get(code)
            if entry is not None:
                return list(self._db_signals(entry, role, weight))

        # Fallback: static tier map
        tier = self._config.country_tiers.get(code, 0)
        if tier == 0:
            return []
        value = min(tier / 4.0, 1.0)
        return [
            DimensionSignal(
                name=f"{role}_country_tier_{tier}",
                value=value,
                weight=weight,
                evidence=(
                    f"{role.capitalize()} country {code} is configured as "
                    f"tier {tier} (of 4)"
                ),
            )
        ]

    def _db_signals(self, entry, role: str, weight: float):
        """Yield multiple signals derived from a country risk entry."""
        code = entry.country_code
        country_name = entry.country_name or code

        # Primary overall risk signal
        value = max(
            _COUNTRY_RISK_VALUE[entry.overall_risk],
            entry.overall_score,
        )
        if value > 0:
            yield DimensionSignal(
                name=f"{role}_country_risk",
                value=value,
                weight=weight,
                evidence=(
                    f"{role.capitalize()} country {country_name} "
                    f"({code}) has {entry.overall_risk.value} overall risk "
                    f"(score {entry.overall_score:.2f}, source {entry.source})"
                ),
            )

        if entry.is_sanctioned:
            yield DimensionSignal(
                name=f"{role}_country_sanctioned",
                value=1.0,
                weight=weight * 2.0,
                evidence=(
                    f"{role.capitalize()} country {country_name} ({code}) "
                    f"carries {entry.sanctions_status.value} sanctions"
                ),
            )

        if entry.is_fatf_listed:
            yield DimensionSignal(
                name=f"{role}_fatf_{entry.fatf_status.value}",
                value=0.8 if entry.fatf_status == FatfStatus.BLACK else 0.6,
                weight=weight,
                evidence=(
                    f"{role.capitalize()} country {country_name} ({code}) "
                    f"is on the FATF {entry.fatf_status.value} list"
                ),
            )

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


# --- 9. CROSS-BORDER / ROUTING ------------------------------------------

@dataclass
class CrossBorderConfig:
    """Tunable thresholds and heuristics for cross-border routing risk."""

    # Multi-leg opacity: transit stops above this threshold raise a signal
    many_transit_threshold: int = 2
    # Weight applied to each discovered routing signal
    sanctioned_transit_weight: float = 2.5
    fatf_transit_weight: float = 1.5
    transshipment_hub_weight: float = 0.8
    manufacture_origin_mismatch_weight: float = 1.0
    diversion_neighbor_weight: float = 1.8
    final_destination_mismatch_weight: float = 1.0
    many_transit_weight: float = 0.6


class CrossBorderAssessor(BaseAssessor):
    """Surface routing, transit, transshipment, and diversion risk.

    Uses a CountryRiskDatabase (e.g. loaded from LexisNexis) to evaluate
    every country on the shipment's route: origin, transit stops,
    destination, final destination, and country of manufacture.

    Signals surfaced:
        - Any sanctioned country in the route
        - Transit through FATF-listed jurisdictions
        - Route through known transshipment hubs (HK, SG, AE, PA, ...)
        - Country of manufacture != country of origin (re-export signal)
        - Final destination != immediate destination (multi-leg opacity)
        - Destination adjacent to a sanctioned country (diversion risk)
        - Unusually many transit stops (opacity indicator)
    """

    dimension = RiskDimension.CROSS_BORDER

    def __init__(
        self,
        country_db: CountryRiskDatabase | None = None,
        config: CrossBorderConfig | None = None,
    ) -> None:
        self._db = country_db
        self._config = config or CrossBorderConfig()

    def assess(self, ctx: DimensionContext) -> DimensionAssessment:
        signals: list[DimensionSignal] = []
        ship = ctx.shipment
        cfg = self._config

        route_codes = self._route_codes(ship)

        # --- Signals from the country risk database ---
        if self._db is not None:
            for code, role in route_codes:
                entry = self._db.get(code)
                if entry is None:
                    continue
                country_name = entry.country_name or code

                if entry.is_sanctioned:
                    signals.append(
                        DimensionSignal(
                            name=f"sanctioned_{role}",
                            value=1.0,
                            weight=cfg.sanctioned_transit_weight,
                            evidence=(
                                f"Route includes sanctioned country "
                                f"{country_name} ({code}) as {role} "
                                f"[{entry.sanctions_status.value}]"
                            ),
                        )
                    )
                elif entry.is_fatf_listed:
                    signals.append(
                        DimensionSignal(
                            name=f"fatf_listed_{role}",
                            value=0.75 if entry.fatf_status == FatfStatus.BLACK else 0.55,
                            weight=cfg.fatf_transit_weight,
                            evidence=(
                                f"Route includes FATF "
                                f"{entry.fatf_status.value}-listed country "
                                f"{country_name} ({code}) as {role}"
                            ),
                        )
                    )
                elif entry.overall_risk in (CountryRiskLevel.HIGH, CountryRiskLevel.CRITICAL):
                    signals.append(
                        DimensionSignal(
                            name=f"high_risk_transit_{role}",
                            value=_COUNTRY_RISK_VALUE[entry.overall_risk],
                            weight=cfg.fatf_transit_weight,
                            evidence=(
                                f"Route includes high-risk country "
                                f"{country_name} ({code}) as {role}"
                            ),
                        )
                    )

                if entry.transshipment_hub and role in ("transit", "destination"):
                    signals.append(
                        DimensionSignal(
                            name=f"transshipment_hub_{role}",
                            value=0.45,
                            weight=cfg.transshipment_hub_weight,
                            evidence=(
                                f"Route uses known transshipment hub "
                                f"{country_name} ({code}) as {role}"
                            ),
                        )
                    )

            # --- Diversion risk: destination adjacent to sanctioned country ---
            dest_entry = self._db.get(ship.destination_country)
            if dest_entry is not None and dest_entry.sanctioned_neighbors:
                sanctioned_near = [
                    n for n in dest_entry.sanctioned_neighbors
                    if self._db.is_sanctioned(n)
                ]
                if sanctioned_near:
                    signals.append(
                        DimensionSignal(
                            name="diversion_risk_sanctioned_neighbor",
                            value=0.75,
                            weight=cfg.diversion_neighbor_weight,
                            evidence=(
                                f"Destination {ship.destination_country} borders "
                                f"sanctioned country/ies: {', '.join(sanctioned_near)} "
                                f"(diversion risk)"
                            ),
                        )
                    )

        # --- Manufacture != origin (re-export or misdeclaration) ---
        if (
            ship.country_of_manufacture
            and ship.origin_country
            and ship.country_of_manufacture.upper() != ship.origin_country.upper()
        ):
            signals.append(
                DimensionSignal(
                    name="manufacture_origin_mismatch",
                    value=0.55,
                    weight=cfg.manufacture_origin_mismatch_weight,
                    evidence=(
                        f"Country of manufacture "
                        f"({ship.country_of_manufacture.upper()}) differs from "
                        f"country of origin ({ship.origin_country.upper()}) — "
                        f"possible re-export pattern"
                    ),
                )
            )

        # --- Final destination differs from immediate destination ---
        if (
            ship.final_destination
            and ship.destination_country
            and ship.final_destination.upper() != ship.destination_country.upper()
        ):
            signals.append(
                DimensionSignal(
                    name="final_destination_mismatch",
                    value=0.6,
                    weight=cfg.final_destination_mismatch_weight,
                    evidence=(
                        f"Final destination "
                        f"({ship.final_destination.upper()}) differs from "
                        f"immediate destination "
                        f"({ship.destination_country.upper()}) — "
                        f"multi-leg shipment"
                    ),
                )
            )
            # If the final destination itself is sanctioned/high-risk, escalate
            if self._db is not None:
                final_entry = self._db.get(ship.final_destination)
                if final_entry is not None and final_entry.is_sanctioned:
                    signals.append(
                        DimensionSignal(
                            name="final_destination_sanctioned",
                            value=1.0,
                            weight=cfg.sanctioned_transit_weight,
                            evidence=(
                                f"Declared final destination "
                                f"{ship.final_destination.upper()} is sanctioned "
                                f"({final_entry.sanctions_status.value})"
                            ),
                        )
                    )

        # --- Opacity: many transit stops ---
        n_transit = len(ship.transit_countries or [])
        if n_transit >= cfg.many_transit_threshold:
            signals.append(
                DimensionSignal(
                    name="many_transit_stops",
                    value=min(0.3 + 0.15 * n_transit, 1.0),
                    weight=cfg.many_transit_weight,
                    evidence=(
                        f"Shipment has {n_transit} transit stops — "
                        f"route opacity is elevated"
                    ),
                )
            )

        score = _combine(signals)
        severity = _severity_from_score(score)

        # If any sanctioned-country or final-destination-sanctioned signal
        # is present, force CRITICAL severity regardless of arithmetic.
        if any(
            s.name.startswith("sanctioned_") or s.name == "final_destination_sanctioned"
            for s in signals
        ):
            severity = Severity.CRITICAL

        summary = self._summary(signals, severity, ship)
        return DimensionAssessment(
            dimension=self.dimension,
            severity=severity,
            score=score,
            signals=signals,
            summary=summary,
        )

    @staticmethod
    def _route_codes(ship: Shipment) -> list[tuple[str, str]]:
        """Return (country_code, role) pairs for every leg of the route."""
        pairs: list[tuple[str, str]] = []
        if ship.origin_country:
            pairs.append((ship.origin_country, "origin"))
        for c in ship.transit_countries or []:
            if c:
                pairs.append((c, "transit"))
        if ship.destination_country:
            pairs.append((ship.destination_country, "destination"))
        if ship.final_destination and (
            not ship.destination_country
            or ship.final_destination.upper() != ship.destination_country.upper()
        ):
            pairs.append((ship.final_destination, "final_destination"))
        if ship.country_of_manufacture and (
            not ship.origin_country
            or ship.country_of_manufacture.upper() != ship.origin_country.upper()
        ):
            pairs.append((ship.country_of_manufacture, "manufacture"))
        return pairs

    @staticmethod
    def _summary(signals, severity, ship) -> str:
        if not signals:
            return "No cross-border routing concerns detected."
        hops: list[str] = []
        if ship.origin_country:
            hops.append(ship.origin_country.upper())
        hops.extend([c.upper() for c in (ship.transit_countries or []) if c])
        if ship.destination_country:
            hops.append(ship.destination_country.upper())
        if (
            ship.final_destination
            and (not ship.destination_country
                 or ship.final_destination.upper() != ship.destination_country.upper())
        ):
            hops.append(f"→{ship.final_destination.upper()} (final)")
        route = " → ".join(hops) if hops else "route unspecified"
        return f"Cross-border risk {severity.value}: {route}."
