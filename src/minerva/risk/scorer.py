"""Multi-feature risk scoring for shipments.

Combines structured features (origin, destination, consignee, value, HS code)
into a risk tier (1-5). Text classification alone misses 70%+ of real-world
risk signals — this layer surfaces those.

Scoring is config-driven and auditable. Final action can be escalated
based on risk tier (e.g. risk 5 always forces manual review even if
AI says allowed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from minerva.schema import RiskAssessment, RiskScore, RiskSignal, Shipment


@dataclass
class RiskConfig:
    """Configurable risk scoring weights and thresholds."""

    # Country tiers — higher number = higher risk
    country_tiers: dict[str, int] = field(default_factory=dict)
    # Value threshold above which declared value raises a flag
    high_value_threshold: float = 100_000.0
    # Suspicious round-number values (classic TBML indicator)
    round_value_tolerance: float = 0.01
    # HS code chapters considered sensitive
    sensitive_hs_prefixes: list[str] = field(default_factory=list)
    # Weights for each signal (must be positive)
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "origin_country": 0.25,
            "destination_country": 0.25,
            "high_value": 0.15,
            "round_value": 0.10,
            "sensitive_hs": 0.15,
            "missing_fields": 0.10,
        }
    )

    @classmethod
    def from_json(cls, path: Path) -> "RiskConfig":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            country_tiers=data.get("country_tiers", {}),
            high_value_threshold=data.get("high_value_threshold", 100_000.0),
            round_value_tolerance=data.get("round_value_tolerance", 0.01),
            sensitive_hs_prefixes=data.get("sensitive_hs_prefixes", []),
            weights=data.get("weights", cls().weights),
        )


class RiskScorer:
    """Produces a RiskAssessment for a shipment based on structured features."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self._config = config or RiskConfig()

    def score(self, shipment: Shipment) -> RiskAssessment:
        signals: list[RiskSignal] = []

        # 1. Origin country tier
        origin_signal = self._country_signal(
            shipment.origin_country, "origin_country"
        )
        if origin_signal:
            signals.append(origin_signal)

        # 2. Destination country tier
        dest_signal = self._country_signal(
            shipment.destination_country, "destination_country"
        )
        if dest_signal:
            signals.append(dest_signal)

        # 3. High declared value
        value_signal = self._high_value_signal(shipment.declared_value)
        if value_signal:
            signals.append(value_signal)

        # 4. Round-number value (TBML indicator)
        round_signal = self._round_value_signal(shipment.declared_value)
        if round_signal:
            signals.append(round_signal)

        # 5. Sensitive HS code
        hs_signal = self._sensitive_hs_signal(shipment.hs_code)
        if hs_signal:
            signals.append(hs_signal)

        # 6. Missing critical fields (data quality risk)
        missing_signal = self._missing_fields_signal(shipment)
        if missing_signal:
            signals.append(missing_signal)

        raw_score = min(
            sum(s.weight * s.value for s in signals),
            1.0,
        )

        return RiskAssessment(
            score=self._to_tier(raw_score),
            raw_score=raw_score,
            signals=signals,
        )

    def score_batch(
        self, shipments: list[Shipment]
    ) -> list[RiskAssessment]:
        return [self.score(s) for s in shipments]

    # --- Signal helpers ---

    def _country_signal(
        self,
        country: str | None,
        signal_name: str,
    ) -> RiskSignal | None:
        if country is None:
            return None
        tier = self._config.country_tiers.get(country.upper(), 0)
        if tier == 0:
            return None
        # Tier 1..4 → value 0.25, 0.50, 0.75, 1.0
        value = min(tier / 4.0, 1.0)
        return RiskSignal(
            name=signal_name,
            weight=self._config.weights.get(signal_name, 0.25),
            value=value,
            description=f"Country {country.upper()} is tier {tier}",
        )

    def _high_value_signal(
        self, declared_value: float | None
    ) -> RiskSignal | None:
        if declared_value is None:
            return None
        if declared_value < self._config.high_value_threshold:
            return None
        # Saturate at 10x threshold
        value = min(
            declared_value / (self._config.high_value_threshold * 10),
            1.0,
        )
        return RiskSignal(
            name="high_value",
            weight=self._config.weights.get("high_value", 0.15),
            value=value,
            description=(
                f"Declared value {declared_value:.2f} exceeds threshold "
                f"{self._config.high_value_threshold:.2f}"
            ),
        )

    def _round_value_signal(
        self, declared_value: float | None
    ) -> RiskSignal | None:
        """Round-number values like 50,000.00 or 100,000.00 are a classic
        TBML indicator (indicative, not definitive)."""
        if declared_value is None or declared_value <= 0:
            return None
        # Check closeness to multiple of 10,000
        remainder = declared_value % 10_000
        if remainder < self._config.round_value_tolerance * declared_value or \
                remainder > (1 - self._config.round_value_tolerance) * 10_000:
            return RiskSignal(
                name="round_value",
                weight=self._config.weights.get("round_value", 0.1),
                value=0.5,
                description=f"Declared value {declared_value:.2f} is suspiciously round",
            )
        return None

    def _sensitive_hs_signal(
        self, hs_code: str | None
    ) -> RiskSignal | None:
        if hs_code is None:
            return None
        for prefix in self._config.sensitive_hs_prefixes:
            if hs_code.startswith(prefix):
                return RiskSignal(
                    name="sensitive_hs",
                    weight=self._config.weights.get("sensitive_hs", 0.15),
                    value=1.0,
                    description=f"HS code {hs_code} matches sensitive prefix '{prefix}'",
                )
        return None

    def _missing_fields_signal(self, shipment: Shipment) -> RiskSignal | None:
        critical = [
            shipment.origin_country,
            shipment.destination_country,
            shipment.consignee,
            shipment.declared_value,
        ]
        missing_count = sum(1 for v in critical if v is None)
        if missing_count == 0:
            return None
        value = missing_count / len(critical)
        return RiskSignal(
            name="missing_fields",
            weight=self._config.weights.get("missing_fields", 0.1),
            value=value,
            description=f"{missing_count} of {len(critical)} critical fields are missing",
        )

    @staticmethod
    def _to_tier(raw_score: float) -> RiskScore:
        """Map raw score [0.0, 1.0] to RiskScore tier."""
        if raw_score >= 0.80:
            return RiskScore.CRITICAL
        elif raw_score >= 0.60:
            return RiskScore.HIGH
        elif raw_score >= 0.40:
            return RiskScore.MEDIUM
        elif raw_score >= 0.20:
            return RiskScore.LOW
        else:
            return RiskScore.VERY_LOW
