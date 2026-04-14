"""Entity resolution and denied-party matching.

Cascade matching strategy (per Federal Reserve 2025 cascade pattern):
    1. Exact match (normalized) → instant block
    2. Fuzzy match >= auto_block_threshold → block
    3. Fuzzy match >= review_threshold → manual review
    4. Below review threshold → pass

Uses rapidfuzz for efficient fuzzy matching. Configurable thresholds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from minerva.entity_resolution.denied_party_list import DeniedParty
from minerva.schema import EntityMatch, Shipment


@dataclass
class EntityResolverConfig:
    """Thresholds for the fuzzy matching cascade."""

    exact_match_bonus: float = 1.0
    auto_block_threshold: float = 0.92
    review_threshold: float = 0.80


_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^\w\s]")


def _normalize(name: str) -> str:
    """Normalize a party name for matching: lowercase, strip punctuation, collapse spaces."""
    lowered = name.lower()
    no_punct = _NON_ALNUM_RE.sub(" ", lowered)
    collapsed = _WHITESPACE_RE.sub(" ", no_punct).strip()
    return collapsed


class EntityResolver:
    """Resolve shipment parties against denied-party lists."""

    def __init__(
        self,
        parties: list[DeniedParty],
        config: EntityResolverConfig | None = None,
    ) -> None:
        self._parties = parties
        self._config = config or EntityResolverConfig()

        # Build lookup: normalized_name → (original_name, denied_party)
        self._lookup: dict[str, tuple[str, DeniedParty]] = {}
        for party in parties:
            for name in party.all_names:
                normalized = _normalize(name)
                if normalized:
                    self._lookup[normalized] = (name, party)

        self._normalized_names: list[str] = list(self._lookup.keys())

    def resolve_shipment(self, shipment: Shipment) -> list[EntityMatch]:
        """Return any denied-party matches for this shipment."""
        matches: list[EntityMatch] = []

        if shipment.consignee:
            match = self._match_party(shipment.consignee, "consignee")
            if match:
                matches.append(match)

        if shipment.shipper:
            match = self._match_party(shipment.shipper, "shipper")
            if match:
                matches.append(match)

        return matches

    def resolve_batch(
        self, shipments: list[Shipment]
    ) -> list[list[EntityMatch]]:
        return [self.resolve_shipment(s) for s in shipments]

    def _match_party(
        self,
        party_name: str,
        role: str,
    ) -> EntityMatch | None:
        """Match a single party name against the denied list."""
        if not self._normalized_names:
            return None

        normalized_input = _normalize(party_name)
        if not normalized_input:
            return None

        # 1. Exact match
        if normalized_input in self._lookup:
            original_name, party = self._lookup[normalized_input]
            return EntityMatch(
                matched_party=party.name,
                input_party=party_name,
                role=role,
                score=1.0,
                list_name=party.list_name,
                exact_match=True,
            )

        # 2. Fuzzy match via rapidfuzz (use token_set_ratio for robustness)
        result = process.extractOne(
            normalized_input,
            self._normalized_names,
            scorer=fuzz.token_set_ratio,
        )
        if result is None:
            return None

        matched_name, score, _ = result
        score_normalized = score / 100.0

        if score_normalized < self._config.review_threshold:
            return None

        original_name, party = self._lookup[matched_name]
        return EntityMatch(
            matched_party=party.name,
            input_party=party_name,
            role=role,
            score=score_normalized,
            list_name=party.list_name,
            exact_match=False,
        )

    def should_auto_block(self, match: EntityMatch) -> bool:
        """Return True if the match score warrants an auto-block."""
        return (
            match.exact_match
            or match.score >= self._config.auto_block_threshold
        )
