"""Tests for entity resolution and denied-party matching."""

import json

import pytest

from minerva.entity_resolution.denied_party_list import (
    DeniedParty,
    load_denied_parties,
)
from minerva.entity_resolution.resolver import (
    EntityResolver,
    EntityResolverConfig,
)
from minerva.schema import Shipment


@pytest.fixture
def parties():
    return [
        DeniedParty(
            name="Blocked Industries Ltd",
            aliases=["Blocked Industries", "Blocked Industries Limited"],
            country="XX",
            list_name="test_list",
        ),
        DeniedParty(
            name="Sanctioned Holding Corp",
            aliases=["Sanctioned Holdings"],
            country="YY",
            list_name="test_list",
        ),
    ]


@pytest.fixture
def resolver(parties):
    return EntityResolver(
        parties=parties,
        config=EntityResolverConfig(
            auto_block_threshold=0.92,
            review_threshold=0.80,
        ),
    )


class TestLoadDeniedParties:
    def test_load_from_dict(self, tmp_path):
        data = {
            "list_name": "my_list",
            "parties": [
                {"name": "Party A", "aliases": ["A Inc"], "country": "XX"},
                {"name": "Party B"},
            ],
        }
        path = tmp_path / "parties.json"
        path.write_text(json.dumps(data))

        result = load_denied_parties(path)
        assert len(result) == 2
        assert result[0].name == "Party A"
        assert "A Inc" in result[0].aliases
        assert result[1].name == "Party B"

    def test_load_from_list(self, tmp_path):
        data = [{"name": "Party X"}]
        path = tmp_path / "parties.json"
        path.write_text(json.dumps(data))

        result = load_denied_parties(path)
        assert len(result) == 1
        assert result[0].name == "Party X"

    def test_missing_file_returns_empty(self, tmp_path):
        result = load_denied_parties(tmp_path / "nonexistent.json")
        assert result == []

    def test_skips_missing_name(self, tmp_path):
        data = {"parties": [{"country": "XX"}, {"name": "Valid"}]}
        path = tmp_path / "parties.json"
        path.write_text(json.dumps(data))

        result = load_denied_parties(path)
        assert len(result) == 1
        assert result[0].name == "Valid"


class TestEntityResolver:
    def test_exact_match(self, resolver):
        shipment = Shipment(
            id="S1", description="test", consignee="Blocked Industries Ltd"
        )
        matches = resolver.resolve_shipment(shipment)
        assert len(matches) == 1
        assert matches[0].exact_match is True
        assert matches[0].score == 1.0
        assert matches[0].role == "consignee"

    def test_exact_match_normalized(self, resolver):
        # Different punctuation, same name
        shipment = Shipment(
            id="S1", description="test", consignee="BLOCKED INDUSTRIES, LTD."
        )
        matches = resolver.resolve_shipment(shipment)
        assert len(matches) == 1
        assert matches[0].exact_match is True

    def test_alias_match(self, resolver):
        shipment = Shipment(
            id="S1", description="test", consignee="Blocked Industries Limited"
        )
        matches = resolver.resolve_shipment(shipment)
        assert len(matches) == 1
        assert matches[0].matched_party == "Blocked Industries Ltd"

    def test_fuzzy_match(self, resolver):
        # Slight typo
        shipment = Shipment(
            id="S1", description="test", consignee="Blockd Industries Ltd"
        )
        matches = resolver.resolve_shipment(shipment)
        assert len(matches) == 1
        assert not matches[0].exact_match
        assert matches[0].score >= 0.80

    def test_no_match(self, resolver):
        shipment = Shipment(
            id="S1", description="test", consignee="Totally Unrelated Company"
        )
        matches = resolver.resolve_shipment(shipment)
        assert matches == []

    def test_no_consignee_no_match(self, resolver):
        shipment = Shipment(id="S1", description="test")
        matches = resolver.resolve_shipment(shipment)
        assert matches == []

    def test_shipper_match(self, resolver):
        shipment = Shipment(
            id="S1", description="test", shipper="Sanctioned Holding Corp"
        )
        matches = resolver.resolve_shipment(shipment)
        assert len(matches) == 1
        assert matches[0].role == "shipper"

    def test_both_parties_match(self, resolver):
        shipment = Shipment(
            id="S1",
            description="test",
            consignee="Blocked Industries Ltd",
            shipper="Sanctioned Holding Corp",
        )
        matches = resolver.resolve_shipment(shipment)
        assert len(matches) == 2
        roles = {m.role for m in matches}
        assert roles == {"consignee", "shipper"}

    def test_should_auto_block_exact(self, resolver):
        from minerva.schema import EntityMatch

        match = EntityMatch(
            matched_party="X", input_party="x", role="consignee",
            score=1.0, list_name="t", exact_match=True,
        )
        assert resolver.should_auto_block(match)

    def test_should_auto_block_high_fuzzy(self, resolver):
        from minerva.schema import EntityMatch

        match = EntityMatch(
            matched_party="X", input_party="x", role="consignee",
            score=0.95, list_name="t", exact_match=False,
        )
        assert resolver.should_auto_block(match)

    def test_should_not_auto_block_medium_fuzzy(self, resolver):
        from minerva.schema import EntityMatch

        match = EntityMatch(
            matched_party="X", input_party="x", role="consignee",
            score=0.85, list_name="t", exact_match=False,
        )
        assert not resolver.should_auto_block(match)

    def test_batch(self, resolver):
        shipments = [
            Shipment(id="S1", description="t", consignee="Blocked Industries Ltd"),
            Shipment(id="S2", description="t", consignee="Safe Company"),
        ]
        results = resolver.resolve_batch(shipments)
        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 0

    def test_empty_parties_list(self):
        resolver = EntityResolver(parties=[])
        shipment = Shipment(
            id="S1", description="t", consignee="Anyone"
        )
        assert resolver.resolve_shipment(shipment) == []
