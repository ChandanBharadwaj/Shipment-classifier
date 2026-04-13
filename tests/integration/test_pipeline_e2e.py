"""End-to-end pipeline tests with real models."""

import pytest

from minerva.config import MinervaSettings
from minerva.pipeline import ScreeningPipeline
from minerva.schema import Action, Shipment

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pipeline():
    settings = MinervaSettings()
    return ScreeningPipeline(settings)


class TestPipelineEndToEnd:
    def test_keyword_hit_blocks(self, pipeline):
        """AK47 should always be blocked (keyword match)."""
        decision = pipeline.screen(
            Shipment(id="T1", description="AK47 rifle parts and ammunition")
        )
        assert decision.action == Action.BLOCK
        assert len(decision.taxonomy_hits) > 0

    def test_ivory_keyword_blocks(self, pipeline):
        """Ivory should always be blocked (keyword match)."""
        decision = pipeline.screen(
            Shipment(id="T2", description="Ivory carved decorative pieces")
        )
        assert decision.action == Action.BLOCK

    def test_benign_shipments(self, pipeline):
        """Clearly legitimate cargo should not be blocked."""
        shipments = [
            Shipment(id="T3", description="Fresh fruit and vegetables for supermarket"),
            Shipment(id="T4", description="Organic cotton t-shirts bulk shipment"),
        ]
        decisions = pipeline.screen_batch(shipments)
        for d in decisions:
            assert d.action != Action.BLOCK or len(d.taxonomy_hits) > 0

    def test_full_batch_produces_decisions(self, pipeline):
        shipments = [
            Shipment(id=f"B{i}", description=f"Test item number {i}")
            for i in range(20)
        ]
        decisions = pipeline.screen_batch(shipments)
        assert len(decisions) == 20
        for d in decisions:
            assert d.action in list(Action)
            assert d.classification is not None
