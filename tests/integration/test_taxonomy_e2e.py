"""End-to-end taxonomy matching tests with real embedding model.

These tests require downloading the sentence-transformers model
and are marked as integration tests.
"""

import pytest

from minerva.config import MinervaSettings
from minerva.taxonomy.embeddings import TaxonomyEmbeddingIndex
from minerva.taxonomy.loader import load_taxonomy
from minerva.taxonomy.matcher import TaxonomyMatcher

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def taxonomy_matcher():
    """Build a real taxonomy matcher with the actual embedding model."""
    from sentence_transformers import SentenceTransformer

    settings = MinervaSettings()
    groups = load_taxonomy(settings.taxonomy_path)
    model = SentenceTransformer(settings.embedding_model)
    index = TaxonomyEmbeddingIndex(groups=groups, model=model)
    return TaxonomyMatcher(index=index, model=model, settings=settings)


class TestTaxonomyEndToEnd:
    def test_military_description_triggers_hit(self, taxonomy_matcher):
        results = taxonomy_matcher.match_batch(
            ["Military grade weapon systems for defense contractor"]
        )
        assert len(results[0]) > 0
        hit_groups = {h.group_id for h in results[0]}
        assert "military_and_dual_use" in hit_groups

    def test_keyword_ak47_triggers_hit(self, taxonomy_matcher):
        results = taxonomy_matcher.match_batch(
            ["AK47 rifle parts and ammunition"]
        )
        keyword_hits = [h for h in results[0] if h.matched_keyword]
        assert len(keyword_hits) > 0
        assert any(h.matched_keyword == "ak47" for h in keyword_hits)

    def test_ivory_keyword_triggers_hit(self, taxonomy_matcher):
        results = taxonomy_matcher.match_batch(
            ["Ivory carved decorative pieces"]
        )
        keyword_hits = [h for h in results[0] if h.matched_keyword]
        assert len(keyword_hits) > 0

    def test_benign_descriptions_no_hit(self, taxonomy_matcher):
        benign = [
            "Organic cotton t-shirts bulk shipment",
            "Fresh fruit and vegetables for supermarket",
            "Laptop computers for office use",
        ]
        results = taxonomy_matcher.match_batch(benign)
        for i, desc_hits in enumerate(results):
            assert len(desc_hits) == 0, (
                f"Unexpected hit for '{benign[i]}': {desc_hits}"
            )

    def test_toy_water_pistols_no_hit(self, taxonomy_matcher):
        """The classic false positive case — toy water pistols should NOT trigger."""
        results = taxonomy_matcher.match_batch(
            ["12 plastic toy water pistols for kids summer camp"]
        )
        assert len(results[0]) == 0, (
            f"Toy water pistols should not trigger taxonomy: {results[0]}"
        )
