"""Tests for taxonomy matcher (Layer 1)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from minerva.config import MinervaSettings
from minerva.schema import RiskLevel, TaxonomyGroup
from minerva.taxonomy.embeddings import TaxonomyEmbeddingIndex
from minerva.taxonomy.matcher import TaxonomyMatcher


def _make_matcher(
    groups: list[TaxonomyGroup],
    phrase_embeddings: np.ndarray,
    desc_embeddings_fn=None,
    settings: MinervaSettings | None = None,
) -> TaxonomyMatcher:
    """Helper to create a TaxonomyMatcher with controlled embeddings."""
    if settings is None:
        settings = MinervaSettings()

    # Build index with mock model
    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=phrase_embeddings)
    index = TaxonomyEmbeddingIndex(groups=groups, model=mock_model)

    # Create matcher with a model that returns controlled desc embeddings
    desc_model = MagicMock()
    if desc_embeddings_fn:
        desc_model.encode = MagicMock(side_effect=desc_embeddings_fn)
    else:
        desc_model.encode = MagicMock(return_value=np.zeros((1, phrase_embeddings.shape[1])))

    return TaxonomyMatcher(index=index, model=desc_model, settings=settings)


class TestHardKeywordMatching:
    def test_keyword_match(self, sample_taxonomy):
        mock_model = MagicMock()
        # Provide embeddings for the taxonomy phrases
        n_phrases = sum(len(g.semantic_phrases) for g in sample_taxonomy)
        mock_model.encode = MagicMock(return_value=np.zeros((n_phrases, 384)))

        index = TaxonomyEmbeddingIndex(groups=sample_taxonomy, model=mock_model)

        # Matcher: desc encode returns zeros (no semantic match)
        desc_model = MagicMock()
        desc_model.encode = MagicMock(return_value=np.zeros((1, 384)))

        matcher = TaxonomyMatcher(index=index, model=desc_model, settings=MinervaSettings())

        results = matcher.match_batch(["This shipment contains an AK47 rifle"])
        assert len(results) == 1
        hits = results[0]
        # Should find ak47 keyword match
        keyword_hits = [h for h in hits if h.matched_keyword]
        assert len(keyword_hits) >= 1
        assert keyword_hits[0].matched_keyword == "ak47"
        assert keyword_hits[0].group_id == "military_and_dual_use"

    def test_keyword_case_insensitive(self, sample_taxonomy):
        mock_model = MagicMock()
        n_phrases = sum(len(g.semantic_phrases) for g in sample_taxonomy)
        mock_model.encode = MagicMock(return_value=np.zeros((n_phrases, 384)))

        index = TaxonomyEmbeddingIndex(groups=sample_taxonomy, model=mock_model)

        desc_model = MagicMock()
        desc_model.encode = MagicMock(return_value=np.zeros((2, 384)))
        matcher = TaxonomyMatcher(index=index, model=desc_model, settings=MinervaSettings())

        results = matcher.match_batch(["IVORY carvings", "Ivory items"])
        assert len(results) == 2
        for r in results:
            kw_hits = [h for h in r if h.matched_keyword]
            assert len(kw_hits) >= 1
            assert kw_hits[0].matched_keyword == "ivory"

    def test_no_keyword_match(self, sample_taxonomy):
        mock_model = MagicMock()
        n_phrases = sum(len(g.semantic_phrases) for g in sample_taxonomy)
        mock_model.encode = MagicMock(return_value=np.zeros((n_phrases, 384)))

        index = TaxonomyEmbeddingIndex(groups=sample_taxonomy, model=mock_model)

        desc_model = MagicMock()
        desc_model.encode = MagicMock(return_value=np.zeros((1, 384)))
        matcher = TaxonomyMatcher(index=index, model=desc_model, settings=MinervaSettings())

        results = matcher.match_batch(["Cotton t-shirts"])
        assert len(results) == 1
        # No keyword hits (there might be semantic hits but with zero embeddings they'd be below threshold)
        keyword_hits = [h for h in results[0] if h.matched_keyword]
        assert len(keyword_hits) == 0


class TestSemanticMatching:
    def test_high_similarity_triggers_hit(self):
        """When cosine similarity exceeds threshold, a hit is produced."""
        groups = [
            TaxonomyGroup(
                group_id="test_group",
                name="Test",
                risk_level=RiskLevel.CRITICAL,
                semantic_phrases=["dangerous item"],
            ),
        ]
        # Phrase embedding: unit vector in first dimension
        phrase_emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        # Description embedding: very similar (high cosine similarity)
        desc_emb = np.array([[0.99, 0.1, 0.0, 0.0]], dtype=np.float32)
        desc_emb /= np.linalg.norm(desc_emb, axis=1, keepdims=True)

        matcher = _make_matcher(
            groups, phrase_emb,
            desc_embeddings_fn=lambda *a, **kw: desc_emb,
        )
        results = matcher.match_batch(["some dangerous shipment"])
        assert len(results[0]) >= 1
        hit = results[0][0]
        assert hit.group_id == "test_group"
        assert hit.similarity_score is not None
        assert hit.similarity_score >= 0.75

    def test_low_similarity_no_hit(self):
        """When cosine similarity is below threshold, no hit is produced."""
        groups = [
            TaxonomyGroup(
                group_id="test_group",
                name="Test",
                risk_level=RiskLevel.CRITICAL,
                semantic_phrases=["dangerous item"],
            ),
        ]
        phrase_emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        # Description embedding: orthogonal (zero similarity)
        desc_emb = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

        matcher = _make_matcher(
            groups, phrase_emb,
            desc_embeddings_fn=lambda *a, **kw: desc_emb,
        )
        results = matcher.match_batch(["safe shipment"])
        assert len(results[0]) == 0


class TestBatchProcessing:
    def test_empty_batch(self, sample_taxonomy, mock_sentence_transformer):
        index = TaxonomyEmbeddingIndex(groups=sample_taxonomy, model=mock_sentence_transformer)
        matcher = TaxonomyMatcher(
            index=index, model=mock_sentence_transformer, settings=MinervaSettings()
        )
        results = matcher.match_batch([])
        assert results == []

    def test_multiple_descriptions(self, sample_taxonomy):
        mock_model = MagicMock()
        n_phrases = sum(len(g.semantic_phrases) for g in sample_taxonomy)
        mock_model.encode = MagicMock(return_value=np.zeros((n_phrases, 384)))

        index = TaxonomyEmbeddingIndex(groups=sample_taxonomy, model=mock_model)

        desc_model = MagicMock()
        desc_model.encode = MagicMock(return_value=np.zeros((3, 384)))
        matcher = TaxonomyMatcher(index=index, model=desc_model, settings=MinervaSettings())

        results = matcher.match_batch(["item 1", "item 2", "item 3"])
        assert len(results) == 3
