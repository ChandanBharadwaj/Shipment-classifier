"""Tests for taxonomy embedding index."""

import numpy as np
import pytest

from minerva.schema import RiskLevel, TaxonomyGroup
from minerva.taxonomy.embeddings import TaxonomyEmbeddingIndex


class TestTaxonomyEmbeddingIndex:
    def test_build_index(self, sample_taxonomy, mock_sentence_transformer):
        index = TaxonomyEmbeddingIndex(
            groups=sample_taxonomy, model=mock_sentence_transformer
        )
        # 3 + 3 + 2 = 8 total phrases
        assert index.num_phrases == 8
        assert index.num_groups == 3
        assert index.phrase_embeddings is not None
        assert index.phrase_embeddings.shape[0] == 8

    def test_encode_called_once(self, sample_taxonomy, mock_sentence_transformer):
        TaxonomyEmbeddingIndex(
            groups=sample_taxonomy, model=mock_sentence_transformer
        )
        # Should be called exactly once with all phrases
        mock_sentence_transformer.encode.assert_called_once()
        call_args = mock_sentence_transformer.encode.call_args
        assert len(call_args[0][0]) == 8

    def test_phrase_group_mapping(self, sample_taxonomy, mock_sentence_transformer):
        index = TaxonomyEmbeddingIndex(
            groups=sample_taxonomy, model=mock_sentence_transformer
        )
        # First 3 phrases belong to group 0 (hazardous)
        assert index.get_group_for_phrase(0).group_id == "hazardous_materials"
        assert index.get_group_for_phrase(1).group_id == "hazardous_materials"
        assert index.get_group_for_phrase(2).group_id == "hazardous_materials"
        # Next 3 belong to group 1 (military)
        assert index.get_group_for_phrase(3).group_id == "military_and_dual_use"
        # Last 2 belong to group 2 (wildlife)
        assert index.get_group_for_phrase(6).group_id == "restricted_wildlife"
        assert index.get_group_for_phrase(7).group_id == "restricted_wildlife"

    def test_group_phrase_mask(self, sample_taxonomy, mock_sentence_transformer):
        index = TaxonomyEmbeddingIndex(
            groups=sample_taxonomy, model=mock_sentence_transformer
        )
        mask0 = index.get_group_phrase_mask(0)
        assert mask0.sum() == 3  # hazardous has 3 phrases
        assert mask0[0] and mask0[1] and mask0[2]
        assert not mask0[3]

        mask2 = index.get_group_phrase_mask(2)
        assert mask2.sum() == 2  # wildlife has 2 phrases

    def test_empty_taxonomy(self, mock_sentence_transformer):
        index = TaxonomyEmbeddingIndex(groups=[], model=mock_sentence_transformer)
        assert index.num_phrases == 0
        assert index.num_groups == 0
        mock_sentence_transformer.encode.assert_not_called()

    def test_risk_level_for_group(self, sample_taxonomy, mock_sentence_transformer):
        index = TaxonomyEmbeddingIndex(
            groups=sample_taxonomy, model=mock_sentence_transformer
        )
        assert index.get_risk_level_for_group(0) == RiskLevel.CRITICAL
        assert index.get_risk_level_for_group(2) == RiskLevel.HIGH
