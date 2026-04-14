"""Tests for the screening pipeline orchestration."""

from unittest.mock import MagicMock, patch

import pytest

from minerva.config import MinervaSettings
from minerva.schema import (
    Action,
    ClassificationResult,
    ClassifierLabel,
    RiskLevel,
    ScreeningDecision,
    Shipment,
    TaxonomyHit,
)


class TestScreeningPipeline:
    """Test pipeline orchestration with all layers mocked."""

    @pytest.fixture
    def mock_pipeline(self, tmp_path, sample_taxonomy_path):
        """Create a pipeline with mocked models."""
        settings = MinervaSettings(
            config_dir=sample_taxonomy_path.parent.parent / "fixtures",
        )
        # We need to patch the taxonomy path to point to our fixture
        settings_dict = {
            "config_dir": sample_taxonomy_path.parent,
        }

        with (
            patch("minerva.pipeline.SentenceTransformer") as MockST,
            patch("minerva.pipeline.NLIClassifier") as MockNLI,
            patch("minerva.pipeline.load_taxonomy") as MockLoad,
            patch("minerva.pipeline.TaxonomyEmbeddingIndex") as MockIndex,
            patch("minerva.pipeline.TaxonomyMatcher") as MockMatcher,
            patch("minerva.pipeline.AIRiskEngine") as MockAI,
        ):
            import numpy as np
            from minerva.schema import TaxonomyGroup

            # Mock taxonomy loading
            MockLoad.return_value = [
                TaxonomyGroup(
                    group_id="test",
                    name="Test",
                    risk_level=RiskLevel.CRITICAL,
                    semantic_phrases=["test phrase"],
                ),
            ]

            # Mock embedding model
            mock_st = MagicMock()
            MockST.return_value = mock_st

            # Mock index
            mock_index = MagicMock()
            mock_index.num_phrases = 1
            mock_index.num_groups = 1
            MockIndex.return_value = mock_index

            # Mock matcher - returns no taxonomy hits by default
            mock_matcher = MagicMock()
            mock_matcher.match_batch.return_value = [[] for _ in range(10)]
            MockMatcher.return_value = mock_matcher

            # Mock NLI classifier
            mock_nli = MagicMock()
            mock_nli.classify_batch.return_value = [
                ClassificationResult(label=ClassifierLabel.ALLOWED, confidence=0.95)
                for _ in range(10)
            ]
            MockNLI.return_value = mock_nli

            # Mock AI risk engine — returns one None AISignals per description
            mock_ai = MagicMock()
            mock_ai.hs_chapter_codes = []
            mock_ai.dual_use_concepts = []
            mock_ai.hs_top_k = 3
            mock_ai.dual_use_threshold = 0.55
            mock_ai.build_ai_signals_batch.side_effect = (
                lambda descriptions, hs_codes, **kw: [None] * len(descriptions)
            )
            MockAI.return_value = mock_ai

            from minerva.pipeline import ScreeningPipeline

            pipeline = ScreeningPipeline(MinervaSettings())

            # Store mocks for verification
            pipeline._test_mocks = {
                "matcher": mock_matcher,
                "nli": mock_nli,
            }

            yield pipeline

    def test_screen_batch_calls_all_layers(self, mock_pipeline):
        shipments = [
            Shipment(id="S1", description="test item 1"),
            Shipment(id="S2", description="test item 2"),
        ]

        # Update mock return values for correct batch size
        mock_pipeline._test_mocks["matcher"].match_batch.return_value = [[], []]
        mock_pipeline._test_mocks["nli"].classify_batch.return_value = [
            ClassificationResult(label=ClassifierLabel.ALLOWED, confidence=0.95),
            ClassificationResult(label=ClassifierLabel.ALLOWED, confidence=0.92),
        ]

        decisions = mock_pipeline.screen_batch(shipments)

        assert len(decisions) == 2
        # Verify all layers were called
        mock_pipeline._test_mocks["matcher"].match_batch.assert_called_once()
        mock_pipeline._test_mocks["nli"].classify_batch.assert_called_once()

    def test_screen_single_convenience(self, mock_pipeline):
        mock_pipeline._test_mocks["matcher"].match_batch.return_value = [[]]
        mock_pipeline._test_mocks["nli"].classify_batch.return_value = [
            ClassificationResult(label=ClassifierLabel.ALLOWED, confidence=0.95),
        ]

        decision = mock_pipeline.screen(
            Shipment(id="S1", description="test item")
        )
        assert isinstance(decision, ScreeningDecision)

    def test_screen_empty_batch(self, mock_pipeline):
        decisions = mock_pipeline.screen_batch([])
        assert decisions == []

    def test_taxonomy_hit_produces_block(self, mock_pipeline):
        hit = TaxonomyHit(
            group_id="mil",
            group_name="Military",
            risk_level=RiskLevel.CRITICAL,
            matched_keyword="ak47",
        )
        mock_pipeline._test_mocks["matcher"].match_batch.return_value = [[hit]]
        mock_pipeline._test_mocks["nli"].classify_batch.return_value = [
            ClassificationResult(label=ClassifierLabel.ALLOWED, confidence=0.99),
        ]

        decisions = mock_pipeline.screen_batch(
            [Shipment(id="S1", description="ak47 parts")]
        )
        assert decisions[0].action == Action.BLOCK
