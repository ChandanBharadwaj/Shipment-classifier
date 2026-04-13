"""Tests for NLI classifier (Layer 2, Phase 1)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from minerva.classifier.nli import NLIClassifier
from minerva.config import MinervaSettings
from minerva.schema import ClassifierLabel


class TestNLIClassifier:
    @pytest.fixture
    def settings(self):
        return MinervaSettings()

    def test_classify_batch_produces_results(self, settings):
        """Mock CrossEncoder and verify classify_batch output structure."""
        with patch("minerva.classifier.nli.CrossEncoder") as MockCE:
            mock_model = MagicMock()
            # 2 descriptions × 3 hypotheses = 6 pairs
            # Each pair gets 3 NLI scores (contradiction, neutral, entailment)
            scores = np.array([
                # Desc 1, hyp 1 (allowed): high entailment
                [-1.0, 0.0, 2.0],
                # Desc 1, hyp 2 (restricted): low entailment
                [1.0, 0.5, -1.0],
                # Desc 1, hyp 3 (needs_review): low entailment
                [1.0, 0.0, -0.5],
                # Desc 2, hyp 1 (allowed): low entailment
                [1.0, 0.0, -1.0],
                # Desc 2, hyp 2 (restricted): high entailment
                [-1.0, 0.0, 3.0],
                # Desc 2, hyp 3 (needs_review): low entailment
                [0.5, 0.0, -0.5],
            ], dtype=np.float32)
            mock_model.predict.return_value = scores
            MockCE.return_value = mock_model

            classifier = NLIClassifier(settings)
            results = classifier.classify_batch(["safe item", "weapon system"])

            assert len(results) == 2
            assert results[0].label == ClassifierLabel.ALLOWED
            assert results[1].label == ClassifierLabel.RESTRICTED
            assert 0.0 <= results[0].confidence <= 1.0
            assert 0.0 <= results[1].confidence <= 1.0

    def test_classify_single(self, settings):
        """Test the single-item convenience method."""
        with patch("minerva.classifier.nli.CrossEncoder") as MockCE:
            mock_model = MagicMock()
            scores = np.array([
                [-1.0, 0.0, 2.0],
                [1.0, 0.5, -1.0],
                [1.0, 0.0, -0.5],
            ], dtype=np.float32)
            mock_model.predict.return_value = scores
            MockCE.return_value = mock_model

            classifier = NLIClassifier(settings)
            result = classifier.classify("laptop computers")
            assert result.label == ClassifierLabel.ALLOWED

    def test_empty_batch(self, settings):
        with patch("minerva.classifier.nli.CrossEncoder") as MockCE:
            MockCE.return_value = MagicMock()
            classifier = NLIClassifier(settings)
            results = classifier.classify_batch([])
            assert results == []

    def test_pairs_construction(self, settings):
        """Verify that the correct (description, hypothesis) pairs are built."""
        with patch("minerva.classifier.nli.CrossEncoder") as MockCE:
            mock_model = MagicMock()
            # 1 description × 3 hypotheses = 3 pairs
            mock_model.predict.return_value = np.zeros((3, 3), dtype=np.float32)
            MockCE.return_value = mock_model

            classifier = NLIClassifier(settings)
            classifier.classify_batch(["test description"])

            call_args = mock_model.predict.call_args
            pairs = call_args[0][0]
            assert len(pairs) == 3
            assert all(p[0] == "test description" for p in pairs)
            # Each pair has a different hypothesis
            hypotheses = [p[1] for p in pairs]
            assert len(set(hypotheses)) == 3

    def test_softmax_normalization(self):
        """Verify softmax produces valid probability distribution."""
        x = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        result = NLIClassifier._softmax(x)
        # Each row sums to 1
        np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0], atol=1e-6)
        # All values positive
        assert (result > 0).all()

    def test_single_score_per_pair(self, settings):
        """Handle cross-encoder models that return a single score per pair."""
        with patch("minerva.classifier.nli.CrossEncoder") as MockCE:
            mock_model = MagicMock()
            # 1D array: one score per pair (3 pairs for 1 description)
            mock_model.predict.return_value = np.array([2.0, -1.0, 0.5], dtype=np.float32)
            MockCE.return_value = mock_model

            classifier = NLIClassifier(settings)
            results = classifier.classify_batch(["test item"])
            assert len(results) == 1
            assert results[0].label == ClassifierLabel.ALLOWED  # Highest score
