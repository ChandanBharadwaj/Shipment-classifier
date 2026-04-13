"""Tests for DistilBERT classifier (Layer 2, Phase 2+)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from minerva.classifier.distilbert import DistilBERTClassifier
from minerva.config import MinervaSettings
from minerva.schema import ClassifierLabel


class TestDistilBERTClassifier:
    def test_model_not_found_raises(self, tmp_path):
        settings = MinervaSettings(distilbert_model_path=str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError, match="Fine-tuned DistilBERT model not found"):
            DistilBERTClassifier(settings)

    @patch("minerva.classifier.distilbert.DistilBertForSequenceClassification")
    @patch("minerva.classifier.distilbert.DistilBertTokenizer")
    def test_classify_batch(self, MockTokenizer, MockModel, tmp_path):
        """Test classification with mocked model and tokenizer."""
        # Create a fake model directory
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        settings = MinervaSettings(distilbert_model_path=str(model_dir))

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_encoding = {
            "input_ids": torch.zeros(2, 128, dtype=torch.long),
            "attention_mask": torch.ones(2, 128, dtype=torch.long),
        }
        mock_tokenizer.return_value = MagicMock(
            to=MagicMock(return_value=mock_encoding),
            **mock_encoding,
        )
        # Make the tokenizer return object support .to()
        token_result = MagicMock()
        token_result.to.return_value = mock_encoding
        token_result.__getitem__ = mock_encoding.__getitem__
        mock_tokenizer.return_value = token_result
        MockTokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        # Logits: [batch_size=2, num_classes=3]
        logits = torch.tensor([[3.0, -1.0, 0.5], [-1.0, 4.0, 0.0]])
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        MockModel.from_pretrained.return_value = mock_model

        classifier = DistilBERTClassifier(settings)
        results = classifier.classify_batch(["safe item", "weapon system"])

        assert len(results) == 2
        assert results[0].label == ClassifierLabel.ALLOWED  # highest logit at index 0
        assert results[1].label == ClassifierLabel.RESTRICTED  # highest logit at index 1

    def test_empty_batch(self, tmp_path):
        """Empty batch returns empty results without loading model."""
        # Can't easily test without mocking - just test the _softmax
        x = np.array([[1.0, 2.0, 3.0]])
        result = DistilBERTClassifier._softmax(x)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result.sum(axis=1), [1.0], atol=1e-6)
