"""Phase 2+: Fine-tuned DistilBERT classifier for shipment screening."""

from pathlib import Path

import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from minerva.classifier.base import BaseClassifier
from minerva.config import MinervaSettings
from minerva.schema import ClassificationResult, ClassifierLabel

# Label mapping: model output index → ClassifierLabel
_LABEL_MAP: list[ClassifierLabel] = [
    ClassifierLabel.ALLOWED,
    ClassifierLabel.RESTRICTED,
    ClassifierLabel.NEEDS_REVIEW,
]


class DistilBERTClassifier(BaseClassifier):
    """Fine-tuned DistilBERT 3-class classifier.

    Trained on reviewer-labeled data from Phase 1. Produces:
    - Allowed: Shipment is consistent with legitimate cargo
    - Restricted: Shipment matches restricted item characteristics
    - Needs Review: Uncertain — route to human analyst

    Expected minimum dataset: 5,000–10,000 labeled examples per class.
    """

    def __init__(self, settings: MinervaSettings) -> None:
        self._settings = settings
        model_path = Path(settings.distilbert_model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned DistilBERT model not found at: {model_path}. "
                "Train a model first using the training pipeline, or switch "
                "to 'nli' classifier in settings."
            )

        self._tokenizer = DistilBertTokenizer.from_pretrained(str(model_path))
        self._model = DistilBertForSequenceClassification.from_pretrained(
            str(model_path)
        )
        self._model.eval()
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model.to(self._device)

    def classify_batch(
        self, descriptions: list[str]
    ) -> list[ClassificationResult]:
        if not descriptions:
            return []

        results: list[ClassificationResult] = []
        batch_size = self._settings.batch_size

        for start in range(0, len(descriptions), batch_size):
            batch = descriptions[start : start + batch_size]
            batch_results = self._classify_chunk(batch)
            results.extend(batch_results)

        return results

    def _classify_chunk(
        self, descriptions: list[str]
    ) -> list[ClassificationResult]:
        """Classify a single chunk of descriptions."""
        inputs = self._tokenizer(
            descriptions,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits.cpu().numpy()
        probabilities = self._softmax(logits)

        results: list[ClassificationResult] = []
        for i in range(len(descriptions)):
            best_idx = int(np.argmax(probabilities[i]))
            label = _LABEL_MAP[best_idx]
            confidence = float(probabilities[i][best_idx])
            results.append(
                ClassificationResult(label=label, confidence=confidence)
            )

        return results

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Row-wise softmax."""
        exp_x = np.exp(x - x.max(axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
