"""Phase 1: Zero-shot NLI classification using DeBERTa cross-encoder."""

import numpy as np
from sentence_transformers import CrossEncoder

from minerva.classifier.base import BaseClassifier
from minerva.config import MinervaSettings
from minerva.schema import ClassificationResult, ClassifierLabel


class NLIClassifier(BaseClassifier):
    """Zero-shot Natural Language Inference classifier.

    Uses a cross-encoder NLI model to classify shipment descriptions
    against three hypotheses (allowed, restricted, needs_review).
    The entailment scores are softmax-normalized to produce a confidence.

    Phase 1 recommendation: cross-encoder/nli-deberta-v3-small
    """

    # NLI label indices: contradiction=0, neutral=1, entailment=2
    _ENTAILMENT_IDX = 2

    def __init__(self, settings: MinervaSettings) -> None:
        self._settings = settings
        self._model = CrossEncoder(settings.nli_model)

        # Hypotheses map to classifier labels
        self._hypotheses: list[tuple[ClassifierLabel, str]] = [
            (ClassifierLabel.ALLOWED, settings.nli_hypotheses.allowed),
            (ClassifierLabel.RESTRICTED, settings.nli_hypotheses.restricted),
            (ClassifierLabel.NEEDS_REVIEW, settings.nli_hypotheses.needs_review),
        ]

    def classify_batch(
        self, descriptions: list[str]
    ) -> list[ClassificationResult]:
        if not descriptions:
            return []

        # Build all (description, hypothesis) pairs
        pairs: list[list[str]] = []
        for desc in descriptions:
            for _, hypothesis in self._hypotheses:
                pairs.append([desc, hypothesis])

        # Predict all pairs at once: shape [N_desc * 3, 3] (3 NLI labels per pair)
        raw_scores = self._model.predict(pairs, batch_size=self._settings.batch_size)
        raw_scores = np.array(raw_scores)

        # Reshape to [N_desc, 3_hypotheses, 3_nli_labels]
        n_desc = len(descriptions)
        n_hyp = len(self._hypotheses)

        if raw_scores.ndim == 1:
            # Some cross-encoder models return a single score per pair
            # In this case, treat the score as the entailment score directly
            scores_reshaped = raw_scores.reshape(n_desc, n_hyp)
            entailment_scores = scores_reshaped
        else:
            scores_reshaped = raw_scores.reshape(n_desc, n_hyp, -1)
            # Extract entailment score for each hypothesis
            if scores_reshaped.shape[2] > self._ENTAILMENT_IDX:
                entailment_scores = scores_reshaped[:, :, self._ENTAILMENT_IDX]
            else:
                # Fallback: use the last column as the positive score
                entailment_scores = scores_reshaped[:, :, -1]

        # Softmax across hypotheses to get normalized confidence
        confidences = self._softmax(entailment_scores)

        results: list[ClassificationResult] = []
        for i in range(n_desc):
            best_idx = int(np.argmax(confidences[i]))
            label = self._hypotheses[best_idx][0]
            confidence = float(confidences[i][best_idx])
            results.append(
                ClassificationResult(label=label, confidence=confidence)
            )

        return results

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Row-wise softmax."""
        exp_x = np.exp(x - x.max(axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
