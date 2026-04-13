"""Abstract base class for shipment classifiers."""

from abc import ABC, abstractmethod

from minerva.schema import ClassificationResult


class BaseClassifier(ABC):
    """Interface for Layer 2 classifiers.

    Supports swapping between NLI (Phase 1) and fine-tuned DistilBERT (Phase 2+)
    without changing pipeline code.
    """

    @abstractmethod
    def classify_batch(
        self, descriptions: list[str]
    ) -> list[ClassificationResult]:
        """Classify a batch of shipment descriptions.

        Args:
            descriptions: List of shipment description strings.

        Returns:
            List of ClassificationResult with label and confidence.
        """
        ...

    def classify(self, description: str) -> ClassificationResult:
        """Classify a single shipment description."""
        return self.classify_batch([description])[0]
