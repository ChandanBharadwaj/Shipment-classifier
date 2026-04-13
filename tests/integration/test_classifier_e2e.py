"""End-to-end classifier tests with real NLI model.

Requires downloading the cross-encoder model.
"""

import pytest

from minerva.classifier.nli import NLIClassifier
from minerva.config import MinervaSettings
from minerva.schema import ClassifierLabel

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def nli_classifier():
    settings = MinervaSettings()
    return NLIClassifier(settings)


class TestNLIClassifierEndToEnd:
    def test_benign_shipment_classified_allowed(self, nli_classifier):
        results = nli_classifier.classify_batch([
            "Organic cotton t-shirts bulk shipment",
            "Fresh fruit and vegetables for supermarket",
            "Laptop computers for office use",
        ])
        for r in results:
            assert r.label == ClassifierLabel.ALLOWED or r.confidence < 0.90
            assert 0.0 <= r.confidence <= 1.0

    def test_suspicious_shipment_not_allowed(self, nli_classifier):
        results = nli_classifier.classify_batch([
            "Military grade weapon systems for defense contractor",
        ])
        # Should not be classified as ALLOWED with high confidence
        assert not (
            results[0].label == ClassifierLabel.ALLOWED
            and results[0].confidence >= 0.90
        )

    def test_batch_size_consistency(self, nli_classifier):
        descriptions = [f"Item {i}" for i in range(10)]
        results = nli_classifier.classify_batch(descriptions)
        assert len(results) == 10
        for r in results:
            assert r.label in list(ClassifierLabel)
            assert 0.0 <= r.confidence <= 1.0
