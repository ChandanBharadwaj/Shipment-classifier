"""Tests for drift monitoring."""

from collections import Counter

import numpy as np
import pytest

from minerva.monitoring.drift import (
    DriftBaseline,
    DriftDetector,
    categorical_psi,
    centroid_distance,
    psi,
)


class TestPSI:
    def test_identical_distributions_psi_zero(self):
        rng = np.random.RandomState(42)
        data = rng.randn(1000)
        result = psi(data, data.copy())
        assert result < 0.01

    def test_shifted_distribution_psi_high(self):
        rng = np.random.RandomState(42)
        baseline = rng.randn(1000)
        shifted = rng.randn(1000) + 2.0
        result = psi(baseline, shifted)
        assert result > 0.25

    def test_empty_distributions(self):
        result = psi(np.array([]), np.array([]))
        assert result == 0.0


class TestCategoricalPSI:
    def test_identical_counts_zero(self):
        c1 = Counter({"a": 100, "b": 200})
        c2 = Counter({"a": 100, "b": 200})
        assert categorical_psi(c1, c2) < 0.01

    def test_different_distributions(self):
        c1 = Counter({"a": 100, "b": 100})
        c2 = Counter({"a": 10, "b": 190})
        result = categorical_psi(c1, c2)
        assert result > 0.25


class TestCentroidDistance:
    def test_same_embeddings_zero_distance(self):
        emb = np.eye(3)
        assert centroid_distance(emb, emb) == 0.0

    def test_shifted_embeddings(self):
        base = np.zeros((10, 3))
        shifted = np.ones((10, 3))
        distance = centroid_distance(base, shifted)
        assert distance == pytest.approx(np.sqrt(3), abs=0.01)


class TestDriftBaseline:
    def test_from_decisions(self):
        from minerva.schema import (
            Action,
            ClassificationResult,
            ClassifierLabel,
            RiskLevel,
            ScreeningDecision,
            TaxonomyHit,
        )

        decisions = [
            ScreeningDecision(
                shipment_id=f"S{i}",
                action=Action.APPROVE,
                classification=ClassificationResult(
                    label=ClassifierLabel.ALLOWED,
                    confidence=0.9 + 0.01 * i,
                ),
                reason="",
            )
            for i in range(5)
        ]
        # Add one with a taxonomy hit
        decisions.append(
            ScreeningDecision(
                shipment_id="S99",
                action=Action.BLOCK,
                taxonomy_hits=[
                    TaxonomyHit(
                        group_id="mil",
                        group_name="Military",
                        risk_level=RiskLevel.CRITICAL,
                        matched_keyword="ak47",
                    )
                ],
                classification=ClassificationResult(
                    label=ClassifierLabel.RESTRICTED, confidence=0.95
                ),
                reason="",
            )
        )

        baseline = DriftBaseline.from_decisions(decisions)
        assert len(baseline.confidences) == 6
        assert baseline.labels.count("allowed") == 5
        assert baseline.labels.count("restricted") == 1
        assert baseline.taxonomy_hit_rates.get("mil", 0) > 0


class TestDriftDetector:
    def _make_baseline(self, confidences, labels, hit_rates=None):
        return DriftBaseline(
            confidences=np.array(confidences),
            labels=labels,
            taxonomy_hit_rates=hit_rates or {},
        )

    def test_no_drift_between_identical_distributions(self):
        rng = np.random.RandomState(42)
        confs = rng.uniform(0.7, 0.95, 500)
        labels = ["allowed"] * 400 + ["restricted"] * 100

        baseline = self._make_baseline(confs, labels)
        current = self._make_baseline(confs.copy(), labels.copy())

        report = DriftDetector(baseline).compare(current)
        assert not report.has_significant_drift

    def test_confidence_drift_detected(self):
        rng = np.random.RandomState(42)
        baseline_confs = rng.uniform(0.85, 0.95, 500)
        current_confs = rng.uniform(0.50, 0.75, 500)  # big shift

        baseline = self._make_baseline(baseline_confs, ["allowed"] * 500)
        current = self._make_baseline(current_confs, ["allowed"] * 500)

        report = DriftDetector(baseline).compare(current)
        assert report.has_significant_drift
        assert report.psi_confidence > 0.25

    def test_label_drift_detected(self):
        baseline = self._make_baseline(
            np.full(500, 0.9),
            ["allowed"] * 450 + ["restricted"] * 50,
        )
        current = self._make_baseline(
            np.full(500, 0.9),
            ["allowed"] * 200 + ["restricted"] * 300,
        )
        report = DriftDetector(baseline).compare(current)
        assert report.label_distribution_psi > 0.25

    def test_hit_rate_deltas(self):
        baseline = self._make_baseline(
            np.full(100, 0.9),
            ["allowed"] * 100,
            hit_rates={"group_a": 0.05, "group_b": 0.10},
        )
        current = self._make_baseline(
            np.full(100, 0.9),
            ["allowed"] * 100,
            hit_rates={"group_a": 0.20, "group_b": 0.10},
        )
        report = DriftDetector(baseline).compare(current)
        assert report.hit_rate_deltas["group_a"] == pytest.approx(0.15)
        assert report.hit_rate_deltas["group_b"] == pytest.approx(0.0)
