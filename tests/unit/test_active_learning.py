"""Tests for active learning sampler."""

import numpy as np
import pytest

from minerva.active_learning.sampler import (
    entropy,
    greedy_diverse_selection,
    informativeness_score,
    margin_uncertainty,
    select_for_review,
)


class TestMarginUncertainty:
    def test_confident_predictions_low_uncertainty(self):
        # Very confident: [0.95, 0.03, 0.02] → margin 0.92, uncertainty 0.08
        probs = np.array([[0.95, 0.03, 0.02]])
        result = margin_uncertainty(probs)
        assert result[0] < 0.1

    def test_uncertain_predictions_high_uncertainty(self):
        # Tied: [0.4, 0.4, 0.2] → margin 0, uncertainty 1.0
        probs = np.array([[0.4, 0.4, 0.2]])
        result = margin_uncertainty(probs)
        assert result[0] >= 0.95

    def test_uniform_predictions(self):
        probs = np.array([[0.33, 0.33, 0.34]])
        result = margin_uncertainty(probs)
        assert result[0] > 0.9

    def test_batch(self):
        probs = np.array([
            [0.95, 0.03, 0.02],
            [0.4, 0.4, 0.2],
            [0.6, 0.3, 0.1],
        ])
        result = margin_uncertainty(probs)
        assert result.shape == (3,)
        assert result[0] < result[2] < result[1]


class TestEntropy:
    def test_confident_low_entropy(self):
        probs = np.array([[0.95, 0.03, 0.02]])
        result = entropy(probs)
        assert result[0] < 0.3

    def test_uniform_max_entropy(self):
        probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
        result = entropy(probs)
        # Max entropy for 3 classes is log(3) ≈ 1.0986
        assert abs(result[0] - np.log(3)) < 0.01


class TestInformativenessScore:
    def test_weighted_combination(self):
        probs = np.array([
            [0.95, 0.03, 0.02],  # confident
            [0.4, 0.4, 0.2],     # uncertain
        ])
        scores = informativeness_score(probs)
        assert scores[0] < scores[1]


class TestGreedyDiverseSelection:
    def test_empty_inputs(self):
        result = greedy_diverse_selection(
            np.empty((0, 4)), np.empty((0,)), budget=5
        )
        assert result == []

    def test_budget_zero(self):
        emb = np.eye(5)
        info = np.ones(5)
        result = greedy_diverse_selection(emb, info, budget=0)
        assert result == []

    def test_picks_most_informative_first(self):
        emb = np.eye(5)
        info = np.array([0.1, 0.5, 0.9, 0.3, 0.2])
        result = greedy_diverse_selection(emb, info, budget=1)
        assert result == [2]

    def test_diversity_spreads_picks(self):
        # 4 points: two clusters of two
        emb = np.array([
            [0.0, 0.0],
            [0.1, 0.0],   # close to item 0
            [10.0, 10.0],
            [10.1, 10.0],  # close to item 2
        ])
        info = np.array([1.0, 1.0, 1.0, 1.0])  # all equally informative
        result = greedy_diverse_selection(emb, info, budget=2)
        # Should pick from both clusters
        assert (0 in result or 1 in result) and (2 in result or 3 in result)

    def test_budget_exceeds_n(self):
        emb = np.eye(3)
        info = np.ones(3)
        result = greedy_diverse_selection(emb, info, budget=10)
        assert len(result) == 3


class TestSelectForReview:
    def test_end_to_end(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(20, 8)
        # Mix of confident and uncertain predictions
        probs = np.array(
            [[0.9, 0.05, 0.05]] * 5
            + [[0.4, 0.4, 0.2]] * 10   # uncertain
            + [[0.95, 0.03, 0.02]] * 5
        )
        indices = select_for_review(emb, probs, budget=5)
        assert len(indices) == 5
        # Most selected should be from the uncertain middle (indices 5-14)
        uncertain_count = sum(1 for i in indices if 5 <= i < 15)
        assert uncertain_count >= 3

    def test_mismatched_sizes_raises(self):
        emb = np.eye(3)
        probs = np.ones((5, 2))
        with pytest.raises(ValueError):
            select_for_review(emb, probs, budget=2)
