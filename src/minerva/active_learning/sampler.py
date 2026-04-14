"""Active learning sampler — uncertainty + diversity selection.

Selects the most informative items to send for reviewer labeling,
maximizing model improvement per labeling effort. Based on the
gATE methodology for customs inspection (Kim et al., IEEE 2022).

Algorithm:
    1. Compute per-item informativeness: margin uncertainty + entropy
    2. Greedy core-set diversity: pick highest-informativeness item, then
       iteratively pick items maximally distant from already-selected set

Inputs are numpy arrays so the sampler has no dependency on sentence-transformers.
"""

from __future__ import annotations

import numpy as np


def margin_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    """1 - (top prob - second prob). Higher = more uncertain.

    Args:
        probabilities: shape [n, num_classes]
    Returns:
        shape [n]
    """
    sorted_probs = np.sort(probabilities, axis=1)
    top = sorted_probs[:, -1]
    second = sorted_probs[:, -2] if probabilities.shape[1] >= 2 else np.zeros_like(top)
    return 1.0 - (top - second)


def entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy per row. Higher = more uncertain.

    Args:
        probabilities: shape [n, num_classes]
    Returns:
        shape [n]
    """
    p = np.clip(probabilities, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def informativeness_score(
    probabilities: np.ndarray,
    margin_weight: float = 0.5,
    entropy_weight: float = 0.5,
) -> np.ndarray:
    """Combined informativeness: weighted sum of margin and entropy.

    Normalizes each to [0, 1] before weighted sum.
    """
    margins = margin_uncertainty(probabilities)
    ents = entropy(probabilities)

    def _normalize(x: np.ndarray) -> np.ndarray:
        max_val = x.max() if x.size > 0 else 1.0
        return x / max_val if max_val > 0 else x

    return margin_weight * _normalize(margins) + entropy_weight * _normalize(ents)


def greedy_diverse_selection(
    embeddings: np.ndarray,
    informativeness: np.ndarray,
    budget: int,
) -> list[int]:
    """Greedy core-set selection: first pick the most informative item,
    then iteratively pick items that maximize distance to the selected set.

    Args:
        embeddings: [n, d] item embeddings.
        informativeness: [n] per-item informativeness score.
        budget: number of items to select.

    Returns:
        List of selected indices, length <= budget.
    """
    n = embeddings.shape[0]
    if n == 0 or budget <= 0:
        return []
    budget = min(budget, n)

    selected: list[int] = []
    first = int(np.argmax(informativeness))
    selected.append(first)

    if budget == 1:
        return selected

    # Precompute distances to selected set
    min_dist_to_selected = np.linalg.norm(
        embeddings - embeddings[first], axis=1
    )

    for _ in range(1, budget):
        # Score = distance (diversity) * informativeness (uncertainty)
        scores = min_dist_to_selected * informativeness
        # Exclude already-selected
        for s in selected:
            scores[s] = -np.inf

        next_idx = int(np.argmax(scores))
        if scores[next_idx] == -np.inf:
            break
        selected.append(next_idx)

        # Update min distances
        new_dists = np.linalg.norm(embeddings - embeddings[next_idx], axis=1)
        min_dist_to_selected = np.minimum(min_dist_to_selected, new_dists)

    return selected


def select_for_review(
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    budget: int,
    margin_weight: float = 0.5,
    entropy_weight: float = 0.5,
) -> list[int]:
    """Top-level entry point: combine uncertainty + diversity selection.

    Args:
        embeddings: [n, d] description embeddings.
        probabilities: [n, num_classes] classifier output probabilities.
        budget: number of items to select for review.
        margin_weight, entropy_weight: weights for combined informativeness.

    Returns:
        Indices of selected items.
    """
    if embeddings.shape[0] != probabilities.shape[0]:
        raise ValueError("embeddings and probabilities must have same length")

    info = informativeness_score(
        probabilities,
        margin_weight=margin_weight,
        entropy_weight=entropy_weight,
    )
    return greedy_diverse_selection(embeddings, info, budget)
