"""Drift monitoring for screening pipeline.

Tracks distribution shifts that signal model degradation:
- Confidence distribution drift (PSI)
- Per-taxonomy-group hit rate drift
- Label distribution drift
- Embedding centroid drift

Essential before any Phase 3 automation — source-country of commodity X
can shift silently and degrade model quality.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DriftReport:
    """Summary of drift detected in a comparison against a baseline."""

    psi_confidence: float                   # PSI of confidence distribution
    label_distribution_psi: float           # PSI of class label distribution
    hit_rate_deltas: dict[str, float] = field(default_factory=dict)  # per-group
    centroid_shift: float = 0.0             # L2 distance between centroids
    summary: str = ""

    @property
    def has_significant_drift(self) -> bool:
        """Rule of thumb: PSI >= 0.25 is significant drift."""
        return (
            self.psi_confidence >= 0.25
            or self.label_distribution_psi >= 0.25
            or self.centroid_shift >= 0.3
        )


def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index for a continuous distribution.

    PSI < 0.1: no significant change
    PSI 0.1 - 0.25: moderate change
    PSI >= 0.25: significant change requiring investigation

    Uses bin edges derived from the expected (baseline) distribution.
    """
    if expected.size == 0 or actual.size == 0:
        return 0.0

    bin_edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / max(expected.size, 1)
    actual_pct = actual_counts / max(actual.size, 1)

    expected_pct = np.clip(expected_pct, eps, 1.0)
    actual_pct = np.clip(actual_pct, eps, 1.0)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def categorical_psi(
    expected_counts: Counter,
    actual_counts: Counter,
    eps: float = 1e-6,
) -> float:
    """PSI for discrete categorical distributions."""
    categories = set(expected_counts) | set(actual_counts)
    expected_total = max(sum(expected_counts.values()), 1)
    actual_total = max(sum(actual_counts.values()), 1)

    total_psi = 0.0
    for cat in categories:
        e_pct = max(expected_counts.get(cat, 0) / expected_total, eps)
        a_pct = max(actual_counts.get(cat, 0) / actual_total, eps)
        total_psi += (a_pct - e_pct) * np.log(a_pct / e_pct)
    return float(total_psi)


def centroid_distance(
    baseline_embeddings: np.ndarray,
    current_embeddings: np.ndarray,
) -> float:
    """L2 distance between baseline and current mean embeddings.

    A simple proxy for embedding distribution drift. Larger shifts indicate
    new vocabulary or topic patterns.
    """
    if baseline_embeddings.size == 0 or current_embeddings.size == 0:
        return 0.0
    c_baseline = baseline_embeddings.mean(axis=0)
    c_current = current_embeddings.mean(axis=0)
    return float(np.linalg.norm(c_current - c_baseline))


@dataclass
class DriftBaseline:
    """Baseline distribution snapshot against which drift is measured."""

    confidences: np.ndarray
    labels: list[str]
    taxonomy_hit_rates: dict[str, float]  # group_id -> hit rate
    embeddings: np.ndarray | None = None

    @classmethod
    def from_decisions(
        cls,
        decisions: list,
        embeddings: np.ndarray | None = None,
    ) -> "DriftBaseline":
        """Build a baseline from a list of ScreeningDecision objects."""
        confidences = np.array(
            [d.classification.confidence for d in decisions if d.classification is not None]
        )
        labels = [
            d.classification.label.value
            for d in decisions
            if d.classification is not None
        ]

        # Hit rates per group
        group_counts: Counter = Counter()
        for d in decisions:
            for hit in d.taxonomy_hits:
                group_counts[hit.group_id] += 1
        total = max(len(decisions), 1)
        hit_rates = {g: c / total for g, c in group_counts.items()}

        return cls(
            confidences=confidences,
            labels=labels,
            taxonomy_hit_rates=hit_rates,
            embeddings=embeddings,
        )


class DriftDetector:
    """Detect drift between a baseline and current distribution."""

    def __init__(self, baseline: DriftBaseline) -> None:
        self._baseline = baseline

    def compare(
        self,
        current: DriftBaseline,
    ) -> DriftReport:
        # Confidence distribution PSI
        psi_conf = psi(self._baseline.confidences, current.confidences)

        # Label distribution PSI
        label_psi = categorical_psi(
            Counter(self._baseline.labels),
            Counter(current.labels),
        )

        # Per-group hit-rate deltas
        deltas: dict[str, float] = {}
        all_groups = (
            set(self._baseline.taxonomy_hit_rates)
            | set(current.taxonomy_hit_rates)
        )
        for g in all_groups:
            base = self._baseline.taxonomy_hit_rates.get(g, 0.0)
            curr = current.taxonomy_hit_rates.get(g, 0.0)
            deltas[g] = curr - base

        # Embedding centroid shift
        shift = 0.0
        if (
            self._baseline.embeddings is not None
            and current.embeddings is not None
        ):
            shift = centroid_distance(
                self._baseline.embeddings, current.embeddings
            )

        # Summary string
        summary_parts = [f"confidence_psi={psi_conf:.3f}", f"label_psi={label_psi:.3f}"]
        if shift:
            summary_parts.append(f"centroid_shift={shift:.3f}")
        large_deltas = {g: d for g, d in deltas.items() if abs(d) > 0.05}
        if large_deltas:
            summary_parts.append(f"hit_rate_changes={large_deltas}")

        return DriftReport(
            psi_confidence=psi_conf,
            label_distribution_psi=label_psi,
            hit_rate_deltas=deltas,
            centroid_shift=shift,
            summary="; ".join(summary_parts),
        )
