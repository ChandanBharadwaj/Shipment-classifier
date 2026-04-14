"""Drift monitoring endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body
from pydantic import BaseModel

from minerva.monitoring.drift import (
    DriftBaseline,
    DriftDetector,
    categorical_psi,
    psi,
)

router = APIRouter()


class DriftComparisonRequest(BaseModel):
    """Inputs for a drift comparison.

    Both baseline and current distributions should be produced
    from DriftBaseline.from_decisions() on the client side.
    """

    baseline_confidences: list[float]
    baseline_labels: list[str]
    baseline_hit_rates: dict[str, float]

    current_confidences: list[float]
    current_labels: list[str]
    current_hit_rates: dict[str, float]


@router.post("/drift")
def compute_drift(request: DriftComparisonRequest) -> dict[str, Any]:
    """Compare a baseline vs current distribution and return drift metrics."""
    import numpy as np

    baseline = DriftBaseline(
        confidences=np.array(request.baseline_confidences),
        labels=request.baseline_labels,
        taxonomy_hit_rates=request.baseline_hit_rates,
    )
    current = DriftBaseline(
        confidences=np.array(request.current_confidences),
        labels=request.current_labels,
        taxonomy_hit_rates=request.current_hit_rates,
    )

    detector = DriftDetector(baseline)
    report = detector.compare(current)

    return {
        "psi_confidence": report.psi_confidence,
        "label_distribution_psi": report.label_distribution_psi,
        "hit_rate_deltas": report.hit_rate_deltas,
        "centroid_shift": report.centroid_shift,
        "has_significant_drift": report.has_significant_drift,
        "summary": report.summary,
    }
