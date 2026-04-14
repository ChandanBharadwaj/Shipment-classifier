"""Tests for AI-enhanced assessors: HsCode, DualUse, DataQuality.

Uses AISignals fixtures directly (no real embedding model needed)."""

import numpy as np
import pytest

from minerva.risk.ai_engine import AISignals, DualUseSimilarity, HsChapterPrediction
from minerva.risk.dimensions import (
    DataQualityAssessor,
    DimensionContext,
    DualUseAssessor,
    HsCodeAssessor,
    HsCodeConfig,
)
from minerva.schema import RiskDimension, Severity, Shipment


def _ctx(ship, ai: AISignals | None = None):
    return DimensionContext(shipment=ship, ai_signals=ai)


# -------------- HsCodeAssessor with AI signals ---------------

class TestHsCodeAIEnhancement:
    def test_declared_chapter_not_in_top_k(self):
        # Declared HS is chapter 61 (T-shirts); AI predicts 85/84/90 instead
        ai = AISignals(
            hs_predictions=[
                HsChapterPrediction(chapter="85", title="Electrical machinery", similarity=0.81),
                HsChapterPrediction(chapter="84", title="Machinery", similarity=0.72),
                HsChapterPrediction(chapter="90", title="Optical/medical instruments", similarity=0.64),
            ],
            declared_hs_similarity=0.09,
        )
        ship = Shipment(id="S", description="industrial control module", hs_code="6109.10")
        result = HsCodeAssessor().assess(_ctx(ship, ai))
        sig_names = {s.name for s in result.signals}
        assert "hs_declared_not_in_top_k" in sig_names
        # Evidence should mention AI top predictions
        ev = next(s.evidence for s in result.signals if s.name == "hs_declared_not_in_top_k")
        assert "85" in ev and "Electrical" in ev

    def test_declared_in_top_k_no_mismatch_signal(self):
        ai = AISignals(
            hs_predictions=[
                HsChapterPrediction(chapter="61", title="Apparel knitted", similarity=0.78),
                HsChapterPrediction(chapter="62", title="Apparel woven", similarity=0.72),
                HsChapterPrediction(chapter="63", title="Textiles", similarity=0.60),
            ],
            declared_hs_similarity=0.78,
        )
        ship = Shipment(id="S", description="knitted cotton t-shirts", hs_code="6109.10")
        result = HsCodeAssessor().assess(_ctx(ship, ai))
        sig_names = {s.name for s in result.signals}
        assert "hs_declared_not_in_top_k" not in sig_names

    def test_low_declared_similarity_signal(self):
        # Declared IS in top-k but absolute similarity is low
        ai = AISignals(
            hs_predictions=[
                HsChapterPrediction(chapter="61", title="Apparel knitted", similarity=0.28),
                HsChapterPrediction(chapter="62", title="Apparel woven", similarity=0.22),
                HsChapterPrediction(chapter="63", title="Textiles", similarity=0.18),
            ],
            declared_hs_similarity=0.20,
        )
        ship = Shipment(id="S", description="vague product", hs_code="6109.10")
        result = HsCodeAssessor(HsCodeConfig(low_declared_similarity=0.30)).assess(
            _ctx(ship, ai)
        )
        assert any(s.name == "hs_low_declared_similarity" for s in result.signals)

    def test_no_ai_signals_falls_back_to_rules(self):
        ship = Shipment(id="S", description="t", hs_code="9301.10")
        result = HsCodeAssessor(
            HsCodeConfig(sensitive_prefixes=["93"])
        ).assess(_ctx(ship, None))
        # Sensitive prefix signal still fires
        assert any("sensitive_hs_prefix" in s.name for s in result.signals)

    def test_missing_hs_no_ai_signal(self):
        ai = AISignals(
            hs_predictions=[
                HsChapterPrediction(chapter="85", title="X", similarity=0.8),
            ]
        )
        ship = Shipment(id="S", description="t", hs_code=None)
        result = HsCodeAssessor().assess(_ctx(ship, ai))
        sig_names = {s.name for s in result.signals}
        # missing_hs_code fires but no AI-based mismatch signal
        assert "missing_hs_code" in sig_names
        assert "hs_declared_not_in_top_k" not in sig_names

    def test_dimension_correct(self):
        result = HsCodeAssessor().assess(_ctx(Shipment(id="S", description="t"), None))
        assert result.dimension == RiskDimension.HS_CODE


# -------------- DualUseAssessor with AI semantic similarity ---------------

class TestDualUseAIEnhancement:
    def test_ai_dual_use_match_surfaced(self):
        ai = AISignals(
            dual_use_similarities=[
                DualUseSimilarity(
                    concept="encryption or cryptographic equipment for secure communications",
                    similarity=0.78,
                ),
            ]
        )
        ship = Shipment(
            id="S",
            description="secure communications controller",  # no keyword match
            hs_code="8537.10",
        )
        result = DualUseAssessor().assess(_ctx(ship, ai))
        sig_names = {s.name for s in result.signals}
        assert "ai_dual_use_match" in sig_names
        assert result.severity.rank >= Severity.MEDIUM.rank

    def test_multiple_ai_matches_shown_in_evidence(self):
        ai = AISignals(
            dual_use_similarities=[
                DualUseSimilarity(concept="concept A", similarity=0.80),
                DualUseSimilarity(concept="concept B", similarity=0.70),
                DualUseSimilarity(concept="concept C", similarity=0.60),
            ]
        )
        ship = Shipment(id="S", description="t")
        result = DualUseAssessor().assess(_ctx(ship, ai))
        ev = next(s.evidence for s in result.signals if s.name == "ai_dual_use_match")
        assert "concept A" in ev

    def test_ai_signal_complements_keyword(self):
        """Both keyword hit and AI match should fire independently."""
        ai = AISignals(
            dual_use_similarities=[
                DualUseSimilarity(concept="test concept", similarity=0.72),
            ]
        )
        ship = Shipment(
            id="S",
            description="encryption module with GPU cluster",
            hs_code="8471.30",
        )
        result = DualUseAssessor().assess(_ctx(ship, ai))
        names = {s.name for s in result.signals}
        assert "dual_use_keyword" in names
        assert "ai_dual_use_match" in names

    def test_no_ai_match_no_ai_signal(self):
        ai = AISignals(dual_use_similarities=[])
        ship = Shipment(id="S", description="cotton garments")
        result = DualUseAssessor().assess(_ctx(ship, ai))
        assert not any(s.name == "ai_dual_use_match" for s in result.signals)

    def test_no_ai_signals_at_all(self):
        ship = Shipment(id="S", description="cotton garments")
        result = DualUseAssessor().assess(_ctx(ship, None))
        assert not any(s.name == "ai_dual_use_match" for s in result.signals)


# -------------- DataQualityAssessor with AI coherence ---------------

class TestDataQualityAIEnhancement:
    def test_low_coherence_surfaces_signal(self):
        ai = AISignals(declared_hs_similarity=0.10)  # very low
        ship = Shipment(
            id="S",
            description="A full-length complete description of an item",
            origin_country="US",
            destination_country="CA",
            consignee="A",
            shipper="B",
            declared_value=100.0,
            hs_code="6109.10",
        )
        result = DataQualityAssessor().assess(_ctx(ship, ai))
        names = {s.name for s in result.signals}
        assert "description_hs_incoherent" in names

    def test_high_coherence_no_signal(self):
        ai = AISignals(declared_hs_similarity=0.78)
        ship = Shipment(
            id="S",
            description="knitted cotton t-shirts for retail",
            origin_country="US",
            destination_country="CA",
            consignee="A",
            shipper="B",
            declared_value=100.0,
            hs_code="6109.10",
        )
        result = DataQualityAssessor().assess(_ctx(ship, ai))
        names = {s.name for s in result.signals}
        assert "description_hs_incoherent" not in names

    def test_no_hs_declared_no_coherence_signal(self):
        ai = AISignals(declared_hs_similarity=None)
        ship = Shipment(
            id="S",
            description="complete description",
            origin_country="US",
            destination_country="CA",
            consignee="A",
            shipper="B",
            declared_value=100.0,
        )
        result = DataQualityAssessor().assess(_ctx(ship, ai))
        assert not any(s.name == "description_hs_incoherent" for s in result.signals)

    def test_no_ai_signals_no_coherence_signal(self):
        ship = Shipment(
            id="S",
            description="complete description",
            origin_country="US",
            destination_country="CA",
            consignee="A",
            shipper="B",
            declared_value=100.0,
            hs_code="6109",
        )
        result = DataQualityAssessor().assess(_ctx(ship, None))
        assert not any(s.name == "description_hs_incoherent" for s in result.signals)
