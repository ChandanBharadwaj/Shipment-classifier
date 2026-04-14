"""Tests for AIRiskEngine — HS chapter prediction, semantic dual-use,
and description-HS coherence. Embedding model is mocked."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from minerva.risk.ai_engine import (
    DEFAULT_DUAL_USE_CONCEPTS,
    AIRiskEngine,
    AISignals,
    DualUseSimilarity,
    HsChapterPrediction,
)
from minerva.risk.hs_chapters import HS_CHAPTERS


def _controlled_model(
    mapping: dict[str, np.ndarray] | None = None,
    dim: int = 8,
):
    """Build a mock SentenceTransformer where specific texts map to fixed vectors.

    Texts not in `mapping` receive deterministic pseudo-random embeddings.
    """
    mapping = mapping or {}
    rng = np.random.RandomState(0)

    def _encode(texts, **kwargs):
        out = []
        for t in texts:
            if t in mapping:
                v = mapping[t]
            else:
                # deterministic per text
                h = abs(hash(t)) % (2**31)
                r = np.random.RandomState(h)
                v = r.randn(dim).astype(np.float32)
            # normalize so cosine similarity is meaningful
            n = float(np.linalg.norm(v))
            out.append(v / n if n > 0 else v)
        return np.stack(out)

    mock = MagicMock()
    mock.encode = MagicMock(side_effect=_encode)
    return mock


class TestAIRiskEngineConstruction:
    def test_precomputes_hs_and_dual_use_embeddings(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(embedding_model=model)
        # One encode call for HS chapters, one for dual-use concepts
        assert model.encode.call_count == 2
        assert len(engine.hs_chapter_codes) == len(HS_CHAPTERS)
        assert len(engine.dual_use_concepts) == len(DEFAULT_DUAL_USE_CONCEPTS)

    def test_custom_dual_use_concepts(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(
            embedding_model=model,
            dual_use_concepts=["alpha concept", "beta concept"],
        )
        assert engine.dual_use_concepts == ["alpha concept", "beta concept"]

    def test_top_k_config(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(embedding_model=model, hs_top_k=5)
        assert engine.hs_top_k == 5

    def test_dual_use_threshold_config(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(
            embedding_model=model, dual_use_similarity_threshold=0.8
        )
        assert engine.dual_use_threshold == 0.8


class TestHsChapterPrediction:
    def test_identical_embedding_picks_that_chapter(self):
        # Align chapter 85 with a specific vector; description embedding is identical
        target_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        chapter_85_desc = HS_CHAPTERS["85"][1]
        model = _controlled_model(
            mapping={chapter_85_desc: target_vec},
            dim=4,
        )
        engine = AIRiskEngine(embedding_model=model)

        desc_embs = np.array([target_vec / np.linalg.norm(target_vec)])
        preds = engine.predict_hs_chapter_batch(desc_embs, top_k=3)
        assert len(preds) == 1
        assert preds[0][0].chapter == "85"
        assert preds[0][0].similarity == pytest.approx(1.0, abs=1e-5)

    def test_top_k_respected(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(embedding_model=model)
        desc_embs = engine.encode_descriptions(["test description"])
        preds = engine.predict_hs_chapter_batch(desc_embs, top_k=5)
        assert len(preds[0]) == 5
        # Descending similarity
        sims = [p.similarity for p in preds[0]]
        assert sims == sorted(sims, reverse=True)

    def test_empty_batch(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(embedding_model=model)
        result = engine.predict_hs_chapter_batch(np.empty((0, 8)))
        assert result == []

    def test_batch_multiple_shipments(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(embedding_model=model)
        descs = ["one", "two", "three"]
        embs = engine.encode_descriptions(descs)
        preds = engine.predict_hs_chapter_batch(embs, top_k=3)
        assert len(preds) == 3
        for p in preds:
            assert len(p) == 3


class TestDeclaredHsSimilarity:
    def test_returns_similarity_for_known_chapter(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(embedding_model=model)
        desc_emb = engine.encode_descriptions(["test"])[0]
        sim = engine.similarity_to_declared_chapter(desc_emb, "6109.10")
        assert sim is not None
        assert -1.0 <= sim <= 1.0

    def test_none_for_missing_hs(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(embedding_model=model)
        desc_emb = engine.encode_descriptions(["test"])[0]
        assert engine.similarity_to_declared_chapter(desc_emb, None) is None

    def test_none_for_malformed_hs(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(embedding_model=model)
        desc_emb = engine.encode_descriptions(["test"])[0]
        assert engine.similarity_to_declared_chapter(desc_emb, "abc") is None

    def test_none_for_unknown_chapter(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(embedding_model=model)
        desc_emb = engine.encode_descriptions(["test"])[0]
        # Chapter "77" is reserved/unused in HS → not in HS_CHAPTERS
        assert engine.similarity_to_declared_chapter(desc_emb, "7701") is None

    def test_batch(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(embedding_model=model)
        embs = engine.encode_descriptions(["a", "b", "c"])
        sims = engine.declared_hs_similarity_batch(
            embs, ["6109", None, "84"]
        )
        assert len(sims) == 3
        assert sims[0] is not None
        assert sims[1] is None
        assert sims[2] is not None


class TestDualUseSemanticSimilarity:
    def test_only_concepts_above_threshold_returned(self):
        # Target vector that matches exactly one dual-use concept
        target_concept = DEFAULT_DUAL_USE_CONCEPTS[0]
        target_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        target_vec = target_vec / np.linalg.norm(target_vec)
        model = _controlled_model(
            mapping={target_concept: target_vec},
            dim=4,
        )
        engine = AIRiskEngine(
            embedding_model=model,
            dual_use_similarity_threshold=0.9,
        )

        desc_embs = np.array([target_vec])
        matches = engine.assess_dual_use_batch(desc_embs)
        assert len(matches) == 1
        # At least the matching concept is returned
        concepts_matched = [m.concept for m in matches[0]]
        assert target_concept in concepts_matched

    def test_threshold_respected(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(
            embedding_model=model,
            dual_use_similarity_threshold=0.99,  # effectively impossible
        )
        embs = engine.encode_descriptions(["unrelated cotton garments"])
        matches = engine.assess_dual_use_batch(embs)
        assert matches[0] == []

    def test_sorted_descending(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(
            embedding_model=model,
            dual_use_similarity_threshold=-1.0,  # every concept passes
        )
        embs = engine.encode_descriptions(["test"])
        matches = engine.assess_dual_use_batch(embs)
        sims = [m.similarity for m in matches[0]]
        assert sims == sorted(sims, reverse=True)


class TestBuildAISignalsBatch:
    def test_full_batch_integration(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(embedding_model=model)
        descriptions = [
            "cotton t-shirts",
            "industrial control system with encryption",
        ]
        hs_codes = ["6109.10", "8537"]
        signals = engine.build_ai_signals_batch(descriptions, hs_codes)

        assert len(signals) == 2
        for s in signals:
            assert s.description_embedding is not None
            assert len(s.hs_predictions) == engine.hs_top_k
            assert s.declared_hs_similarity is not None
            assert s.description_hs_coherence is not None

    def test_none_for_missing_hs(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(embedding_model=model)
        signals = engine.build_ai_signals_batch(
            ["test"], hs_codes=[None]
        )
        assert signals[0].declared_hs_similarity is None

    def test_empty_batch(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(embedding_model=model)
        assert engine.build_ai_signals_batch([], []) == []

    def test_reuses_provided_embeddings(self):
        model = _controlled_model(dim=8)
        engine = AIRiskEngine(embedding_model=model)

        descriptions = ["test1", "test2"]
        provided = engine.encode_descriptions(descriptions)

        initial_calls = model.encode.call_count
        signals = engine.build_ai_signals_batch(
            descriptions, hs_codes=[None, None],
            description_embeddings=provided,
        )
        # No additional model.encode() calls when embeddings are provided
        assert model.encode.call_count == initial_calls
        assert len(signals) == 2


class TestAISignalsDataclass:
    def test_defaults(self):
        s = AISignals()
        assert s.description_embedding is None
        assert s.hs_predictions == []
        assert s.declared_hs_similarity is None
        assert s.dual_use_similarities == []
        assert s.description_hs_coherence is None
