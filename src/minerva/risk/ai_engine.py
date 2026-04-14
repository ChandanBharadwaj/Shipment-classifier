"""AI risk engine — shared semantic intelligence across multiple dimensions.

Uses the existing sentence-transformer embedding model (already loaded
for taxonomy matching) to power three AI-driven capabilities:

    1. HS chapter prediction from free-text description
       → compare with declared HS code for misclassification detection
    2. Semantic dual-use concept similarity
       → catches paraphrases that keyword matching misses
    3. Description ↔ declared HS coherence
       → flags descriptions that don't match their declared HS chapter

No new models are loaded. Static reference corpora (HS chapter titles
and dual-use concept phrases) are embedded once at startup; per-shipment
scoring is a single cosine similarity matrix multiply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from minerva.risk.hs_chapters import HS_CHAPTERS, chapter_title, get_chapter


@dataclass(frozen=True)
class HsChapterPrediction:
    """A single predicted HS chapter for a shipment description."""

    chapter: str                  # e.g. "85"
    title: str                    # e.g. "Electrical machinery and equipment"
    similarity: float             # cosine similarity 0.0 - 1.0


@dataclass(frozen=True)
class DualUseSimilarity:
    """A single semantic match to a dual-use concept."""

    concept: str                  # the concept phrase that matched
    similarity: float             # cosine similarity 0.0 - 1.0


@dataclass
class AISignals:
    """AI-derived signals attached to a shipment's DimensionContext.

    All fields are optional — assessors must tolerate missing AI signals
    (e.g. if the AI engine is disabled or no HS code was declared).
    """

    description_embedding: np.ndarray | None = None
    hs_predictions: list[HsChapterPrediction] = field(default_factory=list)
    declared_hs_similarity: float | None = None  # sim between desc and declared chapter
    dual_use_similarities: list[DualUseSimilarity] = field(default_factory=list)
    description_hs_coherence: float | None = None  # highest sim among hs_predictions


# Default corpus of dual-use concept phrases. Richer and more specific than
# keyword matching — captures paraphrases via semantic similarity.
DEFAULT_DUAL_USE_CONCEPTS: list[str] = [
    "encryption or cryptographic equipment for secure communications",
    "high-performance computing cluster or GPU accelerator",
    "gas centrifuge or uranium enrichment equipment",
    "chemical precursor for weapons or narcotics synthesis",
    "surveillance, wiretapping, or interception equipment",
    "unmanned aerial vehicle, drone, or UAV component",
    "LIDAR, night vision, or thermal imaging system",
    "nuclear reactor component or fissile material handling",
    "biotechnology or gene synthesis equipment",
    "aerospace-grade or military-grade alloys or composites",
    "guidance, navigation, or inertial measurement unit",
    "radiation-hardened electronics or rad-hard components",
    "satellite communication terminal or VSAT equipment",
    "machine tool capable of multi-axis precision manufacturing",
    "high-voltage pulse power or directed energy systems",
]


class AIRiskEngine:
    """Centralizes AI-based signal generation shared by multiple dimensions.

    Pre-computes embeddings for the HS chapter corpus and dual-use concept
    corpus at construction time. At runtime, accepts pre-computed shipment
    description embeddings (computed once per batch by the pipeline) and
    produces per-shipment signals via cosine similarity.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        dual_use_concepts: list[str] | None = None,
        hs_top_k: int = 3,
        dual_use_similarity_threshold: float = 0.55,
    ) -> None:
        self._model = embedding_model
        self._hs_top_k = max(hs_top_k, 1)
        self._dual_use_threshold = dual_use_similarity_threshold

        # --- Pre-compute HS chapter embeddings ---
        self._hs_chapters: list[str] = list(HS_CHAPTERS.keys())
        hs_texts = [HS_CHAPTERS[c][1] for c in self._hs_chapters]
        self._hs_embeddings: np.ndarray = self._model.encode(
            hs_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # --- Pre-compute dual-use concept embeddings ---
        self._dual_use_concepts: list[str] = list(
            dual_use_concepts or DEFAULT_DUAL_USE_CONCEPTS
        )
        self._dual_use_embeddings: np.ndarray = self._model.encode(
            self._dual_use_concepts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    # ---------------- accessors ----------------

    @property
    def hs_chapter_codes(self) -> list[str]:
        return list(self._hs_chapters)

    @property
    def dual_use_concepts(self) -> list[str]:
        return list(self._dual_use_concepts)

    @property
    def hs_top_k(self) -> int:
        return self._hs_top_k

    @property
    def dual_use_threshold(self) -> float:
        return self._dual_use_threshold

    # ---------------- embedding ----------------

    def encode_descriptions(self, descriptions: list[str]) -> np.ndarray:
        """Embed a batch of descriptions — exposed for pipeline-level caching."""
        if not descriptions:
            return np.empty((0, self._hs_embeddings.shape[1]))
        return self._model.encode(
            descriptions,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    # ---------------- HS prediction ----------------

    def predict_hs_chapter_batch(
        self,
        description_embeddings: np.ndarray,
        top_k: int | None = None,
    ) -> list[list[HsChapterPrediction]]:
        """Return top-k predicted HS chapters for each description.

        Args:
            description_embeddings: shape [batch, embed_dim]
            top_k: number of predictions per description (default: self._hs_top_k)
        """
        if description_embeddings.size == 0:
            return []
        k = top_k or self._hs_top_k

        # Shape: [batch, n_chapters]
        sim = cosine_similarity(description_embeddings, self._hs_embeddings)
        # Top-k indices per row
        n_chapters = sim.shape[1]
        k = min(k, n_chapters)
        # argsort ascending; take last k and reverse for descending
        top_idx = np.argsort(sim, axis=1)[:, -k:][:, ::-1]

        results: list[list[HsChapterPrediction]] = []
        for row, indices in enumerate(top_idx):
            preds = [
                HsChapterPrediction(
                    chapter=self._hs_chapters[i],
                    title=HS_CHAPTERS[self._hs_chapters[i]][0],
                    similarity=float(sim[row, i]),
                )
                for i in indices
            ]
            results.append(preds)
        return results

    def similarity_to_declared_chapter(
        self,
        description_embedding: np.ndarray,
        hs_code: str | None,
    ) -> float | None:
        """Cosine similarity between a description and a declared HS chapter.

        Returns None if the declared HS code does not resolve to a known chapter.
        """
        chapter = get_chapter(hs_code)
        if chapter is None or chapter not in self._hs_chapters:
            return None
        idx = self._hs_chapters.index(chapter)
        # Reshape to (1, dim) for cosine_similarity
        desc_emb = description_embedding.reshape(1, -1)
        chap_emb = self._hs_embeddings[idx].reshape(1, -1)
        sim = cosine_similarity(desc_emb, chap_emb)[0, 0]
        return float(sim)

    def declared_hs_similarity_batch(
        self,
        description_embeddings: np.ndarray,
        hs_codes: list[str | None],
    ) -> list[float | None]:
        """Batch version — returns None per row if no known chapter."""
        if description_embeddings.size == 0:
            return []
        out: list[float | None] = []
        for i, hs in enumerate(hs_codes):
            out.append(
                self.similarity_to_declared_chapter(description_embeddings[i], hs)
            )
        return out

    # ---------------- Dual-use similarity ----------------

    def assess_dual_use_batch(
        self,
        description_embeddings: np.ndarray,
        threshold: float | None = None,
    ) -> list[list[DualUseSimilarity]]:
        """Return concepts exceeding the similarity threshold for each description."""
        if description_embeddings.size == 0:
            return []
        thr = threshold if threshold is not None else self._dual_use_threshold

        # [batch, n_concepts]
        sim = cosine_similarity(description_embeddings, self._dual_use_embeddings)
        results: list[list[DualUseSimilarity]] = []
        for row in sim:
            matches = [
                DualUseSimilarity(
                    concept=self._dual_use_concepts[i],
                    similarity=float(row[i]),
                )
                for i in range(len(self._dual_use_concepts))
                if row[i] >= thr
            ]
            # Highest similarity first
            matches.sort(key=lambda m: m.similarity, reverse=True)
            results.append(matches)
        return results

    # ---------------- Full batch assembly ----------------

    def build_ai_signals_batch(
        self,
        descriptions: list[str],
        hs_codes: list[str | None],
        description_embeddings: np.ndarray | None = None,
    ) -> list[AISignals]:
        """Compute all AI signals for a batch of shipments in one pass.

        If `description_embeddings` is provided, it is reused (preferred
        when the pipeline has already embedded descriptions for taxonomy
        matching). Otherwise they are computed here.
        """
        if not descriptions:
            return []

        if description_embeddings is None:
            description_embeddings = self.encode_descriptions(descriptions)

        hs_predictions_batch = self.predict_hs_chapter_batch(description_embeddings)
        declared_sims = self.declared_hs_similarity_batch(
            description_embeddings, hs_codes
        )
        dual_use_batch = self.assess_dual_use_batch(description_embeddings)

        signals: list[AISignals] = []
        for i, desc in enumerate(descriptions):
            preds = hs_predictions_batch[i]
            coherence = preds[0].similarity if preds else None
            signals.append(
                AISignals(
                    description_embedding=description_embeddings[i],
                    hs_predictions=preds,
                    declared_hs_similarity=declared_sims[i],
                    dual_use_similarities=dual_use_batch[i],
                    description_hs_coherence=coherence,
                )
            )
        return signals
