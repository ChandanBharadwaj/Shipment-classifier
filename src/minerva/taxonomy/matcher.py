"""Batch taxonomy matching via cosine similarity and hard keyword detection."""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from minerva.config import MinervaSettings
from minerva.schema import TaxonomyHit
from minerva.taxonomy.embeddings import TaxonomyEmbeddingIndex


class TaxonomyMatcher:
    """Layer 1: Deterministic enforcement via semantic similarity and keyword matching.

    Any taxonomy hit is non-negotiable — it is never overridden downstream.
    """

    def __init__(
        self,
        index: TaxonomyEmbeddingIndex,
        model: SentenceTransformer,
        settings: MinervaSettings,
    ) -> None:
        self._index = index
        self._model = model
        self._settings = settings

        # Pre-compile lowercase keyword sets per group for fast matching
        self._keyword_sets: list[list[str]] = []
        for group in index.groups:
            self._keyword_sets.append(
                [kw.lower() for kw in group.hard_keywords]
            )

    def match_batch(self, descriptions: list[str]) -> list[list[TaxonomyHit]]:
        """Screen a batch of descriptions against the taxonomy.

        Args:
            descriptions: List of shipment description strings.

        Returns:
            List of TaxonomyHit lists, one per description.
            An empty inner list means no taxonomy match.
        """
        if not descriptions:
            return []

        results: list[list[TaxonomyHit]] = [[] for _ in descriptions]

        # Check hard keywords first (fast string matching)
        self._check_hard_keywords(descriptions, results)

        # Semantic similarity matching
        if self._index.num_phrases > 0 and self._index.phrase_embeddings is not None:
            self._check_semantic_similarity(descriptions, results)

        return results

    def _check_hard_keywords(
        self,
        descriptions: list[str],
        results: list[list[TaxonomyHit]],
    ) -> None:
        """Check descriptions against hard keyword lists (case-insensitive substring)."""
        lowered = [d.lower() for d in descriptions]

        for group_idx, keywords in enumerate(self._keyword_sets):
            if not keywords:
                continue

            group = self._index.groups[group_idx]
            for desc_idx, desc_lower in enumerate(lowered):
                for keyword in keywords:
                    if keyword in desc_lower:
                        hit = TaxonomyHit(
                            group_id=group.group_id,
                            group_name=group.name,
                            risk_level=group.risk_level,
                            matched_keyword=keyword,
                        )
                        results[desc_idx].append(hit)
                        break  # One hit per group per description is sufficient

    def _check_semantic_similarity(
        self,
        descriptions: list[str],
        results: list[list[TaxonomyHit]],
    ) -> None:
        """Batch cosine similarity between descriptions and taxonomy phrases."""
        desc_embeddings = self._model.encode(
            descriptions,
            batch_size=self._settings.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Single vectorized similarity computation: [batch, N_phrases]
        sim_matrix = cosine_similarity(desc_embeddings, self._index.phrase_embeddings)

        # Already-hit groups per description (from keyword matching)
        already_hit: list[set[str]] = [
            {h.group_id for h in hits} for hits in results
        ]

        # Check per group
        for group_idx, group in enumerate(self._index.groups):
            threshold = self._settings.get_similarity_threshold(
                group.risk_level.value
            )

            group_mask = self._index.get_group_phrase_mask(group_idx)
            if not group_mask.any():
                continue

            # Extract similarities for this group's phrases only
            group_sims = sim_matrix[:, group_mask]  # [batch, N_group_phrases]
            max_sims = group_sims.max(axis=1)  # [batch]
            best_phrase_indices = group_sims.argmax(axis=1)  # [batch]

            # Get the actual phrase indices within the group
            group_phrase_positions = np.where(group_mask)[0]

            for desc_idx in range(len(descriptions)):
                if max_sims[desc_idx] >= threshold:
                    if group.group_id in already_hit[desc_idx]:
                        continue  # Already hit by keyword

                    best_global_idx = group_phrase_positions[
                        best_phrase_indices[desc_idx]
                    ]
                    hit = TaxonomyHit(
                        group_id=group.group_id,
                        group_name=group.name,
                        risk_level=group.risk_level,
                        matched_phrase=self._index.phrase_texts[best_global_idx],
                        similarity_score=float(max_sims[desc_idx]),
                    )
                    results[desc_idx].append(hit)
