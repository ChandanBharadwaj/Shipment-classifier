"""Pre-compute and cache taxonomy phrase embeddings at startup."""

import numpy as np
from sentence_transformers import SentenceTransformer

from minerva.schema import RiskLevel, TaxonomyGroup


class TaxonomyEmbeddingIndex:
    """Pre-computed embedding index for all taxonomy semantic phrases.

    Built once at startup. Maps each embedding vector back to its
    source TaxonomyGroup for hit attribution.
    """

    def __init__(
        self,
        groups: list[TaxonomyGroup],
        model: SentenceTransformer,
    ) -> None:
        self._groups = groups
        self._model = model

        self.phrase_embeddings: np.ndarray | None = None
        self.phrase_texts: list[str] = []
        self.phrase_group_indices: list[int] = []

        self._build()

    def _build(self) -> None:
        """Encode all semantic phrases in a single batched call."""
        all_phrases: list[str] = []
        group_indices: list[int] = []

        for idx, group in enumerate(self._groups):
            for phrase in group.semantic_phrases:
                all_phrases.append(phrase)
                group_indices.append(idx)

        if not all_phrases:
            self.phrase_embeddings = np.empty((0, 0))
            return

        self.phrase_texts = all_phrases
        self.phrase_group_indices = group_indices
        self.phrase_embeddings = self._model.encode(
            all_phrases,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    @property
    def groups(self) -> list[TaxonomyGroup]:
        return self._groups

    def get_group_for_phrase(self, phrase_index: int) -> TaxonomyGroup:
        """Return the TaxonomyGroup that owns the phrase at the given index."""
        return self._groups[self.phrase_group_indices[phrase_index]]

    def get_group_phrase_mask(self, group_index: int) -> np.ndarray:
        """Return a boolean mask of phrases belonging to a specific group."""
        return np.array(
            [gi == group_index for gi in self.phrase_group_indices],
            dtype=bool,
        )

    def get_risk_level_for_group(self, group_index: int) -> RiskLevel:
        return self._groups[group_index].risk_level

    @property
    def num_phrases(self) -> int:
        return len(self.phrase_texts)

    @property
    def num_groups(self) -> int:
        return len(self._groups)
