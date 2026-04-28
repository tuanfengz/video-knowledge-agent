"""Text embedder wrapping sentence-transformers.

Encodes segment descriptions into fixed-length dense vectors suitable
for FAISS similarity search.

Default model: all-MiniLM-L6-v2 (22 MB, fast, good zero-shot quality).
Can be swapped for a larger model (e.g., all-mpnet-base-v2) by passing
``model_name`` to the constructor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Embedder:
    """Sentence-transformer wrapper with lazy model loading."""

    model_name: str = "all-MiniLM-L6-v2"
    """HuggingFace model name. Downloaded once to ~/.cache/huggingface."""

    batch_size: int = 64
    """Number of sentences per encoding batch."""

    _model: object = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* and return a float32 array of shape (N, D).

        Parameters
        ----------
        texts:
            List of strings to encode. Must be non-empty.

        Returns
        -------
        np.ndarray
            Shape (len(texts), embedding_dim), dtype float32, L2-normalised.
        """
        if not texts:
            raise ValueError("texts must be non-empty")

        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # unit-norm → cosine sim == dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string. Returns shape (1, D) float32."""
        return self.encode([query])

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension without encoding anything."""
        return self._get_model().get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name)
        return self._model
