"""FAISS-backed segment index: build, persist, load, and search.

Workflow
--------
1. Process all videos with ``segment_video()`` to get ``VideoSegment`` objects.
2. Call ``SegmentIndex.build(segments, embedder)`` — encodes all descriptions
   and adds them to an inner-product FAISS index (cosine similarity, since
   embeddings are L2-normalised by the Embedder).
3. Save with ``index.save(directory)``; reload later with
   ``SegmentIndex.load(directory)``.
4. Query with ``index.search(query, embedder, top_k=10)``.

On-disk layout (``data/index/``)
---------------------------------
  vectors.faiss   — raw FAISS IndexFlatIP
  segments.json   — ordered list of VideoSegment dicts (parallel to vectors)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from video_knowledge_agent.retrieval.embedder import Embedder
    from video_knowledge_agent.retrieval.segmenter import VideoSegment


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Single retrieval hit returned by ``SegmentIndex.search``."""

    rank: int
    score: float
    video_id: str
    video_path: str
    start_s: float
    end_s: float
    description: str
    labels: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": round(float(self.score), 4),
            "video_id": self.video_id,
            "video_path": self.video_path,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "description": self.description,
            "labels": self.labels,
        }


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

_VECTORS_FILE = "vectors.faiss"
_SEGMENTS_FILE = "segments.json"


def _tokenize(text: str) -> set[str]:
    return {token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(token) > 1}


def _label_match_score(query: str, labels: dict[str, int]) -> float:
    if not labels:
        return 0.0

    query_text = query.lower()
    query_tokens = _tokenize(query)
    matches = 0
    for label in labels:
        label_text = label.lower()
        label_tokens = _tokenize(label)
        if label_text in query_text or (label_tokens and label_tokens <= query_tokens):
            matches += 1
    return matches / max(1, min(len(labels), 4))


def _lexical_overlap_score(query: str, description: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    description_tokens = _tokenize(description)
    return len(query_tokens & description_tokens) / len(query_tokens)


def _rerank_score(query: str, score: float, description: str, labels: dict[str, int]) -> float:
    lexical = _lexical_overlap_score(query, description)
    label = _label_match_score(query, labels)
    return float(score) + 0.18 * lexical + 0.25 * label


class SegmentIndex:
    """Flat inner-product FAISS index over segment description embeddings."""

    def __init__(self, faiss_index, segments: list["VideoSegment"]) -> None:
        self._index = faiss_index
        self._segments = segments  # parallel to FAISS vectors

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        segments: list["VideoSegment"],
        embedder: "Embedder",
    ) -> "SegmentIndex":
        """Encode all segment descriptions and build the FAISS index.

        Parameters
        ----------
        segments:
            List of ``VideoSegment`` objects (from one or many videos).
        embedder:
            Embedder used to encode descriptions. Must be the same instance
            (or same model) used at query time.

        Returns
        -------
        SegmentIndex
            Ready to search.
        """
        import faiss  # type: ignore

        if not segments:
            raise ValueError("Cannot build an index with zero segments.")

        descriptions = [s.description for s in segments]
        print(f"Encoding {len(descriptions)} segment descriptions…")
        vectors = embedder.encode(descriptions)  # (N, D) float32, L2-normalised

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product == cosine for unit vecs
        index.add(vectors)

        print(f"Index built: {index.ntotal} vectors of dimension {dim}")
        return cls(index, list(segments))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Save the FAISS index and segment metadata to *directory*."""
        import faiss  # type: ignore

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(out / _VECTORS_FILE))

        with (out / _SEGMENTS_FILE).open("w", encoding="utf-8") as fh:
            json.dump([s.to_dict() for s in self._segments], fh, indent=2)

        print(f"Index saved to {out} ({self._index.ntotal} vectors)")

    @classmethod
    def load(cls, directory: str | Path) -> "SegmentIndex":
        """Load a previously saved index from *directory*."""
        import faiss  # type: ignore
        from video_knowledge_agent.retrieval.segmenter import VideoSegment

        d = Path(directory)
        if not d.is_dir():
            raise FileNotFoundError(f"Index directory not found: {d}")

        faiss_index = faiss.read_index(str(d / _VECTORS_FILE))

        with (d / _SEGMENTS_FILE).open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        segments = [VideoSegment.from_dict(item) for item in raw]
        print(f"Index loaded: {faiss_index.ntotal} vectors from {d}")
        return cls(faiss_index, segments)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        embedder: "Embedder",
        top_k: int = 10,
        rerank: bool = True,
    ) -> list[SearchResult]:
        """Return the top-k segments most similar to *query*.

        Parameters
        ----------
        query:
            Natural-language search string, e.g. "indoor scene with laptop".
        embedder:
            Must use the same model as was used at index-build time.
        top_k:
            Number of results to return.

        Returns
        -------
        list[SearchResult]
            Ranked from most to least similar.
        """
        q_vec = embedder.encode_query(query)  # (1, D)
        initial_k = min(self._index.ntotal, max(top_k, top_k * 5 if rerank else top_k, 20 if rerank else top_k))
        k = max(1, initial_k)
        scores, indices = self._index.search(q_vec, k)

        candidates: list[tuple[float, int, "VideoSegment"]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:  # FAISS returns -1 when fewer results than k
                break
            seg = self._segments[idx]
            final_score = _rerank_score(query, float(score), seg.description, seg.labels) if rerank else float(score)
            candidates.append((final_score, rank, seg))

        candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        results: list[SearchResult] = []
        for rank, (score, _, seg) in enumerate(candidates[:top_k], start=1):
            results.append(
                SearchResult(
                    rank=rank,
                    score=score,
                    video_id=seg.video_id,
                    video_path=seg.video_path,
                    start_s=seg.start_s,
                    end_s=seg.end_s,
                    description=seg.description,
                    labels=seg.labels,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def total_segments(self) -> int:
        return self._index.ntotal
