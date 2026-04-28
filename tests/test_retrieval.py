"""Unit tests for the retrieval pipeline.

These tests use synthetic data — no real video files or network calls needed.
The Embedder loads the sentence-transformers model (~22 MB, cached after first run).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from video_knowledge_agent.retrieval.embedder import Embedder
from video_knowledge_agent.retrieval.index import SearchResult, SegmentIndex
from video_knowledge_agent.retrieval.segmenter import VideoSegment, build_description


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Shared Embedder instance — model is downloaded once and cached."""
    return Embedder()


def _make_segment(
    video_id: str = "test_video",
    start_s: float = 0.0,
    end_s: float = 5.0,
    labels: dict | None = None,
    description: str = "A person sitting at a desk with a laptop.",
) -> VideoSegment:
    return VideoSegment(
        video_id=video_id,
        video_path=f"data/raw/{video_id}.mp4",
        start_s=start_s,
        end_s=end_s,
        labels=labels or {"person": 3, "laptop": 2},
        description=description,
        keyframe_time=(start_s + end_s) / 2,
    )


# ---------------------------------------------------------------------------
# VideoSegment
# ---------------------------------------------------------------------------

class TestVideoSegment:
    def test_to_dict_round_trip(self):
        seg = _make_segment()
        restored = VideoSegment.from_dict(seg.to_dict())
        assert restored.video_id == seg.video_id
        assert restored.start_s == seg.start_s
        assert restored.end_s == seg.end_s
        assert restored.labels == seg.labels
        assert restored.description == seg.description
        assert restored.keyframe_time == seg.keyframe_time

    def test_from_dict_missing_optional_fields(self):
        minimal = {
            "video_id": "v1",
            "video_path": "data/raw/v1.mp4",
            "start_s": 0.0,
            "end_s": 5.0,
        }
        seg = VideoSegment.from_dict(minimal)
        assert seg.labels == {}
        assert seg.description == ""
        assert seg.keyframe_time == 0.0


# ---------------------------------------------------------------------------
# build_description
# ---------------------------------------------------------------------------

class TestBuildDescription:
    def test_empty_labels_returns_generic(self):
        desc = build_description({})
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_single_label(self):
        desc = build_description({"person": 1})
        assert "person" in desc

    def test_multiple_labels_all_included(self):
        desc = build_description({"person": 3, "laptop": 2, "cup": 1})
        assert "person" in desc
        assert "laptop" in desc
        assert "cup" in desc

    def test_scene_hint_injected(self):
        # "laptop" + "keyboard" should trigger the office hint
        desc = build_description({"laptop": 3, "keyboard": 2})
        assert "office" in desc or "workspace" in desc

    def test_description_includes_detected_objects_suffix(self):
        desc = build_description({"laptop": 3, "keyboard": 2})
        assert "Detected objects:" in desc
        assert "laptop" in desc
        assert "keyboard" in desc

    def test_description_includes_activity_hint(self):
        desc = build_description({"laptop": 2, "mouse": 1})
        assert "Likely activity:" in desc
        assert "computer work" in desc or "study" in desc


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class TestEmbedder:
    def test_encode_returns_float32(self, embedder):
        vecs = embedder.encode(["hello world"])
        assert vecs.dtype == np.float32

    def test_encode_shape(self, embedder):
        texts = ["scene one", "scene two", "scene three"]
        vecs = embedder.encode(texts)
        assert vecs.shape[0] == len(texts)
        assert vecs.shape[1] == embedder.embedding_dim

    def test_encode_unit_norm(self, embedder):
        vecs = embedder.encode(["normalised vector test"])
        norms = np.linalg.norm(vecs, axis=1)
        assert pytest.approx(norms, abs=1e-5) == [1.0]

    def test_encode_query_shape(self, embedder):
        vec = embedder.encode_query("test query")
        assert vec.shape == (1, embedder.embedding_dim)

    def test_encode_empty_raises(self, embedder):
        with pytest.raises(ValueError, match="non-empty"):
            embedder.encode([])

    def test_embedding_dim_is_384(self, embedder):
        assert embedder.embedding_dim == 384


# ---------------------------------------------------------------------------
# SegmentIndex
# ---------------------------------------------------------------------------

class TestSegmentIndex:
    def _build(self, embedder: Embedder) -> tuple[SegmentIndex, list[VideoSegment]]:
        segments = [
            _make_segment("v1", 0.0, 5.0, {"person": 3, "laptop": 2}, "Person typing on a laptop at a desk."),
            _make_segment("v1", 2.5, 7.5, {"cup": 2, "person": 1}, "Someone making coffee in a kitchen."),
            _make_segment("v2", 0.0, 5.0, {"car": 4, "truck": 1}, "Heavy traffic on a highway at night."),
        ]
        return SegmentIndex.build(segments, embedder), segments

    def test_build_empty_raises(self, embedder):
        with pytest.raises(ValueError):
            SegmentIndex.build([], embedder)

    def test_total_segments(self, embedder):
        index, segments = self._build(embedder)
        assert index.total_segments == len(segments)

    def test_search_returns_results(self, embedder):
        index, _ = self._build(embedder)
        results = index.search("laptop on desk", embedder, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_ranks_start_at_1(self, embedder):
        index, _ = self._build(embedder)
        results = index.search("coffee", embedder, top_k=3)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_search_top_k_capped_at_total(self, embedder):
        index, segments = self._build(embedder)
        results = index.search("anything", embedder, top_k=100)
        assert len(results) == len(segments)

    def test_search_scores_descending(self, embedder):
        index, _ = self._build(embedder)
        results = index.search("car on highway", embedder, top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_result_ranked_first(self, embedder):
        index, _ = self._build(embedder)
        results = index.search("car traffic highway", embedder, top_k=3)
        assert results[0].video_id == "v2"

    def test_save_and_load_round_trip(self, embedder):
        index, _ = self._build(embedder)
        with tempfile.TemporaryDirectory() as tmpdir:
            index.save(tmpdir)
            loaded = SegmentIndex.load(tmpdir)
            assert loaded.total_segments == index.total_segments
            results = loaded.search("laptop", embedder, top_k=1)
            assert len(results) == 1

    def test_load_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            SegmentIndex.load("/nonexistent/path/that/does/not/exist")

    def test_reranking_uses_query_label_overlap(self, embedder):
        segments = [
            _make_segment("v1", 0.0, 5.0, {"car": 1}, "A calm indoor workspace scene."),
            _make_segment("v2", 0.0, 5.0, {"laptop": 1}, "A calm indoor workspace scene."),
        ]
        index = SegmentIndex.build(segments, embedder)
        results = index.search("laptop workspace", embedder, top_k=2)
        assert results[0].video_id == "v2"
