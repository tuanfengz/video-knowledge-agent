"""FastAPI application — semantic video segment retrieval."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Video Knowledge Agent API")

# ---------------------------------------------------------------------------
# Retrieval index — loaded once at startup, shared across requests
# ---------------------------------------------------------------------------
_retrieval_index = None
_retrieval_embedder = None
_INDEX_DIR = Path(os.environ.get("INDEX_DIR", "data/index"))


def _get_retrieval_index():
    """Lazy-load the FAISS segment index (thread-safe enough for startup)."""
    global _retrieval_index, _retrieval_embedder
    if _retrieval_index is None:
        from video_knowledge_agent.retrieval.embedder import Embedder
        from video_knowledge_agent.retrieval.index import SegmentIndex
        _retrieval_embedder = Embedder()
        _retrieval_index = SegmentIndex.load(_INDEX_DIR)
    return _retrieval_index, _retrieval_embedder


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SearchResult(BaseModel):
    rank: int
    score: float
    video_id: str
    video_path: str
    start_s: float
    end_s: float
    description: str
    labels: dict[str, int]


class SearchResponse(BaseModel):
    query: str
    total_segments_searched: int
    results: list[SearchResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse)
def search_segments(
    q: str,
    top_k: int = 10,
) -> SearchResponse:
    """Search for relevant video segments by natural-language query.

    Parameters
    ----------
    q:
        Natural-language query, e.g. "indoor scene with laptop and coffee".
    top_k:
        Number of results to return (default 10, max 50).

    Returns
    -------
    SearchResponse
        Ranked list of matching segments with video_id, timestamps, and scores.
    """
    if not q or not q.strip():
        raise HTTPException(status_code=422, detail="Query string 'q' must not be empty.")

    top_k = max(1, min(top_k, 50))

    if not _INDEX_DIR.is_dir():
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval index not found at {_INDEX_DIR}. Run 'python scripts/build_index.py' first.",
        )

    try:
        index, embedder = _get_retrieval_index()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to load index: {exc}")

    raw_results = index.search(q.strip(), embedder, top_k=top_k)

    return SearchResponse(
        query=q.strip(),
        total_segments_searched=index.total_segments,
        results=[
            SearchResult(
                rank=r.rank,
                score=r.score,
                video_id=r.video_id,
                video_path=r.video_path,
                start_s=r.start_s,
                end_s=r.end_s,
                description=r.description,
                labels=r.labels,
            )
            for r in raw_results
        ],
    )


_RAW_VIDEO_DIR = Path("data/raw")


@app.get("/video/raw/{video_id}")
def get_raw_video(video_id: str) -> FileResponse:
    """Stream a raw video from data/raw by its video_id (filename stem).

    Used by the search UI to play back the specific video for a result.
    The client is responsible for seeking to the correct timestamp using
    the HTML5 Media Fragment URI (#t=start,end).
    """
    # Sanitise: reject path traversal attempts
    if "/" in video_id or "\\" in video_id or ".." in video_id:
        raise HTTPException(status_code=400, detail="Invalid video_id")

    path = _RAW_VIDEO_DIR / f"{video_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

    return FileResponse(str(path), media_type="video/mp4")


@app.get("/video/{job_id}/annotated")
def get_annotated_video(job_id: str) -> FileResponse:
    """Stream the annotated video with bounding boxes and labels."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job status: {job['status']}")
    path = job.get("annotated_path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Annotated video not ready")
    return FileResponse(path, media_type="video/mp4")


@app.post("/query/{job_id}")
def query_video(job_id: str, body: QueryRequest) -> dict:
    """Ask a natural-language question about a processed video."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job status: {job['status']}")

    from video_knowledge_agent.agent.runner import VLMRunner

    artifact_paths: dict = job.get("result", {}).get("artifact_paths", {})
    runner = VLMRunner(
        video_path=str(job.get("video_path", "")),
        events_path=str(artifact_paths.get("events", "data/processed/events.json")),
        detections_path=str(artifact_paths.get("detections", "data/processed/detections.json")),
        tracks_path=str(artifact_paths.get("tracks", "data/processed/tracks.json")),
    )
    result = runner.run(body.question)
    return {
        "job_id": job_id,
        "question": body.question,
        "answer": result["answer"],
        "source": result.get("source", "graph"),
    }

