"""Build the segment retrieval index over all videos in a directory.

Usage
-----
    # Index all MP4s in data/raw, save to data/index
    python scripts/build_index.py

    # Custom paths and window size
    python scripts/build_index.py --video-dir data/raw --index-dir data/index \\
        --window 5.0 --stride 2.5

    # Limit to first N videos (useful for a quick test)
    python scripts/build_index.py --max-videos 5

    # Skip detection and reuse an existing segments cache
    python scripts/build_index.py --segments-cache data/index/segments_cache.json

This script:
  1. Iterates over all MP4 files in --video-dir.
  2. Runs YOLO detection + windowed segmentation on each video.
  3. (Optional) Captions each segment's keyframe with GPT-4o-mini (--use-vlm).
  4. Serialises all segments to a JSON cache (data/index/segments_cache.json).
  5. Encodes segment descriptions with sentence-transformers (all-MiniLM-L6-v2).
  6. Builds a FAISS IndexFlatIP and saves it alongside the segment metadata.

VLM enrichment example (requires OPENAI_API_KEY)
-------------------------------------------------
    # Build a VLM-enriched index into data/index_vlm (reuses existing segment cache)
    python scripts/build_index.py --use-vlm --index-dir data/index_vlm

    # Test on 3 videos first
    python scripts/build_index.py --use-vlm --max-videos 3 --index-dir data/index_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _process_videos(
    video_dir: Path,
    window_s: float,
    stride_s: float,
    max_videos: int | None,
    cache_path: Path,
    use_vlm: bool = False,
    vlm_backend: str = "ollama",
    vlm_model: str = "",
    vlm_cache_path: str | None = None,
) -> list[dict]:
    """Run segmentation on each video and return all segment dicts.

    Caches per-video results incrementally so the script can be resumed
    if interrupted.
    """
    from video_knowledge_agent.retrieval.segmenter import segment_video

    # Load existing cache so we can skip already-processed videos.
    cached: dict[str, list[dict]] = {}
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as fh:
            cached_list: list[dict] = json.load(fh)
        for seg in cached_list:
            cached.setdefault(seg["video_id"], []).append(seg)
        print(f"Loaded {len(cached_list)} cached segments for {len(cached)} videos.")

    mp4_files = sorted(video_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"No MP4 files found in {video_dir}", file=sys.stderr)
        sys.exit(1)

    if max_videos is not None:
        mp4_files = mp4_files[:max_videos]

    all_segments: list[dict] = []
    n_total = len(mp4_files)

    for i, video_path in enumerate(mp4_files, start=1):
        video_id = video_path.stem
        print(f"\n[{i}/{n_total}] {video_path.name}")

        if video_id in cached:
            print(f"  → cached ({len(cached[video_id])} segments), skipping detection")
            all_segments.extend(cached[video_id])
            continue

        t0 = time.perf_counter()
        try:
            segments = segment_video(
                str(video_path),
                window_s=window_s,
                stride_s=stride_s,
                use_vlm=use_vlm,
                vlm_backend=vlm_backend,
                vlm_model=vlm_model,
                vlm_cache_path=vlm_cache_path,
            )
        except Exception as exc:
            print(f"  ✗ failed: {exc}", file=sys.stderr)
            continue

        elapsed = time.perf_counter() - t0
        print(f"  ✓ {len(segments)} segments in {elapsed:.1f}s")
        for s in segments[:2]:
            print(f"    [{s.start_s:.1f}s–{s.end_s:.1f}s] {s.description[:80]}")

        seg_dicts = [s.to_dict() for s in segments]
        all_segments.extend(seg_dicts)

        # Write incremental cache after each video so progress is preserved.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(all_segments, fh)

    return all_segments


def main() -> int:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Build segment retrieval index.")
    parser.add_argument(
        "--video-dir",
        default=str(root / "data" / "raw"),
        help="Directory containing MP4 files (default: data/raw).",
    )
    parser.add_argument(
        "--index-dir",
        default=str(root / "data" / "index"),
        help="Output directory for FAISS index files (default: data/index).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Segment window size in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=2.5,
        help="Stride between windows in seconds (default: 2.5).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Process at most N videos (useful for quick tests).",
    )
    parser.add_argument(
        "--segments-cache",
        default=None,
        help="Path to existing segments JSON cache to skip re-detection.",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--use-vlm",
        action="store_true",
        default=False,
        help=(
            "Enrich segment descriptions with VLM keyframe captions. "
            "Defaults to Ollama/llava (free, local). Use --vlm-backend openai for GPT-4o-mini."
        ),
    )
    parser.add_argument(
        "--vlm-backend",
        default="ollama",
        choices=["ollama", "openai"],
        help="VLM backend to use with --use-vlm (default: ollama).",
    )
    parser.add_argument(
        "--vlm-model",
        default="",
        help="Override VLM model name (default: llava for ollama, gpt-4o-mini for openai).",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    index_dir = Path(args.index_dir)
    cache_path = Path(args.segments_cache) if args.segments_cache else index_dir / "segments_cache.json"
    vlm_cache_path = str(index_dir / "vlm_captions_cache.json") if args.use_vlm else None

    # --- Step 1 & 2: segment all videos ---
    print("=" * 60)
    vlm_status = "ON ({} / {})".format(args.vlm_backend, args.vlm_model or "default model") if args.use_vlm else "OFF (YOLO labels only)"
    print(f"STEP 1 & 2: Video segmentation + description generation [VLM: {vlm_status}]")
    print("=" * 60)
    seg_dicts = _process_videos(
        video_dir=video_dir,
        window_s=args.window,
        stride_s=args.stride,
        max_videos=args.max_videos,
        cache_path=cache_path,
        use_vlm=args.use_vlm,
        vlm_backend=args.vlm_backend,
        vlm_model=args.vlm_model,
        vlm_cache_path=vlm_cache_path,
    )

    if not seg_dicts:
        print("No segments produced. Aborting.", file=sys.stderr)
        return 1

    print(f"\nTotal segments across all videos: {len(seg_dicts)}")

    # --- Step 3: embed + index ---
    print("\n" + "=" * 60)
    print("STEP 3: Embedding + FAISS index construction")
    print("=" * 60)

    from video_knowledge_agent.retrieval.embedder import Embedder
    from video_knowledge_agent.retrieval.index import SegmentIndex
    from video_knowledge_agent.retrieval.segmenter import VideoSegment

    embedder = Embedder(model_name=args.model)
    segments = [VideoSegment.from_dict(d) for d in seg_dicts]

    t0 = time.perf_counter()
    idx = SegmentIndex.build(segments, embedder)
    elapsed = time.perf_counter() - t0
    print(f"Embedding + indexing complete in {elapsed:.1f}s")

    idx.save(index_dir)

    print("\n" + "=" * 60)
    print("Index ready. Run a quick sanity-check query:")
    print('  python scripts/search_segments.py "person with laptop"')
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
