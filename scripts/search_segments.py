"""Query the segment retrieval index with a natural-language string.

Usage
-----
    python scripts/search_segments.py "indoor scene with laptop and coffee"
    python scripts/search_segments.py "vehicles on highway" --top-k 5
    python scripts/search_segments.py "person with backpack" --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Search video segments by natural-language query.")
    parser.add_argument("query", help="Natural-language search query.")
    parser.add_argument(
        "--index-dir",
        default=str(root / "data" / "index"),
        help="Directory containing the FAISS index (default: data/index).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10).",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model (must match the one used to build the index).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON instead of human-readable text.",
    )
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    if not index_dir.is_dir():
        print(
            f"Index not found at {index_dir}.\n"
            "Run 'python scripts/build_index.py' first.",
            file=sys.stderr,
        )
        return 1

    from video_knowledge_agent.retrieval.embedder import Embedder
    from video_knowledge_agent.retrieval.index import SegmentIndex

    embedder = Embedder(model_name=args.model)
    idx = SegmentIndex.load(index_dir)

    results = idx.search(args.query, embedder, top_k=args.top_k)

    if args.output_json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0

    print(f"\nQuery: \"{args.query}\"")
    print(f"Top {len(results)} results (index has {idx.total_segments} segments):\n")
    for r in results:
        print(f"  #{r.rank:>2}  score={r.score:.4f}  {r.video_id}  [{r.start_s:.1f}s – {r.end_s:.1f}s]")
        print(f"        {r.description[:100]}")
        if r.labels:
            top_labels = sorted(r.labels.items(), key=lambda kv: -kv[1])[:5]
            label_str = ", ".join(f"{lbl}({cnt})" for lbl, cnt in top_labels)
            print(f"        labels: {label_str}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
