# Video Knowledge Agent

Semantic video segment retrieval over a corpus of short-form videos. Search by natural language — results return the exact 5-second window to play back.

## How it works


### Pipeline Overview

**Offline (Index Building):**
1. **Segment videos:** Each video is split into short, overlapping windows (e.g., 5 seconds).
2. **Describe each segment:**
	- Run YOLOv8 object detection on each segment to get structured labels (e.g., "person", "laptop").
	- (Optional) Use a Vision-Language Model (VLM, e.g., LLaVA or moondream via Ollama) to caption the keyframe for richer, natural language descriptions.
	- Combine YOLO labels and VLM caption into a single, rich description for each segment.
3. **Embed and index:** Encode all segment descriptions using sentence-transformers (`all-MiniLM-L6-v2`) and build a FAISS index for fast retrieval.

**Online (Retrieval):**
4. **Semantic search:**
	- User query is embedded with the same model.
	- FAISS retrieves top matching segments by cosine similarity.
	- Lightweight reranker boosts results with direct lexical/label overlap.
	- Results are displayed in the React UI with inline video playback at the exact segment.

**Diagram:**
```
Offline:  video → YOLO detection → sliding window segments → [VLM captions] → sentence-transformers → FAISS index
Online:   query → sentence-transformers → FAISS cosine search → ranked segments → React UI playback
```

## Tech Stack

- **YOLOv8n** (Ultralytics) — object detection every 3rd frame
- **moondream / LLaVA** (Ollama, local) — optional VLM keyframe captioning
- **sentence-transformers** `all-MiniLM-L6-v2` — 384-dim embeddings
- **FAISS** `IndexFlatIP` — exact cosine nearest-neighbour search
- **FastAPI** — `/search` and `/video/raw/{id}` endpoints
- **React + Vite** — search UI with inline video playback

## Quickstart

1. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

2. Build the index (YOLO-only, fast):

	```bash
	python scripts/build_index.py --video-dir data/raw --index-dir data/index
	```

	Or with VLM enrichment (requires [Ollama](https://ollama.com) running locally):

	```bash
	python scripts/build_index.py --use-vlm --vlm-model moondream --index-dir data/index_vlm
	```

3. Start the API:

	```bash
	INDEX_DIR=data/index_vlm uvicorn video_knowledge_agent.api.app:app --reload
	```

4. Start the frontend:

	```bash
	cd frontend && npm install && npm run dev
	```

5. CLI search:

	```bash
	python scripts/search_segments.py "someone making coffee" --top-k 5
	```

6. Run tests:

	```bash
	pytest -q
	```

## Project Structure

```
video_knowledge_agent/
├── README.md
├── pyproject.toml
├── requirements.txt
├── setup.py
├── yolov8n.pt                          ← YOLOv8n weights
├── data/
│   ├── raw/                            ← source videos (.mp4)
│   ├── index/                          ← YOLO-only FAISS index
│   ├── index_vlm/                      ← VLM-enhanced FAISS index
│   └── uploads/                        ← upload staging dir
├── docs/
│   └── retrieval_logic_summary.md      ← design decisions + query capability
├── scripts/
│   ├── build_index.py                  ← build FAISS index (YOLO or VLM)
│   ├── search_segments.py              ← CLI semantic search
│   └── create_sample_video.py          ← generate test MP4
├── src/video_knowledge_agent/
│   ├── api/
│   │   └── app.py                      ← FastAPI: /health, /search, /video/raw/
│   ├── retrieval/
│   │   ├── segmenter.py                ← sliding window + YOLO detection
│   │   ├── captioner.py                ← VLM keyframe captioning (Ollama/OpenAI)
│   │   ├── embedder.py                 ← sentence-transformers wrapper
│   │   └── index.py                    ← FAISS build/save/load/search
│   ├── vision/
│   │   ├── detector.py                 ← YOLOv8 inference
│   │   └── video_reader.py             ← OpenCV frame extraction
│   └── utils/
│       ├── io.py
│       ├── logging.py
│       └── time_utils.py
├── frontend/
│   └── src/
│       └── components/
│           ├── SearchPanel.jsx         ← search bar + results list
│           └── VideoPlayer.jsx         ← inline video playback with seek
└── tests/
```

## Design notes

See [docs/retrieval_logic_summary.md](docs/retrieval_logic_summary.md) for design decisions, component breakdown, and query capability spectrum.

## Retrieval Quality Choices

- **Embedder choice: `all-MiniLM-L6-v2`**
	- We intentionally keep `all-MiniLM-L6-v2` as the embedding model because it gives a strong speed/quality tradeoff for this project.
	- It is fast to build over thousands of 5-second segments, keeps query latency low, and avoids the heavier offline cost of larger embedding models.
	- Larger models such as `all-mpnet-base-v2` may improve semantic quality, but they noticeably increase index-build time and memory usage. For this project, improving the text being embedded gives a better return than immediately switching models.

- **Richer descriptions before embedding**
	- Retrieval quality now relies on richer segment text, not a heavier embedder.
	- Each segment description includes:
		- detected objects
		- scene/context hints such as `office or workspace`, `traffic or street scene`, `dining or kitchen`
		- likely activity hints such as `computer work or study`, `eating or drinking`, `travelling or commuting`
		- explicit person visibility cues when relevant
	- When VLM captioning is enabled, the caption prompt is tuned to capture actions, setting, brands, readable text, colours, and other scene details that YOLO misses.
	- The VLM output is then fused with the structured object/context suffix so the embedding still preserves concrete object evidence for queries like `laptop`, `cup`, or `car`.

- **Lightweight reranking after FAISS**
	- Search still uses FAISS dense retrieval first.
	- After the initial top matches are retrieved, a lightweight reranker adjusts ranking using:
		- lexical overlap between the query and the segment description
		- direct overlap between query terms and YOLO labels
	- This improves precision for obvious object-driven queries without adding a heavy cross-encoder or significantly increasing query cost.

- **Practical tradeoff**
	- The current retrieval strategy is:
		- keep the fast default embedder
		- improve the text that gets embedded
		- add a cheap reranking stage on top of dense retrieval
	- This keeps indexing practical while improving relevance more efficiently than a model swap alone.

- **Important**
	- Description improvements only take effect after rebuilding the index, because the saved FAISS vectors reflect whatever text was embedded at build time.
