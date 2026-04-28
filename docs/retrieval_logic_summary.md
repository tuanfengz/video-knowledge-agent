# Retrieval Logic Summary

## 1. Offline — Index Build (`build_index.py`)

```
For each video in data/raw/:
    │
    ├── YOLO (YOLOv8n) scans every 3rd frame
    │       → { "person": 12, "cup": 3, "laptop": 5 }
    │         (count = distinct frames label appears in)
    │
    ├── Sliding window (5s window, 2.5s stride, 50% overlap)
    │       → ~37 segments per video
    │
    ├── [Optional --use-vlm] LLaVA / moondream captions keyframe
    │       → "A barista preparing espresso at a café counter."
    │         cached to vlm_captions_cache.json per segment
    │
    ├── Description = VLM sentence + YOLO suffix
    │       → "A barista preparing espresso at a café counter.
    │          Detected: cup, person. Context: dining or kitchen."
    │
    └── sentence-transformers (all-MiniLM-L6-v2)
            → 384-dim L2-normalised vector
            → stored in FAISS IndexFlatIP

Output: data/index_vlm/vectors.faiss + segments.json (4,816 segments)
```

---

## 2. Online — Query (`GET /search?q=...`)

```
User query: "someone making coffee"
    │
    ├── sentence-transformers encodes query → 384-dim vector
    │
    ├── FAISS cosine similarity (inner product on unit vectors)
    │       → top-K (video_id, start_s, end_s, score, description)
    │
    └── FastAPI returns ranked JSON → React UI
            → inline video player seeks to start_s and loops to end_s
```

---

## 3. What each component contributes

| Component | Role | Handles |
|---|---|---|
| **YOLO** | Structured grounding | Object names, counts — exact noun queries |
| **LLaVA/moondream** | Semantic richness | Actions, brands, scene context, text in frame |
| **sentence-transformers** | Embedding | Maps descriptions + queries to the same vector space |
| **FAISS** | Similarity search | Sub-millisecond nearest-neighbour over 4,816 vectors |

---

## 4. YOLO-only vs VLM-enhanced descriptions

| Query | YOLO-only | VLM-enhanced |
|---|---|---|
| `"laptop on desk"` | ✅ | ✅ |
| `"MacBook"` | ❌ | ✅ |
| `"students studying"` | ❌ | ✅ |
| `"university library"` | ❌ | ✅ |
| `"3 people with computers"` | ✅ (counts) | ✅ |
| `"someone making coffee"` | ❌ (just `cup`) | ✅ |
| `"election results on screen"` | ❌ | ✅ |

---

## 5. Design decision: detection only, no tracking

### What tracking adds vs what retrieval needs

| Capability | Tracking provides | Retrieval needs |
|---|---|---|
| Same object identity across frames | ✅ track ID continuity | ❌ not required |
| Object trajectory / movement | ✅ path over time | ❌ not required |
| Duration an object was present | ✅ track length | ❌ not required |
| Was object X present in segment? | ✅ (but detection alone suffices) | ✅ detection is enough |

### Why tracking is not needed here

For semantic retrieval, a segment description only needs to know **which objects appeared** and **how prominently**. Whether it is the *same* cup across 3 frames or 3 different cups does not change the description or the embedding.

```
Segment [10s–15s] — detection only is sufficient:
  Frame 1: person ✓, cup ✓
  Frame 3: person ✓, cup ✓
  Frame 5: person ✓, laptop ✓
  → labels = { "person": 3, "cup": 2, "laptop": 1 }
  → "Scene with a person, a cup, and a laptop."  ← all retrieval needs
```

Tracking would confirm it is the *same* cup — but the description and its embedding are identical either way.

### When tracking WOULD become necessary

Tracking is the right tool if the system is extended to support:

| Use case | Why tracking is needed |
|---|---|
| Action detection (`"person picks up bag"`) | Requires linking person + object across consecutive frames |
| Interaction events (`"two people shake hands"`) | Requires co-occurrence of specific track IDs |
| Duration-based queries (`"car parked for >10s"`) | Requires track lifespan |
| Re-ID across cuts (`"same person appears again"`) | Requires appearance-based track matching |

### Verdict for this system

Tracking adds **computational overhead** (optical flow, track ID management, re-identification) with **zero benefit** to description quality or embedding accuracy for short-form content retrieval.

Detection → description → embedding is the right design for this use case.

### When tracking + knowledge graph become relevant

For **long-form video** (lectures, sports matches, documentaries, surveillance), the calculus changes:

- Videos are too long for a single segment to capture meaningful context — you need to understand **how events relate across time**
- Tracking provides the thread connecting objects/people across segments
- A **knowledge graph** then stores those relationships (`person_A → picked_up → bag_B → at_time → 00:03:42`) and enables structured queries like *"find all moments where person A interacted with any object"*
- This is the architecture the earlier pipeline (YOLO + tracker + event extractor + Neo4j graph) was designed for

For the current dataset of ~30s short-form videos, neither is needed — each video is essentially a single scene, and semantic retrieval over independent segments is sufficient.

---

## 6. What queries are and aren't supported

Semantic search handles a spectrum from reliable to unsupported:

```
← Reliable                                              Not supported →

Object          Scene / Setting     General activity    Specific action    Causal event
presence        context             (VLM)               (requires          (requires
(YOLO)          (VLM)                                    tracking)          generation)

"bag"           "kitchen"           "cooking"           "picking up bag"   "did X cause Y"
"laptop"        "outdoor market"    "studying"          "handing over"     "what happened
"person"        "café counter"      "playing chess"     "throwing"          after X"
```

- **Objects + scene context** — reliably retrieved (YOLO + VLM)
- **General activities** — retrieved if visible in a single keyframe (VLM)
- **Specific inter-object actions** — outside scope; require tracking + action recognition
- **Causal / sequential events** — outside scope; require generation layer

---

## 7. What it is / isn't

- ✅ **Semantic retrieval** — finds relevant video segments by meaning
- ✅ **Timestamp-precise** — returns exact 5-second window to seek to
- ✅ **Graceful fallback** — VLM unavailable → YOLO-only descriptions used automatically
- ✅ **Resumable** — per-segment VLM caption cache survives interruptions
- ❌ **Not Q&A / generation** — returns ranked segments, not a synthesized answer
- ❌ **Not re-ranking** — no second-pass VLM verification of retrieved results

---

## 8. CLI reference

```bash
# Build YOLO-only index (fast, ~minutes)
python scripts/build_index.py --index-dir data/index

# Build VLM-enhanced index with moondream (free, local, slower first run)
python scripts/build_index.py --use-vlm --vlm-model moondream --index-dir data/index_vlm

# Build VLM-enhanced index with llava
python scripts/build_index.py --use-vlm --vlm-model llava --index-dir data/index_vlm

# Serve API with VLM index
INDEX_DIR=data/index_vlm uvicorn video_knowledge_agent.api.app:app --reload

# CLI search
python scripts/search_segments.py "someone making coffee" --top-k 5
```
