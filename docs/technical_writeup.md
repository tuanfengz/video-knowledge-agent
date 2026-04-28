# Technical Writeup: Multimodal Video Segment Retrieval

## 1. Approach & System Design

The goal of this system is to enable semantic retrieval of video segments (not entire videos) given a natural language query. The core design principle is to convert multimodal video content into a searchable semantic space by combining structured visual signals with natural language descriptions.

### Pipeline Overview

The system is divided into two stages:

**Offline (Indexing):**
- Videos are segmented into short, overlapping windows (5 seconds).
- Each segment is enriched with:
  - Structured signals: object detections using YOLO (e.g., “person”, “laptop”)
  - Unstructured signals (optional): VLM-generated captions describing scene, activity, and context
- These are combined into a unified text description per segment.
- Descriptions are embedded using a lightweight sentence embedding model and indexed with FAISS.

**Online (Retrieval):**
- A user query is embedded into the same vector space.
- FAISS retrieves top candidate segments via cosine similarity.
- A lightweight reranker boosts results based on lexical overlap with detected objects.
- The system returns the top-ranked segments with timestamps for playback.

## 2. Design Rationale

### Why text-based multimodal fusion?

Instead of directly embedding images or video frames, the system converts visual signals into textual descriptions before embedding. This enables:
- better alignment with natural language queries
- easier integration of structured (YOLO) and unstructured (VLM) signals
- use of fast, compact embedding models

This approach prioritizes representation quality over model size, which proved more efficient than switching to heavier embedding models.

### Why 5-second sliding windows?

Fixed windows provide:
- full temporal coverage
- simple parallelization

However, they introduce tradeoffs:
- semantic boundaries may not align with segment boundaries
- short events may be diluted

This was accepted for simplicity, with future improvements planned via adaptive segmentation.

### Why lightweight embedding + reranking?

The system uses a fast embedding model (all-MiniLM-L6-v2) for scalability. Instead of increasing model complexity:
- semantic richness is improved upstream (better descriptions)
- precision is improved downstream (light reranking)

This balances:
- indexing cost
- query latency
- retrieval quality

## 3. Alternatives Considered

### End-to-end VLM embeddings

Using models like LLaVA directly for embedding was considered but rejected due to:
- higher compute cost
- slower indexing
- limited benefit compared to structured + caption fusion

### Frame-level retrieval

Retrieving individual frames instead of segments:
- improves granularity
- but loses temporal context and increases index size significantly

Segment-level retrieval provides a better balance between context and efficiency.

### Cross-encoder reranking

More advanced reranking (e.g., cross-encoders) could improve accuracy but was not used due to:
- higher latency
- reduced scalability for interactive use

## 4. Failure Analysis

The system’s limitations are closely tied to its design:

- **Compositional queries:** Relationships between objects (e.g., “person drinking coffee while working”) are not explicitly modeled, since segment representations flatten objects and captions into text.
- **Temporal mismatch:** Fixed segmentation can split or dilute short events.
- **Fine-grained recognition:** Brand-level queries (e.g., “Nike backpack”) are unreliable due to detector limitations and inconsistent captioning.
- **Visual variability:** Performance degrades under low-light or motion-heavy scenes.
- **Redundancy:** Overlapping windows can produce near-duplicate results.

These highlight the tradeoff between efficiency and richer temporal/relational modeling.

## 5. Scaling to Large Datasets

To scale to millions of videos:
- **Offline preprocessing:** All embeddings are precomputed using distributed workers (e.g., Ray/Spark), eliminating heavy compute at query time.
- **Efficient indexing:** FAISS can be upgraded to approximate search (IVF/HNSW) or replaced with a vector database for horizontal scaling.
- **Hierarchical retrieval:** First retrieve top candidate videos using coarse embeddings, then search segments within them to reduce search space.
- **Metadata filtering:** Use structured signals (e.g., detected objects) to pre-filter candidates before vector search.

For long videos:
- adaptive segmentation (e.g., shot detection)
- multi-resolution indexing (coarse-to-fine)
- keyframe-based representations

These reduce both compute and storage while preserving semantic relevance.

## 6. Future Improvements

Several extensions could significantly improve retrieval quality:
- **Temporal modeling:** Use sequence models (e.g., transformers) to capture motion and event dynamics within segments.
- **Query decomposition:** Use LLMs to break complex queries into components and combine retrieval signals.
- **Better multimodal embeddings:** Replace text-only embedding with joint vision-language embeddings (e.g., CLIP-style).
- **Deduplication:** Remove redundant overlapping segments in post-processing.
- **Fine-grained recognition:** Incorporate specialized detectors or OCR for brand/text queries.

## 7. Summary

This system demonstrates a practical approach to multimodal video retrieval by:
- combining structured and generative visual understanding
- leveraging efficient embedding and indexing strategies
- balancing scalability with semantic richness

The design prioritizes modularity, interpretability, and extensibility, making it suitable for both prototyping and scaling to production systems.
