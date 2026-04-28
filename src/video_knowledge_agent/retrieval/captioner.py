"""VLM-based keyframe captioner for segment descriptions.

Combines YOLO detections (structured, reliable) with a vision-language model
for richer natural-language captions that capture visual details YOLO misses:
brand logos, actions, scene context, text in frame.

Supported backends
------------------
``"ollama"`` (default)
    Free, local. Uses Ollama's OpenAI-compatible API at http://localhost:11434.
    Requires Ollama running with a vision model pulled, e.g.::

        ollama pull llava

``"openai"``
    Uses OpenAI GPT-4o-mini. Requires OPENAI_API_KEY env var.

Falls back gracefully to the YOLO-only description when:
- The backend is unavailable (Ollama not running, no API key)
- The API call fails
- cv2 is not installed (no frame extraction)

Usage
-----
>>> from video_knowledge_agent.retrieval.captioner import VLMCaptioner
>>> cap = VLMCaptioner()                       # Ollama / llava
>>> cap_oai = VLMCaptioner(backend="openai")   # GPT-4o-mini
>>> desc = cap.caption(
...     video_path="data/raw/55-DET0eRRNKOm.mp4",
...     keyframe_time=12.5,
...     labels={"laptop": 15},
...     fallback_description="Scene with many laptops.",
... )
>>> print(desc)
'A tidy classroom with rows of open MacBook laptops on desks. No people visible. Context: office or workspace.'
"""

from __future__ import annotations

import base64
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_OLLAMA_DEFAULT_MODEL = "llava"

_CAPTION_SYSTEM = (
    "You are a concise video scene captioner. "
    "You will receive a single keyframe image and a list of YOLO-detected objects in the scene. "
    "Write ONE sentence (max 28 words) describing what is happening or visible in this scene. "
    "Be specific: mention actions, setting, brands, readable text, colours, or scene details that YOLO cannot capture. "
    "Prefer concrete nouns and verbs that would help semantic search. "
    "Do NOT start with 'The image shows' or 'In this frame'. Start directly with the scene content."
)

_CAPTION_USER_TEMPLATE = (
    "YOLO detections: {labels_str}\n\n"
    "Describe this scene in one sentence. If the detected objects are clearly present, naturally include the most important ones."
)

# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def _extract_frame_b64(video_path: str, timestamp_s: float, max_dim: int = 512) -> str | None:
    """Extract a single frame at *timestamp_s* and return it as base64 JPEG."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    path = Path(video_path)
    if not path.exists():
        return None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_idx = int(timestamp_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None

    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_labels(labels: dict[str, int]) -> str:
    """Turn label counts into a short readable string."""
    if not labels:
        return "none detected"
    parts = sorted(labels.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{lbl}({cnt})" for lbl, cnt in parts[:8])


def _ollama_reachable() -> bool:
    """Return True if the Ollama server is reachable at localhost:11434."""
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# VLMCaptioner
# ---------------------------------------------------------------------------


@dataclass
class VLMCaptioner:
    """Caption a keyframe using a VLM + YOLO detections.

    Parameters
    ----------
    backend:
        ``"ollama"`` (default) — free, local Llama-based vision model via Ollama.
        ``"openai"`` — GPT-4o-mini via OpenAI API (requires OPENAI_API_KEY).
    model:
        Model name to use.  Defaults to ``"llava"`` for Ollama and
        ``"gpt-4o-mini"`` for OpenAI.  Pass any vision-capable model name
        supported by your chosen backend.
    ollama_base_url:
        Base URL for the Ollama OpenAI-compatible endpoint.
        Default: ``"http://localhost:11434/v1"``.
    max_tokens:
        Max tokens in the caption response (one sentence, keep low).
    frame_max_dim:
        Resize keyframe so its largest dimension is at most this many pixels
        before sending to the backend (reduces cost/latency).
    cache_path:
        Optional path to a JSON file for persisting captions across runs.
        Keyed by ``"{video_id}@{keyframe_time}"``.  Set to ``None`` to disable.
    """

    backend: str = "ollama"
    model: str = ""  # resolved in __post_init__
    ollama_base_url: str = _OLLAMA_BASE_URL
    max_tokens: int = 40
    frame_max_dim: int = 512
    cache_path: str | None = None

    _client: object = field(default=None, init=False, repr=False)
    _cache: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model:
            self.model = _OLLAMA_DEFAULT_MODEL if self.backend == "ollama" else "gpt-4o-mini"
        if self.cache_path:
            self._load_cache()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, video_path: str, keyframe_time: float) -> str:
        video_id = Path(video_path).stem
        return f"{video_id}@{keyframe_time:.3f}"

    def _load_cache(self) -> None:
        path = Path(self.cache_path)  # type: ignore[arg-type]
        if path.exists():
            import json
            with path.open("r", encoding="utf-8") as fh:
                self._cache = json.load(fh)

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        import json
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self._cache, fh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def available(self) -> bool:
        """True when the selected backend is ready to accept requests."""
        try:
            import openai  # noqa: F401
        except ImportError:
            return False

        if self.backend == "ollama":
            return _ollama_reachable()
        elif self.backend == "openai":
            return bool(os.environ.get("OPENAI_API_KEY"))
        return False

    def caption(
        self,
        video_path: str,
        keyframe_time: float,
        labels: dict[str, int],
        fallback_description: str,
    ) -> str:
        """Return a VLM caption fused with YOLO labels, or *fallback_description*.

        Results are read from / written to the on-disk cache when *cache_path*
        is set, so interrupted runs can resume without re-calling the VLM.
        """
        if not self.available():
            return fallback_description

        # Check cache first
        key = self._cache_key(video_path, keyframe_time)
        if key in self._cache:
            return self._cache[key]

        frame_b64 = _extract_frame_b64(video_path, keyframe_time, self.frame_max_dim)
        if frame_b64 is None:
            return fallback_description

        labels_str = _format_labels(labels)

        try:
            vlm_sentence = self._call_vlm(frame_b64, labels_str)
        except Exception:
            return fallback_description

        if not vlm_sentence or len(vlm_sentence.strip()) < 5:
            return fallback_description

        # Fuse: VLM sentence + YOLO label suffix (helps embedding match object queries)
        yolo_suffix = _build_yolo_suffix(labels)
        combined = vlm_sentence.rstrip(".") + ". " + yolo_suffix if yolo_suffix else vlm_sentence

        # Persist to cache
        self._cache[key] = combined
        self._save_cache()

        return combined

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # type: ignore
            if self.backend == "ollama":
                self._client = OpenAI(
                    base_url=self.ollama_base_url,
                    api_key="ollama",  # Ollama ignores the key but the client requires one
                )
            else:
                self._client = OpenAI()
        return self._client

    def _call_vlm(self, frame_b64: str, labels_str: str) -> str:
        client = self._get_client()

        image_content: dict
        if self.backend == "ollama":
            # LLaVA via Ollama uses the same OpenAI-compat format with base64 data URLs
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            }
        else:
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}",
                    "detail": "low",
                },
            }

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": _CAPTION_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        image_content,
                        {
                            "type": "text",
                            "text": _CAPTION_USER_TEMPLATE.format(labels_str=labels_str),
                        },
                    ],
                },
            ],
        )
        return response.choices[0].message.content.strip()


def _build_yolo_suffix(labels: dict[str, int]) -> str:
    """Build a short object-listing suffix from YOLO labels for the embedding."""
    from video_knowledge_agent.retrieval.segmenter import build_label_suffix

    return build_label_suffix(labels)
