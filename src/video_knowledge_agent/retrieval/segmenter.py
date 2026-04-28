"""Segment generator: slices a video into overlapping time windows.

Each segment groups YOLO detections within a time window and attaches
a natural-language description suitable for embedding and semantic search.

Typical usage
-------------
>>> from video_knowledge_agent.retrieval.segmenter import segment_video
>>> segments = segment_video("data/raw/01-abc.mp4", window_s=5.0, stride_s=2.5)
>>> for seg in segments:
...     print(seg.video_id, seg.start_s, seg.end_s, seg.description)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VideoSegment:
    """One time-windowed segment of a video with its detection summary."""

    video_id: str
    """Filename stem, e.g. '01-abc' (no extension)."""

    video_path: str
    """Absolute or relative path to the source video file."""

    start_s: float
    """Start time of the segment in seconds."""

    end_s: float
    """End time of the segment in seconds (exclusive)."""

    labels: dict[str, int] = field(default_factory=dict)
    """Detected object labels mapped to their detection count within this window."""

    description: str = ""
    """Natural-language description suitable for text embedding."""

    keyframe_time: float = 0.0
    """Mid-point timestamp (seconds) — useful for frame sampling later."""

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "labels": self.labels,
            "description": self.description,
            "keyframe_time": self.keyframe_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VideoSegment":
        return cls(
            video_id=d["video_id"],
            video_path=d["video_path"],
            start_s=d["start_s"],
            end_s=d["end_s"],
            labels=d.get("labels", {}),
            description=d.get("description", ""),
            keyframe_time=d.get("keyframe_time", 0.0),
        )


# ---------------------------------------------------------------------------
# Description builder (Step 2)
# ---------------------------------------------------------------------------

# Map of label sets that hint at a scene type.
_SCENE_HINTS: list[tuple[frozenset[str], str]] = [
    (frozenset({"laptop", "monitor", "keyboard", "mouse", "desk", "chair"}), "office or workspace"),
    (frozenset({"car", "truck", "bus", "motorcycle", "bicycle"}), "traffic or street scene"),
    (frozenset({"dining table", "fork", "knife", "spoon", "cup", "bowl", "bottle"}), "dining or kitchen"),
    (frozenset({"bed", "pillow", "couch", "sofa", "tv", "television"}), "living room or bedroom"),
    (frozenset({"sports ball", "tennis racket", "baseball bat", "skateboard", "surfboard"}), "sports activity"),
    (frozenset({"backpack", "handbag", "suitcase"}), "travel or commuting"),
    (frozenset({"cell phone", "phone"}), "phone usage"),
    (frozenset({"book", "scissors", "clock"}), "study or home environment"),
]

_ACTIVITY_HINTS: list[tuple[frozenset[str], str]] = [
    (frozenset({"laptop", "keyboard", "mouse", "monitor"}), "computer work or study"),
    (frozenset({"cup", "bottle", "bowl", "fork", "knife", "spoon"}), "eating or drinking"),
    (frozenset({"car", "truck", "bus", "motorcycle", "bicycle"}), "travelling or commuting"),
    (frozenset({"cell phone", "phone"}), "using a phone"),
    (frozenset({"sports ball", "tennis racket", "baseball bat", "skateboard", "surfboard"}), "playing sports"),
]

_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _matched_hint_texts(
    labels: dict[str, int],
    hint_map: list[tuple[frozenset[str], str]],
) -> list[str]:
    detected_set = frozenset(labels.keys())
    return [hint_text for hint_labels, hint_text in hint_map if detected_set & hint_labels]


def build_label_suffix(labels: dict[str, int]) -> str:
    """Build structured metadata text that helps embedding-based retrieval."""
    if not labels:
        return ""

    scene_hints = _matched_hint_texts(labels, _SCENE_HINTS)
    activity_hints = _matched_hint_texts(labels, _ACTIVITY_HINTS)
    ordered_labels = sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))[:6]

    parts: list[str] = [
        "Detected objects: " + ", ".join(label for label, _ in ordered_labels) + "."
    ]
    if scene_hints:
        parts.append("Context: " + "; ".join(scene_hints) + ".")
    if activity_hints:
        parts.append("Likely activity: " + "; ".join(activity_hints) + ".")
    return " ".join(parts)


def build_description(labels: dict[str, int]) -> str:
    """Convert a label-count mapping into a natural-language scene description.

    Examples
    --------
    >>> build_description({"person": 2, "laptop": 1, "chair": 1})
    'Scene with two persons, one laptop, and one chair. Context: office or workspace.'

    >>> build_description({})
    'Empty scene with no detected objects.'
    """
    if not labels:
        return "Empty scene with no detected objects."

    # Sort by count descending so the most prominent objects come first.
    sorted_labels = sorted(labels.items(), key=lambda kv: -kv[1])

    parts: list[str] = []
    for label, count in sorted_labels:
        count_word = _COUNT_WORDS.get(count, "several" if count <= 8 else "many")
        # Naive pluralisation — good enough for description quality.
        plural = label if label.endswith("s") else f"{label}s"
        noun = label if count == 1 else plural
        parts.append(f"{count_word} {noun}")

    # Build the main clause.
    if len(parts) == 1:
        objects_clause = parts[0]
    elif len(parts) == 2:
        objects_clause = f"{parts[0]} and {parts[1]}"
    else:
        objects_clause = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    description = f"Scene with {objects_clause}."

    suffix = build_label_suffix(labels)
    if suffix:
        description += " " + suffix

    # Explicitly note people when present (improves query matching).
    if "person" in labels:
        count = labels["person"]
        if count == 1:
            description += " A person is visible."
        else:
            description += " Multiple people are visible."

    return description


# ---------------------------------------------------------------------------
# Segment generation (Step 1)
# ---------------------------------------------------------------------------

def segment_video(
    video_path: str,
    window_s: float = 5.0,
    stride_s: float = 2.5,
    confidence_threshold: float = 0.4,
    frame_stride: int = 3,
    use_vlm: bool = False,
    vlm_backend: str = "ollama",
    vlm_model: str = "",
    vlm_cache_path: str | None = None,
) -> list[VideoSegment]:
    """Slice *video_path* into overlapping time windows and return segments.

    Parameters
    ----------
    video_path:
        Path to the MP4 file to process.
    window_s:
        Duration of each segment window in seconds. Default 5 s.
    stride_s:
        Step between consecutive window start times. Default 2.5 s (50 % overlap).
        Must be > 0 and <= window_s.
    confidence_threshold:
        Minimum detection confidence to include. Lower values are noisier.
    frame_stride:
        Sample every N-th frame during detection (speed/quality trade-off).
    use_vlm:
        If True and OPENAI_API_KEY is set, use GPT-4o-mini to caption the
        keyframe of each segment, fused with YOLO labels. Falls back to
        label-only description when the API is unavailable.

    Returns
    -------
    list[VideoSegment]
        Segments in chronological order. An empty list is returned when the
        video cannot be read or yields no detections.
    """
    from video_knowledge_agent.vision.detector import Detector
    from video_knowledge_agent.vision.video_reader import VideoReader

    if stride_s <= 0 or stride_s > window_s:
        raise ValueError(f"stride_s must be in (0, window_s]. Got stride_s={stride_s}, window_s={window_s}")

    path = Path(video_path)
    video_id = path.stem

    # --- 1. Read video metadata ---
    reader = VideoReader()
    try:
        video_info = reader.read(str(path))
    except Exception:
        video_info = {}

    duration_s: float = float(video_info.get("duration", 0.0))

    # --- 2. Run object detection ---
    detector = Detector(
        confidence_threshold=confidence_threshold,
        frame_stride=frame_stride,
    )
    try:
        raw_detections: list[dict] = detector.detect(
            video_path=str(path), video_info=video_info
        )
    except Exception:
        raw_detections = []

    # Infer duration from detections if video_info gave nothing.
    if duration_s <= 0 and raw_detections:
        duration_s = max(d.get("timestamp", 0.0) for d in raw_detections) + 1.0

    if duration_s <= 0:
        return []

    # Initialise captioner once (lazy — only if use_vlm requested)
    captioner = None
    if use_vlm:
        from video_knowledge_agent.retrieval.captioner import VLMCaptioner
        kwargs: dict = {"backend": vlm_backend}
        if vlm_model:
            kwargs["model"] = vlm_model
        if vlm_cache_path:
            kwargs["cache_path"] = vlm_cache_path
        captioner = VLMCaptioner(**kwargs)
        if not captioner.available():
            print(f"  [VLM] backend '{vlm_backend}' not available — falling back to label descriptions.")
            captioner = None

    # --- 3. Build time-window segments ---
    segments: list[VideoSegment] = []
    window_start = 0.0

    while window_start < duration_s:
        window_end = min(window_start + window_s, duration_s)
        mid_s = (window_start + window_end) / 2.0

        # Collect detections that fall within [window_start, window_end).
        # We count distinct frames in which each label appears (presence-per-frame)
        # rather than raw detection instances, so the count reflects "how consistently
        # is this object present" instead of "how many bounding boxes fired".
        frames_with_label: dict[str, set[int]] = {}
        for det in raw_detections:
            t = det.get("timestamp", 0.0)
            if window_start <= t < window_end:
                lbl = det.get("label", "unknown")
                fi = det.get("frame_index", int(t * 30))
                frames_with_label.setdefault(lbl, set()).add(fi)

        label_counts: dict[str, int] = {lbl: len(frames) for lbl, frames in frames_with_label.items()}

        fallback_description = build_description(label_counts)
        if captioner is not None:
            description = captioner.caption(
                video_path=str(path),
                keyframe_time=mid_s,
                labels=label_counts,
                fallback_description=fallback_description,
            )
        else:
            description = fallback_description

        segments.append(
            VideoSegment(
                video_id=video_id,
                video_path=str(path),
                start_s=round(window_start, 3),
                end_s=round(window_end, 3),
                labels=label_counts,
                description=description,
                keyframe_time=round(mid_s, 3),
            )
        )

        # Stop if we've reached the end.
        if window_end >= duration_s:
            break
        window_start += stride_s

    return segments
