from pydantic import BaseModel, Field
from typing import Optional


class BoundingBox(BaseModel):
    """Bounding box with coordinates [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """Object detection in a single frame."""
    video_path: str
    frame_index: int
    timestamp: float = Field(description="Timestamp in seconds")
    label: str = Field(description="Object class label")
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(description="[x1, y1, x2, y2]")
    model_name: str = Field(default="yolov8n")


class Track(BaseModel):
    """Object track across multiple frames."""
    track_id: str
    frame_index: int
    timestamp: float
    label: str
    bbox: list[float]
    confidence: float
    age: int = Field(description="Number of frames this track has been active")
    start_frame: int
    algorithm: str = Field(default="centroid-distance-tracker")
