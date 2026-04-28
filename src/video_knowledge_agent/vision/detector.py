from dataclasses import dataclass
from pathlib import Path


@dataclass
class Detector:
    model_name: str = "yolov8n"
    frame_stride: int = 1  # Process every frame for accuracy
    confidence_threshold: float = 0.5
    use_gpu: bool = False

    def _handle_failure(
        self,
        reason: str,
        video_path: str,
        total_frames: int,
        fps: float,
        exception: Exception | None = None,
    ) -> list[dict]:
        if exception is not None:
            raise RuntimeError(reason) from exception
        raise RuntimeError(reason)

    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO

            return YOLO(f"{self.model_name}.pt")
        except ImportError:
            return None
        except Exception as e:
            return e

    def detect(self, video_path: str, video_info: dict | None = None) -> list[dict]:
        """Detect objects in video. Returns frame-level detections with timestamps."""
        info = video_info or {}
        total_frames = int(info.get("frame_count") or 300)
        fps = float(info.get("fps") or 30.0)
        if total_frames <= 0:
            total_frames = 300
        if fps <= 0:
            fps = 30.0

        path = Path(video_path)

        # Check if file exists and is valid
        if not path.exists() or path.stat().st_size == 0:
            if not path.exists():
                return self._handle_failure(
                    reason=f"Video file not found: {video_path}",
                    video_path=video_path,
                    total_frames=total_frames,
                    fps=fps,
                )
            else:
                return self._handle_failure(
                    reason=f"Video file is empty: {video_path}",
                    video_path=video_path,
                    total_frames=total_frames,
                    fps=fps,
                )

        model = self._load_model()
        if model is None or isinstance(model, Exception):
            model_error = (
                "ultralytics is not installed"
                if model is None
                else f"Failed to load model: {model}"
            )
            return self._handle_failure(
                reason=f"Unable to initialize detector model ({model_error})",
                video_path=video_path,
                total_frames=total_frames,
                fps=fps,
                exception=model if isinstance(model, Exception) else None,
            )

        detections: list[dict] = []
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))

            # Verify the video was opened successfully
            if not cap.isOpened():
                cap.release()
                return self._handle_failure(
                    reason=f"Failed to open video: {video_path}",
                    video_path=video_path,
                    total_frames=total_frames,
                    fps=fps,
                )

            frame_idx = 0
            frames_read = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frames_read += 1

                # Process every N frames
                if frame_idx % self.frame_stride == 0:
                    results = model(frame, conf=self.confidence_threshold, verbose=False)
                    timestamp = frame_idx / fps

                    for result in results:
                        for box in result.boxes:
                            cls_id = int(box.cls[0]) if len(box.cls) > 0 else 0
                            label = result.names[cls_id] if cls_id in result.names else "object"
                            conf = float(box.conf[0]) if len(box.conf) > 0 else 0.0
                            bbox = box.xyxy[0].tolist() if len(box.xyxy) > 0 else [0, 0, 0, 0]

                            detections.append({
                                "video_path": str(path),
                                "frame_index": frame_idx,
                                "timestamp": round(timestamp, 3),
                                "label": label,
                                "confidence": round(conf, 3),
                                "bbox": [round(x, 2) for x in bbox],
                                "model_name": self.model_name,
                            })

                frame_idx += 1
            cap.release()

            # Treat unreadable videos as hard failures.
            if frames_read == 0:
                return self._handle_failure(
                    reason=f"No frames read from video: {video_path}",
                    video_path=video_path,
                    total_frames=total_frames,
                    fps=fps,
                )

        except Exception as e:
            return self._handle_failure(
                reason=f"Error during detection: {e}",
                video_path=video_path,
                total_frames=total_frames,
                fps=fps,
                exception=e,
            )

        return detections
