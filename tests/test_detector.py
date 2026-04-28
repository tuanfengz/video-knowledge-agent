import pytest

from video_knowledge_agent.vision.detector import Detector


def test_detector_raises_for_missing_video() -> None:
    with pytest.raises(RuntimeError, match="Video file not found"):
        Detector().detect("data/raw/does_not_exist.mp4")
