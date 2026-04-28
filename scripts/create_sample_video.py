#!/usr/bin/env python3
"""
Helper script to create a sample video file for testing.
This generates a simple MP4 video with colored frames for testing the detector.
"""

import sys
from pathlib import Path


def create_sample_video(output_path: str = "data/raw/sample_video.mp4", num_frames: int = 100, width: int = 640, height: int = 480, fps: int = 30):
    """
    Create a simple test video file.
    
    Args:
        output_path: Path where to save the video
        num_frames: Number of frames to generate
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("❌ Error: cv2 and numpy are required to create test videos")
        print("   Install with: pip install opencv-python numpy")
        return False

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"📹 Creating sample video: {output_path}")
    print(f"   Frames: {num_frames}, Size: {width}x{height}, FPS: {fps}")

    # Define video codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    if not out.isOpened():
        print("❌ Failed to create video writer")
        return False

    try:
        for frame_idx in range(num_frames):
            # Create a frame with some content (gradient + circles)
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Add gradient background
            for i in range(height):
                frame[i, :] = [int(255 * i / height), 100, int(200 - 100 * i / height)]

            # Draw some shapes that simulate detectable objects
            # "Person" - blue circle
            person_x = 100 + (frame_idx % 200)
            cv2.circle(frame, (person_x, 200), 40, (255, 0, 0), -1)  # Blue circle

            # "Backpack" - red rectangle
            backpack_x = person_x + 60
            cv2.rectangle(frame, (backpack_x, 180), (backpack_x + 50, 250), (0, 0, 255), -1)  # Red rectangle

            # Add frame number
            cv2.putText(frame, f"Frame: {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Write frame
            out.write(frame)

        out.release()
        file_size = output_file.stat().st_size
        print(f"✅ Video created successfully!")
        print(f"   File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
        return True

    except Exception as e:
        print(f"❌ Error creating video: {e}")
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Create a sample video file for testing")
    parser.add_argument("--output", default="data/raw/sample_video.mp4", help="Output path for video file")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to generate")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")

    args = parser.parse_args()

    success = create_sample_video(
        output_path=args.output,
        num_frames=args.frames,
        width=args.width,
        height=args.height,
        fps=args.fps
    )

    if success:
        print()
        print("Now you can run:")
        print(f"  python -m video_knowledge_agent.main --video {args.output}")
        print("  or")
        print("  python scripts/demo_refactored.py")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
