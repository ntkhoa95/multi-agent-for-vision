import logging
import os
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from vision_framework import VisionOrchestrator, VisionTaskType


def download_test_video():
    """Download a test video if not exists."""
    import urllib.request

    video_dir = Path("tests/data/videos")
    video_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / "pedestrians.mp4"
    if not video_path.exists():
        logger.info(f"Downloading test video to {video_path}")
        # Replace with your video URL
        url = "https://example.com/sample_video.mp4"
        urllib.request.urlretrieve(url, video_path)
    return video_path


def print_video_info(video_path: str):
    """Print video information"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info("Video Information:")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Frame count: {frame_count}")
    logger.info(f"  Duration: {frame_count/fps:.2f} seconds")
    cap.release()


def process_video_with_caption(
    orchestrator: VisionOrchestrator,
    video_path: str,
    output_path: str,
    target_classes: list = None,
    process_duration: float = None,
    caption_interval: int = 30,  # Generate caption every N frames
):
    """Process video with detection and periodic captioning."""
    try:
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Process video with detection
        logger.info("Processing video with detection and tracking...")
        query = f"detect {' and '.join(target_classes)}" if target_classes else "detect all objects"

        video_result, frame_captions = orchestrator.process_video(
            video_path=video_path,
            output_path=output_path,
            user_comment=query,
            start_time=0,
            end_time=process_duration,
        )

        # Print statistics
        logger.info("\nProcessing Results:")
        logger.info(f"  Processed frames: {video_result.num_frames}")
        logger.info(f"  Average FPS: {video_result.num_frames / video_result.total_time:.2f}")

        # Calculate detection statistics
        total_detections = 0
        all_classes = set()
        tracked_objects = set()

        for frame_idx, frame in enumerate(video_result.frames_results):
            detections = frame.results["detections"]
            total_detections += len(detections)

            # Update statistics
            for det in detections:
                all_classes.add(det["class"])
                if "track_id" in det:
                    tracked_objects.add((det["class"], det["track_id"]))

            # Generate caption for periodic frames
            if frame_idx % caption_interval == 0 and frame_captions:
                caption = frame_captions[frame_idx].results.get("caption")
                if caption:
                    logger.info(f"\nFrame {frame_idx} Caption:")
                    logger.info(caption)

        avg_detections = total_detections / video_result.num_frames
        logger.info(f"\nDetection Statistics:")
        logger.info(f"  Total detections: {total_detections}")
        logger.info(f"  Average detections per frame: {avg_detections:.2f}")
        logger.info(f"  Detected classes: {sorted(list(all_classes))}")

        if tracked_objects:
            logger.info(f"\nTracking Statistics:")
            logger.info(f"  Unique tracked objects: {len(tracked_objects)}")
            class_counts = {}
            for obj_class, _ in tracked_objects:
                class_counts[obj_class] = class_counts.get(obj_class, 0) + 1
            for obj_class, count in class_counts.items():
                logger.info(f"    {obj_class}: {count} instances")

        logger.info(f"\nOutput saved to: {output_path}")
        return video_result, frame_captions

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise


def main():
    # Configuration
    config = {
        "ENABLE_CAPTIONING": True,
        "YOLO_MODEL_NAME": "yolov8s.pt",
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.45,
        "DETECTION_IMAGE_SIZE": 640,
        "ENABLE_TRACK": True,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "MAX_CAPTION_LENGTH": 50,
        "MIN_CAPTION_LENGTH": 10,
        "VIDEO_FPS": 30,
    }

    # Initialize orchestrator
    logger.info("Initializing VisionOrchestrator...")
    orchestrator = VisionOrchestrator(config)

    # Process video examples
    video_path = "tests/data/videos/crosswalk.avi"  # Replace with your video path
    logger.info(f"\nProcessing video: {video_path}")
    print_video_info(video_path)

    # Example 1: Track and caption people
    logger.info("\nExample 1: Track and caption people")
    logger.info("-" * 50)

    try:
        output_path = "examples/output/people_detection.mp4"
        result, captions = process_video_with_caption(
            orchestrator=orchestrator,
            video_path=video_path,
            output_path=output_path,
            target_classes=["person"],
            process_duration=10.0,  # Process first 10 seconds
            caption_interval=15,  # Caption every 15 frames
        )

    except Exception as e:
        logger.error(f"Error in people tracking: {str(e)}")

    # Example 2: Track multiple object types
    logger.info("\nExample 2: Track multiple objects")
    logger.info("-" * 50)

    try:
        output_path = "examples/output/multi_object_detection.mp4"
        result, captions = process_video_with_caption(
            orchestrator=orchestrator,
            video_path=video_path,
            output_path=output_path,
            target_classes=["person", "car", "bicycle"],
            process_duration=10.0,
            caption_interval=15,
        )

    except Exception as e:
        logger.error(f"Error in multi-object tracking: {str(e)}")


if __name__ == "__main__":
    main()
