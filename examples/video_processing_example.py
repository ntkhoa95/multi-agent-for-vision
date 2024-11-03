import logging
import os
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from vision_framework import VisionOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config():
    """Get video processing configuration"""
    return {
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "YOLO_MODEL_NAME": "yolov8s.pt",
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.45,
        "DETECTION_IMAGE_SIZE": 640,
        "ENABLE_TRACK": True,
        "BATCH_SIZE": 1,
        "NUM_WORKERS": 0,
    }


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


def main():
    # Initialize framework
    config = get_config()
    orchestrator = VisionOrchestrator(config)
    logger.info(f"Using device: {config['DEVICE']}")

    # Ensure output directory exists
    os.makedirs("examples/output", exist_ok=True)

    # Example video
    video_path = "tests/data/videos/crosswalk.avi"
    logger.info(f"\nProcessing video: {video_path}")
    print_video_info(video_path)

    # Test different video processing queries
    video_queries = [
        ("detect all objects in video", None),
        ("track people", ["person"]),
        ("detect and track cars", ["car"]),
        ("find people and vehicles", ["person", "car", "truck", "bus"]),
    ]

    for query, target_classes in video_queries:
        try:
            # Create output path
            output_name = (
                f"output_{Path(video_path).stem}_{query.replace(' ', '_')}.mp4"
            )
            output_path = f"examples/output/{output_name}"

            logger.info(f"\nProcessing with query: {query}")
            if target_classes:
                logger.info(f"Target classes: {target_classes}")

            # Process video
            result = orchestrator.process_video(
                video_path=video_path,
                user_comment=query,
                output_path=output_path,
                start_time=0,
                end_time=10,  # Process first 10 seconds
            )

            # Print statistics
            logger.info("\nProcessing Results:")
            logger.info(f"  Processed frames: {result.num_frames}")
            logger.info(f"  Average FPS: {result.num_frames / result.total_time:.2f}")

            # Calculate detection statistics
            total_detections = 0
            all_classes = set()

            for frame in result.frames_results:
                detections = frame.results["detections"]
                total_detections += len(detections)
                all_classes.update(det["class"] for det in detections)

            avg_detections = total_detections / result.num_frames
            logger.info(f"  Total detections: {total_detections}")
            logger.info(f"  Average detections per frame: {avg_detections:.2f}")
            logger.info(f"  Detected classes: {all_classes}")
            logger.info(f"  Output saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")


if __name__ == "__main__":
    main()
