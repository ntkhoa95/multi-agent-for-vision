import logging
import os
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from vision_framework import VisionOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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


def print_video_info(video_path: str) -> bool:
    """Print video information"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info("Video Information:")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Frame count: {frame_count}")
        logger.info(f"  Duration: {frame_count/fps:.2f} seconds")

        return True
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        return False
    finally:
        if "cap" in locals():
            cap.release()


def process_video_query(
    orchestrator: VisionOrchestrator,
    video_path: str,
    query: str,
    target_classes: list,
    output_path: str,
    duration: float = 10.0,
):
    """Process a single video query with error handling"""
    try:
        logger.info(f"\nProcessing with query: {query}")
        if target_classes:
            logger.info(f"Target classes: {target_classes}")

        # Process video
        video_result, frame_captions = orchestrator.process_video(
            video_path=video_path,
            user_comment=query,
            output_path=output_path,
            start_time=0,
            end_time=duration,
        )

        # Print statistics
        logger.info("\nProcessing Results:")
        logger.info(f"  Processed frames: {video_result.num_frames}")
        logger.info(f"  Total processing time: {video_result.total_time:.2f} seconds")
        logger.info(f"  Average FPS: {video_result.num_frames / video_result.total_time:.2f}")

        # Calculate detection statistics
        total_detections = 0
        all_classes = set()
        tracked_objects = set()

        for frame in video_result.frames_results:
            detections = frame.results.get("detections", [])
            total_detections += len(detections)

            for det in detections:
                all_classes.add(det["class"])
                if "track_id" in det:
                    tracked_objects.add((det["class"], det["track_id"]))

        avg_detections = (
            total_detections / video_result.num_frames if video_result.num_frames > 0 else 0
        )

        logger.info(f"  Total detections: {total_detections}")
        logger.info(f"  Average detections per frame: {avg_detections:.2f}")
        logger.info(f"  Detected classes: {sorted(all_classes)}")

        if tracked_objects:
            logger.info(f"  Unique tracked objects: {len(tracked_objects)}")
            for obj_class, track_id in sorted(tracked_objects):
                logger.info(f"    {obj_class} (ID: {track_id})")

        logger.info(f"  Output saved to: {output_path}")

        if frame_captions:
            logger.info(f"  Generated {len(frame_captions)} frame captions")

    except Exception as e:
        logger.error(f"Error processing query '{query}': {str(e)}")
        logger.debug("Error details:", exc_info=True)


def main():
    try:
        # Initialize framework
        config = get_config()
        orchestrator = VisionOrchestrator(config)
        logger.info(f"Using device: {config['DEVICE']}")

        # Ensure output directory exists
        os.makedirs("examples/output", exist_ok=True)

        # Example video
        video_path = "tests/data/videos/crosswalk.avi"
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return

        logger.info(f"\nProcessing video: {video_path}")
        if not print_video_info(video_path):
            return

        # Test different video processing queries
        video_queries = [
            ("detect all objects in video", None),
            ("track people", ["person"]),
            ("detect cars", ["car"]),  # Removed 'track' from query
            ("find people and vehicles", ["person", "car", "truck", "bus"]),
        ]

        for query, target_classes in video_queries:
            output_name = f"output_{Path(video_path).stem}_{query.replace(' ', '_')}.mp4"
            output_path = f"examples/output/{output_name}"

            process_video_query(
                orchestrator=orchestrator,
                video_path=video_path,
                query=query,
                target_classes=target_classes,
                output_path=output_path,
            )

    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        logger.debug("Error details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
