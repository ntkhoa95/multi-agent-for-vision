import logging
import os
import sys
from pathlib import Path

import cv2
import torch

sys.path.append(str(Path(__file__).parent.parent))
from vision_framework import VisionOrchestrator, VisionTaskType

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_config():
    """Get detection configuration"""
    return {
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "YOLO_MODEL_NAME": "yolov8s.pt",
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.45,
        "DETECTION_IMAGE_SIZE": 640,
        "BATCH_SIZE": 1,
        "NUM_WORKERS": 0,
    }


def process_detection(
    orchestrator: VisionOrchestrator, image_path: str, query: str, target_classes: list = None
) -> None:
    """Process a single detection with caption."""
    try:
        # Process image for detection
        detection_result = orchestrator.process_image(
            image_path=image_path,
            user_comment=query,
            task_type=VisionTaskType.OBJECT_DETECTION,
            additional_params={"detect_classes": target_classes},
        )

        # Generate caption if captioning is enabled
        caption = None
        if VisionTaskType.IMAGE_CAPTIONING in orchestrator.router.agents:
            try:
                caption_result = orchestrator.process_image(
                    image_path=image_path,
                    user_comment="Describe this image",
                    task_type=VisionTaskType.IMAGE_CAPTIONING,
                )
                caption = caption_result.results.get("caption")
            except Exception as e:
                logger.warning(f"Failed to generate caption: {str(e)}")

        # Get detection agent and visualize results
        detection_agent = orchestrator.router.agents[VisionTaskType.OBJECT_DETECTION]

        # Create output filename
        output_name = f"detection_{Path(image_path).stem}_{query.replace(' ', '_')}.jpg"
        output_path = f"examples/output/{output_name}"

        # Visualize with caption
        detection_agent.visualize_detections(
            image=image_path,
            detections=detection_result.results["detections"],
            output_path=output_path,
            caption=caption,
        )

        # Print results
        logger.info(f"\nQuery: {query}")
        if target_classes:
            logger.info(f"Target classes: {target_classes}")

        logger.info("Detections:")
        for det in detection_result.results["detections"]:
            logger.info(f"  {det['class']}: {det['confidence']:.3f}")

        if caption:
            logger.info(f"Caption: {caption}")

        logger.info(f"Visualization saved to: {output_path}")
        logger.info(f"Processing time: {detection_result.processing_time:.3f} seconds")

    except Exception as e:
        logger.error(f"Error processing query '{query}': {str(e)}")
        logger.debug("Error details:", exc_info=True)


def main():
    # Initialize framework
    config = get_config()
    orchestrator = VisionOrchestrator(config)
    logger.info(f"Using device: {config['DEVICE']}")

    # Ensure output directory exists
    os.makedirs("examples/output", exist_ok=True)

    # Example images
    image_paths = [
        "tests/data/images/street.jpg",
        "tests/data/images/bus.jpg",
    ]

    # Test different detection queries with explicit class lists
    detection_queries = [
        ("detect all objects", None),
        ("find people", ["person"]),
        ("detect cars and trucks", ["car", "truck"]),
        ("find people and cars", ["person", "car"]),
    ]

    # Verify YOLO model and available classes
    try:
        detection_agent = orchestrator.router.agents[VisionTaskType.OBJECT_DETECTION]
        available_classes = detection_agent.get_available_classes()
        logger.info(f"Available detection classes: {sorted(list(available_classes))}")
    except Exception as e:
        logger.error(f"Error getting available classes: {str(e)}")
        return

    # Process images
    for image_path in image_paths:
        if not Path(image_path).exists():
            logger.error(f"Image not found: {image_path}")
            continue

        logger.info(f"\nProcessing image: {image_path}")

        for query, target_classes in detection_queries:
            process_detection(
                orchestrator=orchestrator,
                image_path=image_path,
                query=query,
                target_classes=target_classes,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        logger.debug("Error details:", exc_info=True)
        sys.exit(1)
