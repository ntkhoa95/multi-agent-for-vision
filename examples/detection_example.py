import logging
import os
from pathlib import Path

import cv2
import torch

from vision_framework import VisionOrchestrator, VisionTaskType

# Setup logging
logging.basicConfig(level=logging.INFO)
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

    # Test different detection queries
    detection_queries = [
        ("detect all objects", None),
        ("find people", ["person"]),
        ("detect cars and trucks", ["car", "truck"]),
        ("find people and cars", ["person", "car"]),
    ]

    for image_path in image_paths:
        logger.info(f"\nProcessing image: {image_path}")

        for query, target_classes in detection_queries:
            try:
                # Process image
                result = orchestrator.process_image(
                    image_path=image_path, user_comment=query
                )

                # Get detection agent for visualization
                detection_agent = orchestrator.router.agents[
                    VisionTaskType.OBJECT_DETECTION
                ]

                # Save visualization
                output_name = (
                    f"detection_{Path(image_path).stem}_{query.replace(' ', '_')}.jpg"
                )
                output_path = f"examples/output/{output_name}"

                detection_agent.visualize_detections(
                    image=image_path,
                    detections=result.results["detections"],
                    output_path=output_path,
                )

                # Print results
                logger.info(f"\nQuery: {query}")
                if target_classes:
                    logger.info(f"Target classes: {target_classes}")

                logger.info("Detections:")
                for det in result.results["detections"]:
                    logger.info(f"  {det['class']}: {det['confidence']:.3f}")

                logger.info(f"Visualization saved to: {output_path}")
                logger.info(f"Processing time: {result.processing_time:.3f} seconds")

            except Exception as e:
                logger.error(f"Error processing query '{query}': {str(e)}")


if __name__ == "__main__":
    main()
