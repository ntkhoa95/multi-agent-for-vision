import logging
from pathlib import Path

import torch

from vision_framework import VisionOrchestrator, VisionTaskType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config():
    """Get basic configuration"""
    return {
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "MODEL_NAME": "mobilenetv3_large_100",
        "MODEL_PRETRAINED": True,
        "BATCH_SIZE": 1,
        "NUM_WORKERS": 0,
        "YOLO_MODEL_NAME": "yolov8s.pt",  # Added for detection
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.45,
    }


def print_results(result):
    """Print results based on task type"""
    if result.task_type == VisionTaskType.IMAGE_CLASSIFICATION:
        logger.info("Classification Results:")
        if "top_predictions" in result.results:
            for pred in result.results["top_predictions"]:
                logger.info(f"  {pred['class']}: {pred['confidence']:.3f}")

            # Print model configuration details
            if "model_config" in result.results:
                config = result.results["model_config"]
                logger.info("\nModel Configuration:")
                logger.info(f"  Model: {config['name']}")
                logger.info(f"  Input size: {config['input_size']}")
                logger.info(f"  Interpolation: {config['interpolation']}")
                logger.info(f"  Mean: {config['mean']}")
                logger.info(f"  Std: {config['std']}")
        else:
            logger.info("No classification predictions available")

    elif result.task_type == VisionTaskType.OBJECT_DETECTION:
        logger.info("Detection Results:")
        for det in result.results["detections"]:
            logger.info(f"  {det['class']}: {det['confidence']:.3f}")

    logger.info(f"Processing time: {result.processing_time:.3f} seconds")
    logger.info(f"Overall confidence: {result.confidence:.3f}")


def main():
    # Initialize framework
    config = get_config()
    orchestrator = VisionOrchestrator(config)
    logger.info(f"Using device: {config['DEVICE']}")

    # Example images
    image_paths = [
        "tests/data/images/bus.jpg",  # Updated path to match your structure
        "tests/data/images/dog.jpg",
    ]

    # Test queries with explicit task types
    test_cases = [
        ("What is in this image?", VisionTaskType.IMAGE_CLASSIFICATION),
        ("Classify this image", VisionTaskType.IMAGE_CLASSIFICATION),
        (
            "Identify objects in this image",
            VisionTaskType.OBJECT_DETECTION,
        ),  # Changed to detection
    ]

    for image_path in image_paths:
        logger.info(f"\nProcessing image: {image_path}")

        for query, task_type in test_cases:
            try:
                # Process image with explicit task type
                result = orchestrator.process_image(
                    image_path=image_path,
                    user_comment=query,
                    # task_type=task_type  # Explicitly set task type
                )

                # Print results
                logger.info(f"\nQuery: {query}")
                logger.info(f"Task Type: {result.task_type}")
                print_results(result)

            except Exception as e:
                logger.error(f"Error processing query '{query}': {str(e)}")
                logger.debug("Error details:", exc_info=True)


def verify_paths(image_paths):
    """Verify that all image paths exist"""
    for path in image_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Image not found: {path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error running example: {str(e)}")
        raise
