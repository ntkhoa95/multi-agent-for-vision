import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.append(str(Path(__file__).parent.parent))
from vision_framework import VisionOrchestrator, VisionTaskType

# Setup logging with more detail
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_config() -> Dict:
    """Get configuration for the vision framework."""
    return {
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "MODEL_NAME": "mobilenetv3_large_100",
        "MODEL_PRETRAINED": True,
        "BATCH_SIZE": 1,
        "NUM_WORKERS": 0,
        "USE_FP16": False,  # Add FP16 option for faster inference
        # Optional detection config (in case detection agent is loaded)
        "YOLO_MODEL_NAME": "yolov8s.pt",
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.45,
    }


def print_results(result) -> None:
    """Print vision processing results in a structured format."""
    logger.info("-" * 50)
    logger.info(f"Task Type: {result.task_type}")
    logger.info(f"Processing Time: {result.processing_time:.3f} seconds")
    logger.info(f"Overall Confidence: {result.confidence:.3f}")

    if result.task_type == VisionTaskType.IMAGE_CLASSIFICATION:
        if "top_predictions" in result.results:
            logger.info("\nTop Predictions:")
            for i, pred in enumerate(result.results["top_predictions"], 1):
                logger.info(f"  {i}. {pred['class']}: {pred['confidence']:.3f}")

            if "model_config" in result.results:
                config = result.results["model_config"]
                logger.info("\nModel Configuration:")
                logger.info(f"  Model: {config['name']}")
                logger.info(f"  Input size: {config['input_size']}")
                logger.info(f"  Interpolation: {config['interpolation']}")
                logger.info(f"  Mean: {[f'{x:.3f}' for x in config['mean']]}")
                logger.info(f"  Std: {[f'{x:.3f}' for x in config['std']]}")
        else:
            logger.warning("No classification predictions available")

    logger.info("-" * 50)


def verify_paths(image_paths: List[str]) -> None:
    """Verify all image paths exist and are readable."""
    for path in image_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        # Try to open the file to verify read permissions
        try:
            path.open("rb").close()
        except PermissionError:
            raise PermissionError(f"Cannot read file: {path}")


def get_test_cases() -> List[Tuple[str, VisionTaskType]]:
    """Define test cases for classification."""
    return [
        ("What is in this image?", VisionTaskType.IMAGE_CLASSIFICATION),
        ("Classify this image", VisionTaskType.IMAGE_CLASSIFICATION),
        ("Identify the main object", VisionTaskType.IMAGE_CLASSIFICATION),
    ]


def process_image(
    orchestrator: VisionOrchestrator, image_path: str, query: str, task_type: VisionTaskType
) -> None:
    """Process a single image with error handling."""
    try:
        logger.info(f"\nProcessing: {image_path}")
        logger.info(f"Query: {query}")

        result = orchestrator.process_image(
            image_path=image_path, user_comment=query, task_type=task_type
        )
        print_results(result)

    except Exception as e:
        logger.error(f"Error processing '{image_path}' with query '{query}': {str(e)}")
        logger.debug("Error details:", exc_info=True)


def main():
    """Main execution function."""
    # Initialize framework
    config = get_config()
    logger.info(f"Initializing with device: {config['DEVICE']}")

    try:
        orchestrator = VisionOrchestrator(config)

        # Define test images
        image_paths = [
            "tests/data/images/bus.jpg",
            "tests/data/images/dog.jpg",
            "tests/data/images/street.jpg",
        ]

        # Verify paths before processing
        verify_paths(image_paths)

        # Get test cases
        test_cases = get_test_cases()

        # Process each image with each test case
        for image_path in image_paths:
            for query, task_type in test_cases:
                process_image(orchestrator, image_path, query, task_type)

    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        logger.debug("Error details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
