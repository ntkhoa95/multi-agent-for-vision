import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from vision_framework import VisionOrchestrator, VisionTaskType


def download_test_image():
    """Download a test image if not exists."""
    import urllib.request

    image_dir = Path("tests/data/images")
    image_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "bus.jpg"
    if not image_path.exists():
        logger.info(f"Downloading test image to {image_path}")
        url = "[MASKED]/ultralytics/assets/raw/main/yolo/bus.jpg"
        urllib.request.urlretrieve(url, image_path)
    return image_path


def main():
    # Install required packages if needed
    try:
        import accelerate
        import torch
        import transformers
    except ImportError:
        logger.info("Required packages not installed. Installing...")
        import subprocess

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "transformers>=4.35.0",
                "accelerate>=0.26.0",
                "safetensors>=0.4.0",
            ]
        )
        logger.info("Packages installed successfully!")

        # Re-import after installation
        import accelerate
        import torch
        import transformers

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
    }

    # Initialize orchestrator
    logger.info("Initializing VisionOrchestrator...")
    orchestrator = VisionOrchestrator(config)

    # Download test image
    image_path = download_test_image()
    logger.info(f"Using test image: {image_path}")

    # Example 1: Detection with captioning
    logger.info("\nExample 1: Detection with automatic captioning")
    logger.info("-" * 50)

    try:
        # Be explicit about wanting only people
        detection_result, caption_result = orchestrator.process_image_with_caption(
            image_path=str(image_path),
            user_comment="detect people",
            task_type=VisionTaskType.OBJECT_DETECTION,
        )

        # Only print filtered detections
        logger.info("\nDetection Results (People only):")
        # for detection in detection_result.results.get("detections", []):
        #     if detection['class'].lower() == 'person':
        #         logger.info(
        #             f"- Found person with confidence {detection['confidence']:.2f}"
        #         )

        if caption_result and caption_result.results.get("caption"):
            logger.info("\nGenerated Caption:")
            logger.info(caption_result.results["caption"])
        else:
            logger.warning("No caption was generated")

        # Save annotated image with filtered detections
        output_path = Path("examples/output/annotated_street.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Only visualize people detections
        filtered_detections = [
            det
            for det in detection_result.results["detections"]
            if det["class"].lower() == "person"
        ]

        orchestrator.visualize_detections(
            str(image_path),
            filtered_detections,  # Use filtered detections
            str(output_path),
            caption=caption_result.results.get("caption") if caption_result else None,
        )
        logger.info(f"\nAnnotated image saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error in detection+captioning: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
