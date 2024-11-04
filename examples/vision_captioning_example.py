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
        url = "https://raw.githubusercontent.com/ultralytics/assets/main/yolo/bus.jpg"
        urllib.request.urlretrieve(url, image_path)
    return image_path


def setup_required_packages():
    """Install and import required packages."""
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

    return torch.cuda.is_available()


def analyze_human_behavior(image_path: str, orchestrator: VisionOrchestrator):
    """Analyze human behavior in the image using detection and captioning."""
    try:
        # First detect humans
        logger.info("\nStep 1: Detecting humans in the image")
        logger.info("-" * 50)

        # Debug available agents
        logger.info(f"Available agents: {orchestrator.list_agents()}")
        logger.info(f"Available task types: {[t.value for t in VisionTaskType]}")

        detection_result, caption_result = orchestrator.process_image_with_caption(
            image_path=str(image_path),
            user_comment="detect human in this image",
            task_type=VisionTaskType.OBJECT_DETECTION,
        )

        # Filter and display human detections
        human_detections = [
            det
            for det in detection_result.results.get("detections", [])
            if det["class"].lower() in ["person", "people"]
        ]

        if not human_detections:
            logger.info("No humans detected in the image.")
            return

        logger.info("\nHuman Detection Results:")
        for i, detection in enumerate(human_detections, 1):
            logger.info(f"- Human {i} detected with confidence {detection['confidence']:.2f}")

        # Generate detailed caption focusing on human behavior
        logger.info("\nStep 2: Analyzing human behavior")
        logger.info("-" * 50)

        # Debug task type before calling
        task_type = VisionTaskType.IMAGE_CAPTIONING
        logger.info(f"Using task type: {task_type} ({task_type.value})")

        behavior_caption = orchestrator.process_image(
            image_path=str(image_path),
            user_comment="Describe what the people in this image are doing",
            task_type=task_type,
        )

        if behavior_caption and behavior_caption.results.get("caption"):
            logger.info("\nBehavior Analysis:")
            logger.info(behavior_caption.results["caption"])
        else:
            logger.warning("Could not generate behavior description")

        # Visualize results
        output_path = Path("examples/output/human_analysis.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        orchestrator.visualize_detections(str(image_path), human_detections, str(output_path))
        logger.info(f"\nAnnotated image saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error in human behavior analysis: {str(e)}", exc_info=True)


def main():
    # Setup packages and check GPU
    has_gpu = setup_required_packages()
    device = "cuda" if has_gpu else "cpu"
    logger.info(f"Using device: {device}")

    # Configuration with focus on human detection
    config = {
        "ENABLE_CAPTIONING": True,
        "YOLO_MODEL_NAME": "yolov8s.pt",
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,  # Lower threshold for better human detection
        "YOLO_IOU_THRESHOLD": 0.45,
        "DETECTION_IMAGE_SIZE": 640,
        "ENABLE_TRACK": True,
        "DEVICE": device,
        "MAX_CAPTION_LENGTH": 100,  # Longer captions for better behavior description
        "MIN_CAPTION_LENGTH": 20,
        "GIT_MODEL_NAME": "nlpconnect/vit-gpt2-image-captioning",
        "NUM_BEAMS": 3,
    }

    # Initialize orchestrator
    logger.info("Initializing VisionOrchestrator...")
    orchestrator = VisionOrchestrator(config)

    # Download test image
    image_path = download_test_image()
    logger.info(f"Using test image: {image_path}")

    # Analyze human behavior in the image
    analyze_human_behavior(image_path, orchestrator)

    # Clear GPU memory if available
    orchestrator.clear_gpu_memory()


if __name__ == "__main__":
    main()
