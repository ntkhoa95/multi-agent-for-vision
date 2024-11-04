import os

import pytest

from vision_framework import VisionOrchestrator, VisionTaskType


@pytest.fixture
def orchestrator():
    config = {
        "YOLO_MODEL_NAME": "yolov8s.pt",
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.45,
        "DETECTION_IMAGE_SIZE": 640,
        "ENABLE_TRACK": True,
        "device_type": "cpu",  # or 'cuda' if you have a GPU
    }
    return VisionOrchestrator(config)


def setup_test_image(image_url, image_path):
    """Setup test image for classification tests."""
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if not os.path.exists(image_path):
        os.system(f"wget -O {image_path} {image_url}")


def test_image_classification(orchestrator):
    image_path = "tests/data/images/street.jpg"
    setup_test_image(
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
        image_path,
    )

    result = orchestrator.process_image(image_path=image_path, user_comment="classify this image")

    assert result.task_type == VisionTaskType.IMAGE_CLASSIFICATION
    assert "top_predictions" in result.results
    assert len(result.results["top_predictions"]) == 5
    assert result.confidence > 0.0
    assert result.processing_time > 0.0


def test_classification_results_format(orchestrator):
    image_path = "tests/data/images/street.jpg"
    setup_test_image(
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
        image_path,
    )

    result = orchestrator.process_image(
        image_path=image_path,
        user_comment="what is in this image",
        task_type=VisionTaskType.IMAGE_CLASSIFICATION,  # Explicitly set task type
    )

    assert result.task_type == VisionTaskType.IMAGE_CLASSIFICATION
    assert "model_config" in result.results
    assert "top_predictions" in result.results
    predictions = result.results["top_predictions"]

    # Verify predictions format
    for pred in predictions:
        assert "class" in pred
        assert "confidence" in pred
        assert isinstance(pred["confidence"], float)
        assert 0 <= pred["confidence"] <= 1
