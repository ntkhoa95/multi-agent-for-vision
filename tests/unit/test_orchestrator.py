import os
from pathlib import Path

import pytest

from vision_framework import VisionOrchestrator, VisionTaskType


def setup_test_image(image_url, image_path):
    """Setup test image for orchestrator tests."""
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if not os.path.exists(image_path):
        os.system(f"wget -O {image_path} {image_url}")


@pytest.fixture
def config():
    return {
        "YOLO_MODEL_NAME": "yolov8s.pt",
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.45,
        "DETECTION_IMAGE_SIZE": 640,
        "ENABLE_TRACK": True,
        "device_type": "cpu",  # or 'cuda' if you have a GPU
    }


@pytest.fixture
def orchestrator(config):
    return VisionOrchestrator(config)


def test_list_agents(orchestrator):
    agents = orchestrator.list_agents()
    assert VisionTaskType.IMAGE_CLASSIFICATION in agents
    assert VisionTaskType.OBJECT_DETECTION in agents


@pytest.mark.skip(reason="API change - method moved to detector")
def test_annotate_image(orchestrator):
    # Set up test image
    image_path = "tests/data/images/dog.jpg"
    setup_test_image(
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/dog.jpg",
        image_path,
    )

    result = orchestrator.process_image(image_path, "find dog")
    annotated = orchestrator.annotate_image(image_path, result.results["detections"])
    assert annotated is not None
    assert isinstance(annotated, Path)
    assert annotated.exists()


def test_visualize_detections(orchestrator, tmp_path):
    image_path = "tests/data/images/dog.jpg"
    setup_test_image(
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/dog.jpg",
        image_path,
    )

    result = orchestrator.process_image(image_path, "detect objects")
    output_path = tmp_path / "output.jpg"
    annotated_image = orchestrator.visualize_detections(
        image_path, result.results["detections"], output_path
    )
    assert isinstance(annotated_image, bool)
    assert output_path.exists()
