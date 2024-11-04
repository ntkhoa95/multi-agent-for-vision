import os
from unittest.mock import MagicMock

import pytest

from vision_framework.core.types import (
    BatchDetectionResult,
    VideoDetectionResult,
    VisionInput,
    VisionTaskType,
)
from vision_framework.nlp.processor import NLPProcessor
from vision_framework.orchestrator import VisionOrchestrator


def setup_test_image(image_url, image_path):
    """Setup test image for nlp tests."""
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if not os.path.exists(image_path):
        os.system(f"wget -O {image_path} {image_url}")


def setup_test_video(video_url, video_path):
    """Setup test video for nlp tests."""
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    if not os.path.exists(video_path):
        os.system(f"wget -O {video_path} {video_url}")


@pytest.fixture
def config():
    return {"BATCH_SIZE": 16}  # Example config, modify as needed


@pytest.fixture
def orchestrator(config):
    return VisionOrchestrator(config)


def test_agent_registration(orchestrator):
    """Ensure agents are registered correctly."""
    assert VisionTaskType.IMAGE_CLASSIFICATION in orchestrator.router.agents
    assert VisionTaskType.OBJECT_DETECTION in orchestrator.router.agents


def test_process_image(orchestrator):
    """Test processing a single image with classification."""
    # Set up test image
    image_path = "tests/data/images/dog.jpg"
    setup_test_image(
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/dog.jpg",
        image_path,
    )

    # Mock the classification agent's process function
    classification_agent = orchestrator.router.agents[VisionTaskType.IMAGE_CLASSIFICATION]
    classification_agent.process = MagicMock(return_value="mocked_output")

    result = orchestrator.process_image(
        image_path,
        "classify this image",
        VisionTaskType.IMAGE_CLASSIFICATION,
    )
    classification_agent.process.assert_called_once()  # Verify the agent's process method was called
    assert result == "mocked_output"


def test_process_image_invalid_task(orchestrator):
    """Test processing image with invalid task type."""
    # Set up test image
    image_path = "tests/data/images/dog.jpg"
    setup_test_image(
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/dog.jpg",
        image_path,
    )

    with pytest.raises(ValueError) as excinfo:
        orchestrator.process_image(image_path, "classify this image", "invalid_task")
    assert "No agent registered for task type" in str(excinfo.value)


def test_process_batch(orchestrator):
    """Test batch processing with mocked detection agent."""
    # Set up test images
    dog_path = "tests/data/images/dog.jpg"
    setup_test_image(
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/dog.jpg",
        dog_path,
    )
    bus_path = "tests/data/images/street.jpg"
    setup_test_image(
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
        bus_path,
    )

    # Mock detection agent's process_batch function
    detection_agent = orchestrator.router.agents[VisionTaskType.OBJECT_DETECTION]
    detection_agent.process_batch = MagicMock(return_value=["mocked_batch_output"])

    result = orchestrator.process_batch(
        [dog_path, bus_path],
        VisionTaskType.OBJECT_DETECTION,
    )
    detection_agent.process_batch.assert_called_once()  # Ensure process_batch was called
    assert result == ["mocked_batch_output"]


def test_process_batch_no_batch_support(orchestrator):
    """Test batch processing when agent does not support it."""
    # Set up test image
    image_path = "tests/data/images/dog.jpg"
    setup_test_image(
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/dog.jpg",
        image_path,
    )

    # Mock a classification agent without batch support
    classification_agent = orchestrator.router.agents[VisionTaskType.IMAGE_CLASSIFICATION]
    classification_agent.process = MagicMock(return_value="mocked_single_output")

    result = orchestrator.process_batch([image_path], VisionTaskType.IMAGE_CLASSIFICATION)
    assert len(result) == 1
    assert isinstance(result[0], BatchDetectionResult)


def test_process_video(orchestrator):
    """Test processing a video with detection."""
    # Set up test video
    video_path = "tests/data/videos/crosswalk.avi"
    setup_test_video(
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/videos/human-cropped.mp4",
        video_path,
    )

    # Mock detection agent's process_video function
    detection_agent = orchestrator.router.agents[VisionTaskType.OBJECT_DETECTION]
    detection_agent.process_video = MagicMock(return_value="mocked_video_output")

    result = orchestrator.process_video(video_path, user_comment="detect objects")
    detection_agent.process_video.assert_called_once()  # Ensure process_video was called
    assert result == "mocked_video_output"


def test_process_video_no_video_support(orchestrator):
    """Test video processing when agent does not support video processing."""
    # Set up test video
    video_path = "tests/data/videos/crosswalk.avi"
    setup_test_video(
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/videos/human-cropped.mp4",
        video_path,
    )

    classification_agent = orchestrator.router.agents[VisionTaskType.IMAGE_CLASSIFICATION]

    with pytest.raises(ValueError) as excinfo:
        orchestrator.process_video(video_path, user_comment="classify video")
    assert "does not support video processing" in str(excinfo.value)


def test_nlp_processor_with_orchestrator(orchestrator, nlp_processor):
    # Set up test image
    image_path = "tests/data/images/dog.jpg"
    setup_test_image(
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/dog.jpg",
        image_path,
    )

    query = "detect cats and dogs in this image"
    task_type, objects = nlp_processor.parse_query(query)

    # Convert task_type string to VisionTaskType if needed
    if task_type == "detection":
        task_type = VisionTaskType.OBJECT_DETECTION
    elif task_type == "classification":
        task_type = VisionTaskType.IMAGE_CLASSIFICATION

    result = orchestrator.process_image(
        image_path,
        query,
        task_type=task_type,
        additional_params={"objects": objects},
    )

    # Add assertions for expected results
    assert result  # Replace with specific expected result assertions


@pytest.fixture
def nlp_processor():
    return NLPProcessor()


def test_query_preprocessing(nlp_processor):
    query = "Find Dogs and Cats in the Image!"
    processed = nlp_processor.preprocess_query(query)
    assert processed == "find dogs and cats in the image"


def test_task_type_extraction(nlp_processor):
    queries = {
        "detect cats": "detection",
        "classify this image": "classification",
        "find people": "detection",
        "what is in this image": "classification",
        "identify objects": "detection",
    }

    for query, expected_task in queries.items():
        task_type, _ = nlp_processor.parse_query(query)
        assert task_type == expected_task, f"Failed for query: {query}"


def test_object_extraction(nlp_processor):
    queries = {
        "detect cats and dogs": {"cat", "dog"},
        "find people walking": {"person"},
        "locate cars and trucks": {"car", "truck"},
        "detect all objects": set(),  # Expect an empty set instead of None
    }

    for query, expected_objects in queries.items():
        _, objects = nlp_processor.parse_query(query)
        normalized_objects = {nlp_processor.normalize_object_name(obj) for obj in objects}

        if expected_objects is None:
            # If None was expected, verify the objects set is empty
            assert (
                not normalized_objects
            ), f"Expected no objects, but got {normalized_objects} for query: {query}"
        else:
            assert normalized_objects == expected_objects, f"Failed for query: {query}"
