from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from vision_framework.agents.classification import MobileNetClassificationAgent
from vision_framework.agents.detection import YOLODetectionAgent
from vision_framework.core.types import (
    BatchDetectionResult,
    VideoDetectionResult,
    VisionInput,
    VisionOutput,
    VisionTaskType,
)
from vision_framework.orchestrator import VisionOrchestrator
from vision_framework.router.router import AgentRouter


@pytest.fixture
def config():
    return {
        "YOLO_MODEL_NAME": "yolov8s.pt",
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.45,
        "DETECTION_IMAGE_SIZE": 640,
        "ENABLE_TRACK": True,
        "BATCH_SIZE": 32,
        "device_type": "cpu",  # or 'cuda' if you have a GPU
    }


@pytest.fixture
def orchestrator(config):
    return VisionOrchestrator(config)


def test_initialize_agents(orchestrator):
    assert isinstance(
        orchestrator.router.agents[VisionTaskType.IMAGE_CLASSIFICATION],
        MobileNetClassificationAgent,
    )
    assert isinstance(
        orchestrator.router.agents[VisionTaskType.OBJECT_DETECTION], YOLODetectionAgent
    )


def test_filter_detections(orchestrator):
    detections = [
        {"class": "person", "bbox": [0, 0, 10, 10], "confidence": 0.9},
        {"class": "car", "bbox": [10, 10, 20, 20], "confidence": 0.8},
    ]
    allowed_classes = ["person"]
    filtered = orchestrator.filter_detections(detections, allowed_classes)
    assert len(filtered) == 1
    assert filtered[0]["class"] == "person"


def test_visualize_detections(orchestrator):
    image_path = "tests/data/images/street.jpg"
    detections = [
        {"class": "person", "bbox": [0, 0, 10, 10], "confidence": 0.9},
        {"class": "car", "bbox": [10, 10, 20, 20], "confidence": 0.8},
    ]
    annotated_image = orchestrator.visualize_detections(image_path, detections)
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[0] > 0 and annotated_image.shape[1] > 0


@patch.object(AgentRouter, "process_request")
def test_process_image(mock_process_request, orchestrator):
    mock_process_request.return_value = VisionOutput(
        task_type=VisionTaskType.OBJECT_DETECTION,
        results={"detections": [], "num_detections": 0},
        confidence=0.0,
        processing_time=0.1,
    )
    result = orchestrator.process_image("tests/data/images/street.jpg", "detect objects")
    assert result.task_type == VisionTaskType.OBJECT_DETECTION
    assert "detections" in result.results


@patch.object(YOLODetectionAgent, "process_batch")
def test_process_batch(mock_process_batch, orchestrator):
    mock_process_batch.return_value = [
        BatchDetectionResult(
            "tests/data/images/street.jpg",
            VisionOutput(
                task_type=VisionTaskType.OBJECT_DETECTION,
                results={"detections": [], "num_detections": 0},
                confidence=0.0,
                processing_time=0.1,
            ),
        )
    ]
    results = orchestrator.process_batch(
        ["tests/data/images/street.jpg"], VisionTaskType.OBJECT_DETECTION
    )
    assert len(results) == 1
    assert results[0].vision_output.task_type == VisionTaskType.OBJECT_DETECTION


@patch.object(YOLODetectionAgent, "process_video")
def test_process_video(mock_process_video, orchestrator):
    mock_process_video.return_value = VideoDetectionResult(
        video_path="tests/data/videos/crosswalk.avi",
        frames_results=[],
        fps=30.0,
        total_time=10.0,
    )
    result = orchestrator.process_video("tests/data/videos/crosswalk.avi")
    assert result.video_path == "tests/data/videos/crosswalk.avi"
    assert result.fps == 30.0
    assert result.total_time == 10.0


def test_process_batch_no_agent(orchestrator):
    with patch.object(VisionTaskType, "IMAGE_SEGMENTATION", create=True):
        with pytest.raises(ValueError):
            orchestrator.process_batch(
                ["tests/data/images/street.jpg"], VisionTaskType.IMAGE_SEGMENTATION
            )


def test_process_video_no_agent(orchestrator):
    with patch.object(VisionTaskType, "IMAGE_SEGMENTATION", create=True):
        with patch.object(
            AgentRouter,
            "determine_task_type",
            return_value=(VisionTaskType.IMAGE_SEGMENTATION, None),
        ):
            with pytest.raises(ValueError):
                orchestrator.process_video(
                    "tests/data/videos/crosswalk.avi", user_comment="segment video"
                )


def test_process_video_no_support(orchestrator):
    with patch.object(
        AgentRouter,
        "determine_task_type",
        return_value=(VisionTaskType.IMAGE_CLASSIFICATION, None),
    ):
        with pytest.raises(ValueError):
            orchestrator.process_video(
                "tests/data/videos/crosswalk.avi", user_comment="classify video"
            )
