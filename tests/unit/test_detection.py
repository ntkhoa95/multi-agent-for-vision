import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from vision_framework.agents.detection import YOLODetectionAgent
from vision_framework.core.types import (
    BatchDetectionResult,
    VideoDetectionResult,
    VisionInput,
    VisionOutput,
    VisionTaskType,
)


@pytest.fixture
def config():
    return {
        "YOLO_MODEL_NAME": "yolov8s.pt",
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.45,
        "DETECTION_IMAGE_SIZE": 640,
        "ENABLE_TRACK": True,
        "DEVICE": "cpu",  # or 'cuda' if you have a GPU
    }


@pytest.fixture
def agent(config):
    return YOLODetectionAgent(config)


def test_initialize_agent(agent):
    assert agent.conf_threshold == 0.25
    assert agent.iou_threshold == 0.45
    assert agent.image_size == 640
    assert agent.enable_tracking is True


@patch.object(YOLODetectionAgent, "load_model")
def test_get_available_classes(mock_load_model, agent):
    mock_model = MagicMock()
    mock_model.names = {0: "person", 1: "car"}
    agent.model = mock_model
    available_classes = agent.get_available_classes()
    assert available_classes == {"person", "car"}


@patch.object(YOLODetectionAgent, "load_model")
def test_validate_classes(mock_load_model, agent):
    mock_model = MagicMock()
    mock_model.names = {0: "person", 1: "car"}
    agent.model = mock_model
    valid_classes = agent.validate_classes(["person", "car", "unicorn"])
    assert valid_classes == ["person", "car"]


def test_get_class_indices(agent):
    agent.class_mapping = {"person": 0, "car": 1}
    indices = agent.get_class_indices(["person", "car", "unicorn"])
    assert indices == [0, 1]


@patch("vision_framework.agents.detection.YOLO")
def test_load_model(mock_yolo, agent):
    agent.load_model()
    mock_yolo.assert_called_once_with("yolov8s.pt")


def test_process_image(agent):
    mock_image = np.zeros((640, 640, 3))
    mock_detections = MagicMock()
    mock_detections.boxes = [
        MagicMock(
            cls=0, xyxy=[torch.tensor([50, 50, 100, 100])], conf=torch.tensor(0.9)
        )
    ]
    mock_detections.names = {0: "person"}
    agent.model.predict = MagicMock(return_value=[mock_detections])

    result = agent.process_image(mock_image, user_comment="detect human")
    assert "detections" in result.results
    assert len(result.results["detections"]) == 1
    assert result.results["detections"][0]["class"] == "person"


def test_process_image_with_detect_classes(agent):
    mock_image = np.zeros((640, 640, 3))
    mock_detections = MagicMock()
    mock_detections.boxes = [
        MagicMock(
            cls=0, xyxy=[torch.tensor([50, 50, 100, 100])], conf=torch.tensor(0.9)
        )
    ]
    mock_detections.names = {0: "person"}
    agent.model.predict = MagicMock(return_value=[mock_detections])

    result = agent.process_image(
        mock_image, detect_classes=["person"], user_comment="detect human"
    )
    assert "detections" in result.results
    assert len(result.results["detections"]) == 1
    assert result.results["detections"][0]["class"] == "person"


def test_process_detections(agent):
    mock_detections = MagicMock()
    mock_detections.boxes = [
        MagicMock(
            cls=0, xyxy=[torch.tensor([50, 50, 100, 100])], conf=torch.tensor(0.9)
        )
    ]
    mock_detections.names = {0: "person"}
    results = [mock_detections]

    detections = agent.process_detections(results)
    assert len(detections) == 1
    assert detections[0]["class"] == "person"


def test_filter_detections(agent):
    detections = [
        {"class": "person", "bbox": [0, 0, 10, 10], "confidence": 0.9},
        {"class": "car", "bbox": [10, 10, 20, 20], "confidence": 0.8},
    ]
    filtered = agent.filter_detections(detections, ["person"])
    assert len(filtered) == 1
    assert filtered[0]["class"] == "person"


def test_process(agent):
    mock_image = np.zeros((640, 640, 3))
    mock_detections = MagicMock()
    mock_detections.boxes = [
        MagicMock(
            cls=0, xyxy=[torch.tensor([50, 50, 100, 100])], conf=torch.tensor(0.9)
        )
    ]
    mock_detections.names = {0: "person"}
    agent.model.predict = MagicMock(return_value=[mock_detections])

    vision_input = VisionInput(
        image=mock_image,
        user_comment="detect human",
        additional_params={"detect_classes": ["person"]},
    )
    result = agent.process(vision_input)
    assert result.task_type == VisionTaskType.OBJECT_DETECTION
    assert "detections" in result.results
    assert len(result.results["detections"]) == 1
    assert result.results["detections"][0]["class"] == "person"


def test_process_no_detections(agent):
    mock_image = np.zeros((640, 640, 3))
    mock_detections = MagicMock()
    mock_detections.boxes = []
    mock_detections.names = {0: "person"}
    agent.model.predict = MagicMock(return_value=[mock_detections])

    vision_input = VisionInput(
        image=mock_image,
        user_comment="detect human",
        additional_params={"detect_classes": ["person"]},
    )
    result = agent.process(vision_input)
    assert result.task_type == VisionTaskType.OBJECT_DETECTION
    assert "detections" in result.results
    assert len(result.results["detections"]) == 0


@patch.object(YOLODetectionAgent, "load_image")
@patch.object(YOLODetectionAgent, "process_detections")
def test_process_batch(mock_process_detections, mock_load_image, agent):
    mock_image = np.zeros((640, 640, 3))
    mock_load_image.return_value = mock_image
    mock_process_detections.return_value = [
        {"class": "person", "bbox": [0, 0, 10, 10], "confidence": 0.9}
    ]
    agent.model.predict = MagicMock(return_value=[MagicMock()])

    results = agent.process_batch(["tests/data/images/street.jpg"])
    assert len(results) == 1
    assert results[0].vision_output.task_type == VisionTaskType.OBJECT_DETECTION


def test_process_batch_empty_image_paths(agent):
    results = agent.process_batch([])
    assert len(results) == 0


@patch.object(YOLODetectionAgent, "load_image")
def test_process_batch_invalid_image_paths(mock_load_image, agent):
    mock_load_image.side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError):
        agent.process_batch(["invalid_path.jpg"])


@patch.object(YOLODetectionAgent, "process_detections")
def test_process_batch_no_detections(mock_process_detections, agent):
    mock_image = np.zeros((640, 640, 3))
    mock_process_detections.return_value = []
    agent.model.predict = MagicMock(return_value=[MagicMock()])

    results = agent.process_batch(["tests/data/images/street.jpg"])
    assert len(results) == 1
    assert results[0].vision_output.task_type == VisionTaskType.OBJECT_DETECTION
    assert len(results[0].vision_output.results["detections"]) == 0


@patch.object(YOLODetectionAgent, "process_detections")
def test_process_video(mock_process_detections, agent):
    mock_process_detections.return_value = [
        {"class": "person", "bbox": [0, 0, 10, 10], "confidence": 0.9}
    ]
    agent.model.predict = MagicMock(return_value=[MagicMock()])

    vision_input = VisionInput(
        image="tests/data/videos/crosswalk.avi",
        user_comment="detect human",
        additional_params={"detect_classes": ["person"]},
    )
    result = agent.process_video(vision_input)
    assert result.video_path == "tests/data/videos/crosswalk.avi"
    assert result.fps > 0
    assert result.total_time > 0


def test_process_video_invalid_file(agent):
    vision_input = VisionInput(
        image="invalid_video.avi",
        user_comment="detect human",
        additional_params={"detect_classes": ["person"]},
    )
    with pytest.raises(ValueError):
        agent.process_video(vision_input)


@patch("cv2.VideoCapture")
def test_process_video_no_valid_frames(mock_video_capture, agent):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)
    mock_video_capture.return_value = mock_cap

    vision_input = VisionInput(
        image="tests/data/videos/empty_video.avi",
        user_comment="detect human",
        additional_params={"detect_classes": ["person"]},
    )
    with pytest.raises(RuntimeError):
        agent.process_video(vision_input)


def test_process_video_invalid_frame_range(agent):
    vision_input = VisionInput(
        image="tests/data/videos/crosswalk.avi",
        user_comment="detect human",
        additional_params={"detect_classes": ["person"]},
    )
    with pytest.raises(ValueError):
        agent.process_video(vision_input, start_time=10.0, end_time=5.0)


def test_process_video_frame_range(agent):
    mock_process_detections = MagicMock()
    mock_process_detections.return_value = [
        {"class": "person", "bbox": [0, 0, 10, 10], "confidence": 0.9}
    ]
    agent.model.predict = MagicMock(return_value=[MagicMock()])

    vision_input = VisionInput(
        image="tests/data/videos/crosswalk.avi",
        user_comment="detect human",
        additional_params={"detect_classes": ["person"]},
    )
    result = agent.process_video(vision_input, start_time=1.0, end_time=2.0)
    assert result.video_path == "tests/data/videos/crosswalk.avi"
    assert result.fps > 0
    assert result.total_time > 0


@patch("cv2.VideoCapture")
@patch("time.time", side_effect=(i for i in range(100)))
def test_process_video_with_tracking(mock_time, mock_video_capture, agent):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [
        (True, np.zeros((640, 640, 3), dtype=np.uint8))
    ] * 10 + [(False, None)]
    mock_cap.get.side_effect = lambda x: {
        cv2.CAP_PROP_FPS: 30,
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FRAME_COUNT: 10,
    }[x]
    mock_video_capture.return_value = mock_cap

    mock_detections = MagicMock()
    mock_detections.boxes = [
        MagicMock(
            cls=0,
            xyxy=[torch.tensor([50, 50, 100, 100])],
            conf=torch.tensor(0.9),
            id=torch.tensor(1),
        )
    ]
    mock_detections.names = {0: "person"}
    agent.model.predict = MagicMock(return_value=[mock_detections])

    vision_input = VisionInput(
        image="tests/data/videos/crosswalk.avi",
        user_comment="detect human",
        additional_params={"detect_classes": ["person"]},
    )
    result = agent.process_video(vision_input)
    assert result.video_path == "tests/data/videos/crosswalk.avi"
    assert result.fps == 30
    assert result.total_time > 0
    assert "track_id" in result.frames_results[0].results["detections"][0]


def test_visualize_detections(agent):
    mock_image = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = [{"class": "person", "bbox": [50, 50, 100, 100], "confidence": 0.9}]
    annotated_image = agent.visualize_detections(mock_image, detections)
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[0] > 0 and annotated_image.shape[1] > 0


def test_visualize_detections_invalid_bbox(agent, caplog):
    mock_image = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = [
        {"class": "person", "bbox": [50, 50], "confidence": 0.9}  # Invalid bbox
    ]
    annotated_image = agent.visualize_detections(mock_image, detections)
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[0] > 0 and annotated_image.shape[1] > 0
    assert "Invalid bounding box" in caplog.text


def test_visualize_detections_with_tracking_id(agent):
    mock_image = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = [
        {
            "class": "person",
            "bbox": [50, 50, 100, 100],
            "confidence": 0.9,
            "track_id": 1,
        }
    ]
    annotated_image = agent.visualize_detections(mock_image, detections)
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[0] > 0 and annotated_image.shape[1] > 0


def test_visualize_detections_with_output_path(agent, tmp_path):
    mock_image = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = [{"class": "person", "bbox": [50, 50, 100, 100], "confidence": 0.9}]
    output_path = tmp_path / "annotated_image.jpg"
    annotated_image = agent.visualize_detections(
        mock_image, detections, output_path=str(output_path)
    )
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[0] > 0 and annotated_image.shape[1] > 0
    assert output_path.exists()
