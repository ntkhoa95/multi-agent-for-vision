from pathlib import Path

import pytest
import torch

from vision_framework import VisionOrchestrator


@pytest.fixture(scope="session")
def test_data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def test_images_dir(test_data_dir):
    return test_data_dir / "images"


@pytest.fixture(scope="session")
def test_videos_dir(test_data_dir):
    return test_data_dir / "videos"


@pytest.fixture
def config():
    return {
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "BATCH_SIZE": 32,
        "NUM_WORKERS": 4,
        "YOLO_MODEL_NAME": "yolov8s.pt",
        "MODEL_NAME": "mobilenetv3_large_100",
        "MODEL_PRETRAINED": True,
    }


@pytest.fixture
def orchestrator(config):
    return VisionOrchestrator(config)
