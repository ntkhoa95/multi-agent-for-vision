# tests/conftest.py
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def test_images_dir(test_data_dir):
    return test_data_dir / "images"


@pytest.fixture(scope="session")
def test_videos_dir(test_data_dir):
    return test_data_dir / "videos"


@pytest.fixture(scope="session")
def sample_image_path(test_images_dir):
    return test_images_dir / "street.jpg"


@pytest.fixture(scope="session")
def sample_video_path(test_videos_dir):
    return test_videos_dir / "crosswalk.avi"


@pytest.fixture(autouse=True)
def setup_test_dirs(test_images_dir, test_videos_dir):
    """Ensure test directories exist."""
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_videos_dir.mkdir(parents=True, exist_ok=True)
