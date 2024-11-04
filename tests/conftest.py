import os
import subprocess
from pathlib import Path

import pytest


def download_file(url: str, output_path: str):
    """Download a file using wget."""
    subprocess.run(["wget", "-O", output_path, url], check=True)


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
def sample_image_path(test_images_dir):
    """Download and return path to sample image."""
    image_path = test_images_dir / "street.jpg"
    if not image_path.exists():
        download_file(
            "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
            str(image_path),
        )
    return image_path


@pytest.fixture
def sample_person_image_path(test_images_dir):
    """Download and return path to person image."""
    image_path = test_images_dir / "person.jpg"
    if not image_path.exists():
        download_file(
            "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
            str(image_path),
        )
    return image_path


@pytest.fixture
def sample_dog_image_path(test_images_dir):
    """Download and return path to dog image."""
    image_path = test_images_dir / "dog.jpg"
    if not image_path.exists():
        download_file(
            "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/dog.jpg",
            str(image_path),
        )
    return image_path


@pytest.fixture
def sample_video_path(test_videos_dir):
    """Download and return path to sample video."""
    video_path = test_videos_dir / "crosswalk.avi"
    if not video_path.exists():
        download_file(
            "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/videos/human-cropped.mp4",
            str(video_path),
        )
    return video_path


@pytest.fixture(autouse=True)
def setup_test_dirs(test_images_dir, test_videos_dir):
    """Ensure test directories exist."""
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_videos_dir.mkdir(parents=True, exist_ok=True)
