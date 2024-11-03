import numpy as np
import pytest
from PIL import Image

from vision_framework.agents.base import BaseVisionAgent
from vision_framework.core.types import VisionInput, VisionOutput


@pytest.fixture
def vision_agent():
    class MockVisionAgent(BaseVisionAgent):
        def load_model(self):
            return None

        def process(self, vision_input: VisionInput) -> VisionOutput:
            return VisionOutput()

    config = {"DEVICE": "cpu"}
    return MockVisionAgent(config)


def test_load_image_from_path(vision_agent, tmp_path):
    img = Image.new("RGB", (100, 100))
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)

    loaded_image = vision_agent.load_image(str(img_path))
    assert isinstance(loaded_image, Image.Image)


def test_load_image_from_array(vision_agent):
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    loaded_image = vision_agent.load_image(img_array)
    assert isinstance(loaded_image, Image.Image)


def test_load_image_from_pil_image(vision_agent):
    img = Image.new("RGB", (100, 100))
    loaded_image = vision_agent.load_image(img)
    assert isinstance(loaded_image, Image.Image)


def test_load_image_invalid_type(vision_agent):
    with pytest.raises(ValueError, match="Unsupported image type"):
        vision_agent.load_image(12345)  # Passing an unsupported type


def test_load_model_not_implemented():
    with pytest.raises(NotImplementedError):
        BaseVisionAgent({"DEVICE": "cpu"}).load_model()


def test_process_not_implemented():
    with pytest.raises(NotImplementedError):
        BaseVisionAgent({"DEVICE": "cpu"}).process(
            VisionInput(image="dummy_path", user_comment="test")
        )
