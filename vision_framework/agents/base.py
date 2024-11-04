import logging
from typing import Dict, Optional

import torch
from PIL import Image

from ..core.types import VisionInput, VisionOutput

logger = logging.getLogger(__name__)


class BaseVisionAgent:
    def __init__(self, config: Dict):
        """Initialize base vision agent."""
        self.config = config
        self.device = torch.device(
            config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Using device: {self.device}")
        self.model = None
        self.load_model()

    def load_model(self):
        """Load model - to be implemented by child classes."""
        raise NotImplementedError("Child classes must implement load_model")

    def load_image(self, image_path: str) -> Image.Image:
        """Load image from path."""
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

    def process(self, vision_input: VisionInput) -> VisionOutput:
        """Process input - to be implemented by child classes."""
        raise NotImplementedError("Child classes must implement process")
