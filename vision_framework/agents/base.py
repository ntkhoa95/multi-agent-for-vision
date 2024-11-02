import torch
from PIL import Image
import numpy as np
from typing import Union
from ..core.types import VisionInput, VisionOutput

class BaseVisionAgent:
    def __init__(self, config: dict):
        self.config = config
        self.device = config['DEVICE']
        self.model = self.load_model()
    
    def load_model(self):
        raise NotImplementedError("Each agent must implement its model loading logic")
    
    def process(self, vision_input: VisionInput) -> VisionOutput:
        raise NotImplementedError("Each agent must implement its processing logic")
    
    def load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """Utility method to load images from various input types"""
        if isinstance(image_input, str):
            return Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            return Image.fromarray(np.uint8(image_input))
        elif isinstance(image_input, Image.Image):
            return image_input
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")



