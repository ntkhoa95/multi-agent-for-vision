import timm
import torch
import json
import time
from pathlib import Path
import logging
from ..core.types import VisionTaskType, VisionInput, VisionOutput
from .base import BaseVisionAgent

logger = logging.getLogger(__name__)

class MobileNetClassificationAgent(BaseVisionAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        self.setup_transforms()
        self.load_imagenet_labels()
    
    def load_model(self):
        """Load MobileNetV3 from timm"""
        logger.info(f"Loading {self.config['MODEL_NAME']} model...")
        model = timm.create_model(
            self.config['MODEL_NAME'],
            pretrained=self.config['MODEL_PRETRAINED']
        )
        model = model.to(self.device)
        model.eval()
        return model
    
    def load_imagenet_labels(self):
        """Load ImageNet class labels"""
        labels_path = Path('imagenet_labels.json')
        if not labels_path.exists():
            self.labels = [f"class_{i}" for i in range(1000)]
        else:
            with open(labels_path, 'r') as f:
                label_dict = json.load(f)
                self.labels = [v[1] for v in label_dict.values()]
    
    def setup_transforms(self):
        """Setup transforms using timm's data config"""
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**self.data_config, is_training=False)
    
    def process(self, vision_input: VisionInput) -> VisionOutput:
        start_time = time.time()
        
        image = self.load_image(vision_input.image)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.config.get('USE_FP16', False)):
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        results = {
            "top_predictions": [
                {
                    "class": self.labels[idx.item()],
                    "confidence": prob.item()
                } for prob, idx in zip(top5_prob, top5_idx)
            ],
            "model_config": {
                "name": self.config['MODEL_NAME'],
                "input_size": self.data_config['input_size'],
                "interpolation": self.data_config['interpolation'],
                "mean": self.data_config['mean'],
                "std": self.data_config['std']
            }
        }
        
        return VisionOutput(
            task_type=VisionTaskType.IMAGE_CLASSIFICATION,
            results=results,
            confidence=top5_prob[0].item(),
            processing_time=time.time() - start_time
        )