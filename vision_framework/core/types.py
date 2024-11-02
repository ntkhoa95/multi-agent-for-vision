from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union, Optional
import numpy as np
from PIL import Image

class VisionTaskType(Enum):
    OBJECT_DETECTION = "object_detection"
    IMAGE_CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    OCR = "ocr"
    FACE_DETECTION = "face_detection"

@dataclass
class VisionInput:
    image: Union[str, np.ndarray, Image.Image]
    user_comment: str
    task_type: Optional[VisionTaskType] = None
    additional_params: Optional[Dict] = None

@dataclass
class VisionOutput:
    task_type: VisionTaskType
    results: Dict
    confidence: float
    processing_time: float

@dataclass
class BatchDetectionResult:
    image_path: str
    vision_output: VisionOutput

@dataclass
class VideoDetectionResult:
    video_path: str
    frames_results: List[VisionOutput]
    fps: float
    total_time: float
    
    @property
    def average_confidence(self) -> float:
        return np.mean([r.confidence for r in self.frames_results])
    
    @property
    def num_frames(self) -> int:
        return len(self.frames_results)

