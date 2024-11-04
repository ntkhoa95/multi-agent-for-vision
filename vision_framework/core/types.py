from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image


class VisionTaskType(Enum):
    OBJECT_DETECTION = "object_detection"
    IMAGE_CLASSIFICATION = "classification"
    IMAGE_CAPTIONING = "image_captioning"
    SEGMENTATION = "segmentation"
    OCR = "ocr"
    FACE_DETECTION = "face_detection"

    def __str__(self):
        return self.value


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
    confidence: float = 1.0
    processing_time: float = 0.0

    def get_caption(self) -> Optional[str]:
        """Get caption if this is a captioning result."""
        if self.task_type == VisionTaskType.IMAGE_CAPTIONING:
            return self.results.get("caption")
        return None

    def get_detections(self) -> List[Dict]:
        """Get detections if this is a detection result."""
        if self.task_type == VisionTaskType.OBJECT_DETECTION:
            return self.results.get("detections", [])
        return []


@dataclass
class BatchDetectionResult:
    image_path: str
    vision_output: VisionOutput

    def get_detections(self) -> List[Dict]:
        """Convenience method to get detections."""
        return self.vision_output.get_detections()


@dataclass
class VideoDetectionResult:
    video_path: str
    frames_results: List[VisionOutput]
    fps: float
    total_time: float

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all frames."""
        if not self.frames_results:
            return 0.0
        return np.mean([r.confidence for r in self.frames_results])

    @property
    def num_frames(self) -> int:
        """Get total number of processed frames."""
        return len(self.frames_results)

    def get_frame_detections(self, frame_idx: int) -> List[Dict]:
        """Get detections for a specific frame."""
        if 0 <= frame_idx < len(self.frames_results):
            return self.frames_results[frame_idx].get_detections()
        return []
