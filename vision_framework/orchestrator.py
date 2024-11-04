import logging
from typing import Dict, List, Optional, Set

import cv2

from .agents.classification import MobileNetClassificationAgent
from .agents.detection import YOLODetectionAgent
from .core.config import validate_config
from .core.types import (
    BatchDetectionResult,
    VideoDetectionResult,
    VisionInput,
    VisionOutput,
    VisionTaskType,
)
from .router.router import AgentRouter

logger = logging.getLogger(__name__)


class VisionOrchestrator:
    def __init__(self, config: Dict):
        self.config = validate_config(config)
        self.router = AgentRouter()
        self.initialize_agents()

    def initialize_agents(self):
        classification_agent = MobileNetClassificationAgent(self.config)
        self.router.register_agent(VisionTaskType.IMAGE_CLASSIFICATION, classification_agent)

        detection_agent = YOLODetectionAgent(self.config)
        self.router.register_agent(VisionTaskType.OBJECT_DETECTION, detection_agent)

    def list_agents(self) -> Set[VisionTaskType]:
        """List all registered agents."""
        return set(self.router.agents.keys())

    def filter_detections(self, detections, allowed_classes):
        """Filter detections based on allowed class names."""
        return [
            det for det in detections if det["class"].lower() in map(str.lower, allowed_classes)
        ]

    def visualize_detections(
        self, image_path: str, detections: List[Dict], output_path: Optional[str] = None
    ) -> bool:
        """
        Visualize detections on the image and optionally save to output_path.
        Returns True if visualization is successful.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return False

            for detection in detections:
                if "bbox" not in detection:
                    continue

                x1, y1, x2, y2 = detection["bbox"]
                label = detection["class"]
                confidence = detection.get("confidence", 0)

                # Draw bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Add label with confidence
                cv2.putText(
                    image,
                    f"{label}: {confidence:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            if output_path:
                cv2.imwrite(output_path, image)

            return True
        except Exception as e:
            logger.error(f"Error visualizing detections: {str(e)}")
            return False

    def process_image(
        self,
        image_path: str,
        user_comment: str,
        task_type: Optional[VisionTaskType] = None,
        additional_params: Optional[Dict] = None,
    ) -> VisionOutput:
        """Process single image"""
        vision_input = VisionInput(
            image=image_path,
            user_comment=user_comment,
            task_type=task_type,
            additional_params=additional_params,
        )
        return self.router.process_request(vision_input)

    def process_batch(
        self, image_paths: List[str], task_type: VisionTaskType, user_comment: str = ""
    ) -> List[BatchDetectionResult]:
        """Process batch of images"""
        agent = self.router.agents.get(task_type)
        if agent is None:
            raise ValueError(f"No agent registered for task type: {task_type}")

        if hasattr(agent, "process_batch"):
            return agent.process_batch(
                image_paths=image_paths, batch_size=self.config.get("BATCH_SIZE", 32)
            )
        else:
            from tqdm import tqdm

            results = []
            for image_path in tqdm(image_paths, desc="Processing images"):
                vision_input = VisionInput(
                    image=image_path, user_comment=user_comment, task_type=task_type
                )
                vision_output = agent.process(vision_input)
                results.append(BatchDetectionResult(image_path, vision_output))
            return results

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        start_time: float = 0,
        end_time: Optional[float] = None,
        user_comment: str = "",
    ) -> VideoDetectionResult:
        """Process video file"""
        # Determine task type and parameters
        task_type, additional_params = self.router.determine_task_type(user_comment)
        logger.info(f"additional_params: '{additional_params}'")
        # Create vision input
        vision_input = VisionInput(
            image=video_path,
            user_comment=user_comment,
            task_type=task_type,
            additional_params=additional_params,
        )

        # Get appropriate agent
        agent = self.router.agents.get(task_type)
        if agent is None:
            raise ValueError(f"No agent registered for task type: {task_type}")

        if not hasattr(agent, "process_video"):
            raise ValueError(f"Agent {type(agent).__name__} does not support video processing")

        # Process video
        return agent.process_video(
            vision_input=vision_input,
            output_path=output_path,
            start_time=start_time,
            end_time=end_time,
        )
