import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image

from .agents.captioning import CaptioningAgent
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
        self.enable_captioning = config.get("ENABLE_CAPTIONING", False)
        self.initialize_agents()

    def initialize_agents(self):
        """Initialize and register all vision agents."""
        # Classification agent
        try:
            classification_agent = MobileNetClassificationAgent(self.config)
            self.router.register_agent(VisionTaskType.IMAGE_CLASSIFICATION, classification_agent)
            logger.info("Registered agent for task type: IMAGE_CLASSIFICATION")
        except Exception as e:
            logger.warning(f"Failed to initialize classification agent: {str(e)}")

        # Detection agent
        try:
            detection_agent = YOLODetectionAgent(self.config)
            self.router.register_agent(VisionTaskType.OBJECT_DETECTION, detection_agent)
            logger.info("Registered agent for task type: OBJECT_DETECTION")
        except Exception as e:
            logger.warning(f"Failed to initialize detection agent: {str(e)}")

        # Captioning agent (only if enabled)
        if self.enable_captioning:
            try:
                captioning_agent = CaptioningAgent(self.config)
                self.router.register_agent(VisionTaskType.IMAGE_CAPTIONING, captioning_agent)
                logger.info("Registered agent for task type: IMAGE_CAPTIONING")
            except Exception as e:
                logger.warning(f"Failed to initialize captioning agent: {str(e)}")

    def list_agents(self) -> Set[VisionTaskType]:
        """List all registered agents."""
        return set(self.router.agents.keys())

    def filter_detections(self, detections: List[Dict], allowed_classes: List[str]) -> List[Dict]:
        """Filter detections based on allowed class names."""
        return [
            det for det in detections if det["class"].lower() in map(str.lower, allowed_classes)
        ]

    # def visualize_detections(
    #     self, image_path: str, detections: List[Dict], output_path: Optional[str] = None
    # ) -> bool:
    #     """Visualize detections on the image and optionally save to output_path."""
    #     try:
    #         image = cv2.imread(image_path)
    #         if image is None:
    #             logger.error(f"Failed to read image: {image_path}")
    #             return False

    #         for detection in detections:
    #             if "bbox" not in detection:
    #                 continue

    #             x1, y1, x2, y2 = detection["bbox"]
    #             label = detection["class"]
    #             confidence = detection.get("confidence", 0)

    #             cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #             cv2.putText(
    #                 image,
    #                 f"{label}: {confidence:.2f}",
    #                 (int(x1), int(y1) - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5,
    #                 (255, 255, 255),
    #                 2,
    #             )

    #         if output_path:
    #             cv2.imwrite(output_path, image)

    #         return True
    #     except Exception as e:
    #         logger.error(f"Error visualizing detections: {str(e)}")
    #         return False

    def visualize_detections(
        self,
        image_path: str,
        detections: List[Dict],
        output_path: Optional[str] = None,
        caption: Optional[str] = None,
    ) -> bool:
        """Visualize detections on the image and optionally save to output_path."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return False

            img_height, img_width = image.shape[:2]

            # Add caption first if provided
            if caption:
                margin = 10
                font_scale = 0.7
                thickness = 2

                # Split caption into multiple lines if too long
                max_width = img_width - 2 * margin
                words = caption.split()
                lines = []
                current_line = words[0]

                for word in words[1:]:
                    test_line = current_line + " " + word
                    (test_width, text_height), _ = cv2.getTextSize(
                        test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )
                    if test_width <= max_width:
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = word
                lines.append(current_line)

                # Draw caption background
                line_height = int(text_height * 1.5)
                total_height = line_height * len(lines) + 2 * margin
                cv2.rectangle(image, (0, 0), (img_width, total_height), (0, 0, 0), -1)

                # Draw caption text
                for i, line in enumerate(lines):
                    y_position = margin + (i + 1) * line_height
                    cv2.putText(
                        image,
                        line,
                        (margin, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                    )

            # Draw detections
            for detection in detections:
                if "bbox" not in detection:
                    continue

                x1, y1, x2, y2 = detection["bbox"]
                label = detection["class"]
                confidence = detection.get("confidence", 0)
                track_id = detection.get("track_id", None)

                # Create label text
                label_text = f"{label}: {confidence:.2f}"
                if track_id is not None:
                    label_text += f" ID:{track_id}"

                # Draw bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    image,
                    (int(x1), int(y1) - text_height - 10),
                    (int(x1) + text_width, int(y1)),
                    (0, 255, 0),
                    -1,
                )

                # Draw label text
                cv2.putText(
                    image,
                    label_text,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            if output_path:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(output_path, image)

            return True

        except Exception as e:
            logger.error(f"Error visualizing detections: {str(e)}")
            return False

    # def process_image_with_caption(
    #     self,
    #     image_path: str,
    #     user_comment: Optional[str] = None,
    #     task_type: Optional[VisionTaskType] = None,
    # ) -> Tuple[VisionOutput, Optional[VisionOutput]]:
    #     """Process image with detection and generate caption for detected objects."""
    #     detection_result = self.process_image(
    #         image_path=image_path,
    #         user_comment=user_comment or "detect objects",
    #         task_type=task_type or VisionTaskType.OBJECT_DETECTION,
    #     )

    #     caption_result = None
    #     if VisionTaskType.IMAGE_CAPTIONING in self.router.agents:
    #         try:
    #             detections = detection_result.results.get("detections", [])
    #             objects = [f"{d['class']}" for d in detections]
    #             prompt = f"An image containing {', '.join(objects)}"

    #             caption_result = self.process_image(
    #                 image_path=image_path,
    #                 user_comment=prompt,
    #                 task_type=VisionTaskType.IMAGE_CAPTIONING,
    #             )
    #         except Exception as e:
    #             logger.warning(f"Failed to generate caption: {str(e)}")

    #     return detection_result, caption_result

    def process_image_with_caption(
        self,
        image_path: str,
        user_comment: Optional[str] = None,
        task_type: Optional[VisionTaskType] = None,
    ) -> Tuple[VisionOutput, Optional[VisionOutput]]:
        """Process image with detection and generate caption for detected objects."""
        detection_result = self.process_image(
            image_path=image_path,
            user_comment=user_comment or "detect objects",
            task_type=task_type or VisionTaskType.OBJECT_DETECTION,
        )

        caption_result = None
        if VisionTaskType.IMAGE_CAPTIONING in self.router.agents:
            try:
                detections = detection_result.results.get("detections", [])
                objects = [f"{d['class']}" for d in detections]
                prompt = f"An image containing {', '.join(objects)}"

                caption_result = self.process_image(
                    image_path=image_path,
                    user_comment=prompt,
                    task_type=VisionTaskType.IMAGE_CAPTIONING,
                )

                # You could also visualize here if needed
                output_path = f"results/{Path(image_path).stem}_with_caption.jpg"
                self.visualize_detections(
                    image_path=image_path,
                    detections=detection_result.results["detections"],
                    output_path=output_path,
                    caption=caption_result.results.get("caption"),
                )

            except Exception as e:
                logger.warning(f"Failed to generate caption: {str(e)}")

        return detection_result, caption_result

    def process_image(
        self,
        image_path: str,
        user_comment: str,
        task_type: Optional[VisionTaskType] = None,
        additional_params: Optional[Dict] = None,
    ) -> VisionOutput:
        """Process single image."""
        start_time = time.time()

        if task_type is None:
            task_type, additional_params = self.router.determine_task_type(user_comment)

        agent = self.router.agents.get(task_type)
        if agent is None:
            raise ValueError(f"No agent registered for task type: {task_type}")

        vision_input = VisionInput(
            image=image_path,
            user_comment=user_comment,
            task_type=task_type,
            additional_params=additional_params,
        )

        result = agent.process(vision_input)
        result.processing_time = time.time() - start_time

        return result

    def process_batch(
        self, image_paths: List[str], task_type: VisionTaskType, user_comment: str = ""
    ) -> List[BatchDetectionResult]:
        """Process batch of images."""
        agent = self.router.agents.get(task_type)
        if agent is None:
            raise ValueError(f"No agent registered for task type: {task_type}")

        if hasattr(agent, "process_batch"):
            return agent.process_batch(
                image_paths=image_paths, batch_size=self.config.get("BATCH_SIZE", 32)
            )

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
    ) -> Tuple[VideoDetectionResult, Optional[List[VisionOutput]]]:
        """Process video file with optional captioning."""
        task_type, additional_params = self.router.determine_task_type(user_comment)

        vision_input = VisionInput(
            image=video_path,
            user_comment=user_comment,
            task_type=task_type,
            additional_params=additional_params,
        )

        agent = self.router.agents.get(task_type)
        if agent is None:
            raise ValueError(f"No agent registered for task type: {task_type}")

        if not hasattr(agent, "process_video"):
            raise ValueError(f"Agent {type(agent).__name__} does not support video processing")

        video_result = agent.process_video(
            vision_input=vision_input,
            output_path=output_path,
            start_time=start_time,
            end_time=end_time,
        )

        frame_captions = None
        if self.enable_captioning and VisionTaskType.IMAGE_CAPTIONING in self.router.agents:
            captioning_agent = self.router.agents[VisionTaskType.IMAGE_CAPTIONING]
            frame_captions = []

            for frame_result in video_result.frames_results:
                if "frame" in frame_result.results:
                    caption_input = VisionInput(
                        image=frame_result.results["frame"],
                        user_comment="",
                        task_type=VisionTaskType.IMAGE_CAPTIONING,
                    )
                    caption_output = captioning_agent.process(caption_input)
                    frame_captions.append(caption_output)

        return video_result, frame_captions
