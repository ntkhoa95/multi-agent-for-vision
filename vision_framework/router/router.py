import logging
from typing import Dict, Optional, Tuple

from ..agents.base import BaseVisionAgent
from ..core.types import VisionInput, VisionOutput, VisionTaskType
from ..nlp.processor import NLPProcessor

logger = logging.getLogger(__name__)


class AgentRouter:
    def __init__(self):
        self.agents: Dict[VisionTaskType, BaseVisionAgent] = {}
        self.nlp_processor = NLPProcessor()

        self.task_mapping = {
            "detection": VisionTaskType.OBJECT_DETECTION,
            "classification": VisionTaskType.IMAGE_CLASSIFICATION,
            "segmentation": VisionTaskType.SEGMENTATION,
            "ocr": VisionTaskType.OCR,
            "face": VisionTaskType.FACE_DETECTION,
        }

    def register_agent(self, task_type: VisionTaskType, agent: BaseVisionAgent):
        """Register a new agent for a specific task type"""
        self.agents[task_type] = agent
        logger.info(f"Registered agent for task type: {task_type}")

    def determine_task_type(self, user_comment: str) -> Tuple[VisionTaskType, Dict]:
        """Determine the task type and parameters based on user comment"""
        task_type_str, target_objects = self.nlp_processor.parse_query(user_comment)
        additional_params = {}

        if task_type_str and task_type_str in self.task_mapping:
            task_type = self.task_mapping[task_type_str]
        else:
            task_type = VisionTaskType.OBJECT_DETECTION

        if task_type == VisionTaskType.OBJECT_DETECTION and target_objects:
            if VisionTaskType.OBJECT_DETECTION in self.agents:
                agent = self.agents[VisionTaskType.OBJECT_DETECTION]
                if hasattr(agent.model, "names"):
                    available_classes = set(agent.model.names.values())
                    target_objects = self.nlp_processor.validate_objects(
                        target_objects, available_classes
                    )

            if target_objects:
                logger.info(f"Detected target objects: {target_objects}")
                additional_params["detect_classes"] = target_objects

        return task_type, additional_params

    def process_request(self, vision_input: VisionInput) -> VisionOutput:
        """Route the request to appropriate agent and get results"""
        if vision_input.task_type is None:
            task_type, additional_params = self.determine_task_type(vision_input.user_comment)
            vision_input.task_type = task_type
            vision_input.additional_params = additional_params

            logger.info(f"Determined task type: {task_type}")
            if additional_params:
                logger.info(f"Additional parameters: {additional_params}")

        if vision_input.task_type not in self.agents:
            raise ValueError(f"No agent registered for task type: {vision_input.task_type}")

        return self.agents[vision_input.task_type].process(vision_input)
