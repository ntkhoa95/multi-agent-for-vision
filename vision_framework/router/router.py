# vision_framework/router/router.py
import logging
from typing import Dict, Optional, Tuple

from ..agents.base import BaseVisionAgent
from ..core.types import VisionInput, VisionOutput, VisionTaskType
from ..nlp.processor import NLPProcessor

logger = logging.getLogger(__name__)


class AgentRouter:
    def __init__(self):
        """Initialize router with task mappings and NLP processor."""
        self.agents: Dict[VisionTaskType, BaseVisionAgent] = {}
        self.nlp_processor = NLPProcessor()

        # Define task mappings
        self.task_mapping = {
            "detection": VisionTaskType.OBJECT_DETECTION,
            "classification": VisionTaskType.IMAGE_CLASSIFICATION,
            "segmentation": VisionTaskType.SEGMENTATION,
            "ocr": VisionTaskType.OCR,
            "face": VisionTaskType.FACE_DETECTION,
            "captioning": VisionTaskType.IMAGE_CAPTIONING,  # Add captioning mapping
        }

        # Add explicit keyword mappings for better task detection
        self.keyword_mapping = {
            "describe": VisionTaskType.IMAGE_CAPTIONING,
            "caption": VisionTaskType.IMAGE_CAPTIONING,
            "what": VisionTaskType.IMAGE_CAPTIONING,
            "detect": VisionTaskType.OBJECT_DETECTION,
            "find": VisionTaskType.OBJECT_DETECTION,
            "locate": VisionTaskType.OBJECT_DETECTION,
            "classify": VisionTaskType.IMAGE_CLASSIFICATION,
            "categorize": VisionTaskType.IMAGE_CLASSIFICATION,
        }

        logger.info("AgentRouter initialized with task mappings")
        logger.debug(f"Available task types: {list(self.task_mapping.values())}")

    def register_agent(self, task_type: VisionTaskType, agent: BaseVisionAgent):
        """Register a new agent for a specific task type."""
        self.agents[task_type] = agent
        logger.info(f"Registered agent for task type: {task_type} ({task_type.value})")
        logger.debug(f"Current registered agents: {list(self.agents.keys())}")

    def determine_task_type(self, user_comment: str) -> Tuple[VisionTaskType, Dict]:
        """Determine the task type and parameters based on user comment."""
        logger.debug(f"Determining task type for comment: {user_comment}")

        # First try NLP processor
        task_type_str, target_objects = self.nlp_processor.parse_query(user_comment)
        additional_params = {}

        # Check keyword mapping first
        normalized_comment = user_comment.lower()
        for keyword, task_type in self.keyword_mapping.items():
            if keyword in normalized_comment:
                logger.debug(f"Matched keyword '{keyword}' to task type: {task_type}")
                return task_type, additional_params

        # Then check task mapping
        if task_type_str and task_type_str in self.task_mapping:
            task_type = self.task_mapping[task_type_str]
            logger.debug(f"Mapped task type string '{task_type_str}' to {task_type}")
        else:
            logger.debug(f"No specific task type found, using default: OBJECT_DETECTION")
            task_type = VisionTaskType.OBJECT_DETECTION

        # Handle object detection parameters
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

        logger.info(f"Determined task type: {task_type}")
        if additional_params:
            logger.debug(f"Additional parameters: {additional_params}")

        return task_type, additional_params

    def process_request(self, vision_input: VisionInput) -> VisionOutput:
        """Route the request to appropriate agent and get results."""
        logger.debug(f"Processing request with input: {vision_input}")

        if vision_input.task_type is None:
            task_type, additional_params = self.determine_task_type(vision_input.user_comment)
            vision_input.task_type = task_type
            vision_input.additional_params = additional_params
            logger.info(f"Determined task type: {task_type}")
            if additional_params:
                logger.debug(f"Additional parameters: {additional_params}")

        # Verify agent availability
        if vision_input.task_type not in self.agents:
            available_agents = list(self.agents.keys())
            logger.error(f"No agent found for task type: {vision_input.task_type}")
            logger.error(f"Available agents: {available_agents}")
            raise ValueError(f"No agent registered for task type: {vision_input.task_type}")

        # Process request with appropriate agent
        agent = self.agents[vision_input.task_type]
        logger.debug(f"Processing with agent: {type(agent).__name__}")

        try:
            result = agent.process(vision_input)
            logger.debug(f"Successfully processed request with {type(agent).__name__}")
            return result
        except Exception as e:
            logger.error(f"Error processing request with {type(agent).__name__}: {str(e)}")
            raise

    def get_registered_agents(self) -> Dict[str, str]:
        """Get information about registered agents."""
        return {task_type.value: type(agent).__name__ for task_type, agent in self.agents.items()}
