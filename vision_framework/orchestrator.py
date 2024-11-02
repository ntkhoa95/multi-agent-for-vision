from typing import Dict, List, Optional
import logging
from .core.types import (
    VisionTaskType, VisionInput, VisionOutput, 
    BatchDetectionResult, VideoDetectionResult
)
from .core.config import validate_config
from .router.router import AgentRouter
from .agents.classification import MobileNetClassificationAgent
from .agents.detection import YOLODetectionAgent

logger = logging.getLogger(__name__)

class VisionOrchestrator:
    def __init__(self, config: Dict):
        self.config = validate_config(config)
        self.router = AgentRouter()
        self.initialize_agents()
    
    def initialize_agents(self):
        """Initialize and register all vision agents"""
        classification_agent = MobileNetClassificationAgent(self.config)
        self.router.register_agent(VisionTaskType.IMAGE_CLASSIFICATION, classification_agent)
        
        detection_agent = YOLODetectionAgent(self.config)
        self.router.register_agent(VisionTaskType.OBJECT_DETECTION, detection_agent)
    
    def process_image(self, image_path: str, user_comment: str,
                     task_type: Optional[VisionTaskType] = None,
                     additional_params: Optional[Dict] = None) -> VisionOutput:
        """Process single image"""
        vision_input = VisionInput(
            image=image_path,
            user_comment=user_comment,
            task_type=task_type,
            additional_params=additional_params
        )
        return self.router.process_request(vision_input)
    
    def process_batch(self, image_paths: List[str],
                     task_type: VisionTaskType,
                     user_comment: str = "") -> List[BatchDetectionResult]:
        """Process batch of images"""
        agent = self.router.agents.get(task_type)
        if agent is None:
            raise ValueError(f"No agent registered for task type: {task_type}")
        
        if hasattr(agent, 'process_batch'):
            return agent.process_batch(
                image_paths=image_paths,
                batch_size=self.config.get('BATCH_SIZE', 32)
            )
        else:
            from tqdm import tqdm
            results = []
            for image_path in tqdm(image_paths, desc="Processing images"):
                vision_input = VisionInput(
                    image=image_path,
                    user_comment=user_comment,
                    task_type=task_type
                )
                vision_output = agent.process(vision_input)
                results.append(BatchDetectionResult(image_path, vision_output))
            return results
    
    def process_video(self, video_path: str,
                     output_path: Optional[str] = None,
                     start_time: float = 0,
                     end_time: Optional[float] = None,
                     additional_params: Optional[Dict] = None,
                     user_comment: str = "") -> VideoDetectionResult:
        """Process video file"""
        task_type, _ = self.router.determine_task_type(user_comment)
        
        vision_input = VisionInput(
            image=video_path,
            user_comment=user_comment,
            task_type=task_type,
            additional_params=additional_params
        )
        
        agent = self.router.agents.get(task_type)
        if agent is None:
            raise ValueError(f"No agent registered for task type: {task_type}")
        
        if not hasattr(agent, 'process_video'):
            raise ValueError(f"Agent {type(agent).__name__} does not support video processing")
        
        return agent.process_video(
            vision_input=vision_input,
            output_path=output_path,
            start_time=start_time,
            end_time=end_time,
            additional_params=vision_input.additional_params,
        )