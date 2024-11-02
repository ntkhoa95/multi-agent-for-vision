from ultralytics import YOLO
import cv2
import logging
from tqdm import tqdm
import os
import time
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
from ..core.types import VisionTaskType, VisionInput, VisionOutput, BatchDetectionResult, VideoDetectionResult
from .base import BaseVisionAgent
from ..utils.image import draw_detections
from ..utils.video import get_video_properties

logger = logging.getLogger(__name__)

class YOLODetectionAgent(BaseVisionAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        self.conf_threshold = config.get('YOLO_CONFIDENCE_THRESHOLD', 0.25)
        self.iou_threshold = config.get('YOLO_IOU_THRESHOLD', 0.45)
        self.image_size = config.get('DETECTION_IMAGE_SIZE', 640)
        self.enable_tracking = config.get('ENABLE_TRACK', True)
        
        # Initialize tracking if enabled
        if self.enable_tracking:
            try:
                self.model.tracker = "bytetrack.yaml"
                logger.info("Tracking enabled with ByteTrack")
            except Exception as e:
                logger.warning(f"Failed to enable tracking: {str(e)}")
                self.enable_tracking = False
    
    def load_model(self):
        """Load YOLOv8 model"""
        logger.info(f"Loading YOLOv8 model: {self.config['YOLO_MODEL_NAME']}")
        model = YOLO(self.config['YOLO_MODEL_NAME'])
        return model
    
    def process_detections(self, results, detect_classes: Optional[List[str]] = None) -> List[Dict]:
        """Process YOLO results into standardized detection format"""
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_name = result.names[int(box.cls)]
                
                # Filter by specified classes if any
                if detect_classes and class_name not in detect_classes:
                    continue
                
                detection = {
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf),
                    'class': class_name,
                    'class_id': int(box.cls)
                }
                
                # Add tracking ID if available
                if hasattr(box, 'id') and box.id is not None:
                    try:
                        detection['track_id'] = int(box.id.item())
                    except:
                        detection['track_id'] = str(box.id)
                
                detections.append(detection)
        
        return detections

    def filter_detections(self, detections: List[Dict], allowed_classes: Optional[List[str]] = None) -> List[Dict]:
        """Filter detections based on allowed classes"""
        if not allowed_classes:
            return detections
        
        # Convert to lower case for case-insensitive comparison
        allowed_classes = [cls.lower() for cls in allowed_classes]
        
        return [
            det for det in detections
            if det['class'].lower() in allowed_classes
        ]

    def process(self, vision_input: VisionInput) -> VisionOutput:
        """Process single image detection with strict filtering"""
        start_time = time.time()
        
        # Get specific classes to detect from additional parameters
        detect_classes = None
        if vision_input.additional_params and 'detect_classes' in vision_input.additional_params:
            detect_classes = vision_input.additional_params['detect_classes']
            logger.info(f"Filtering detections for classes: {detect_classes}")

        # Run inference
        results = self.model(
            vision_input.image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device
        )
        
        # Process and filter detections
        detections = self.process_detections(results)
        if detect_classes:
            detections = self.filter_detections(detections, detect_classes)
        
        # Calculate confidence
        overall_confidence = (
            sum(d['confidence'] for d in detections[:5]) / min(5, len(detections))
            if detections else 0.0
        )
        
        return VisionOutput(
            task_type=VisionTaskType.OBJECT_DETECTION,
            results={
                'detections': detections,
                'num_detections': len(detections),
                'detect_classes': detect_classes
            },
            confidence=overall_confidence,
            processing_time=time.time() - start_time
        )
    
    def process_batch(self, image_paths: List[str], 
                     batch_size: int = 32) -> List[BatchDetectionResult]:
        """Process a batch of images"""
        results = []
        
        # Process images in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load batch images
            batch_images = [self.load_image(path) for path in batch_paths]
            
            # Run inference on batch
            batch_results = self.model(
                batch_images,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device
            )
            
            # Process each result
            for img_path, result in zip(batch_paths, batch_results):
                detections = self.process_detections([result])
                
                # Calculate confidence
                overall_confidence = (
                    sum(d['confidence'] for d in detections[:5]) / min(5, len(detections))
                    if detections else 0.0
                )
                
                # Create vision output
                vision_output = VisionOutput(
                    task_type=VisionTaskType.OBJECT_DETECTION,
                    results={
                        'detections': detections,
                        'num_detections': len(detections)
                    },
                    confidence=overall_confidence,
                    processing_time=0.0
                )
                
                results.append(BatchDetectionResult(img_path, vision_output))
        
        return results
    
    def process_video(self, 
                     vision_input: VisionInput,
                     output_path: Optional[str] = None,
                     start_time: float = 0,
                     end_time: Optional[float] = None) -> VideoDetectionResult:
        """Process video file for object detection"""
        video_path = vision_input.image
        detect_classes = None
        
        if vision_input.additional_params and 'detect_classes' in vision_input.additional_params:
            detect_classes = vision_input.additional_params['detect_classes']
            logger.info(f"Filtering detections for classes: {detect_classes}")

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame range
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time else total_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Setup video writer if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frames_results = []
        total_time = 0
        
        try:
            with tqdm(total=end_frame-start_frame, desc="Processing video") as pbar:
                frame_count = start_frame
                
                while frame_count < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    start_time_proc = time.time()
                    results = self.model(
                        frame,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        device=self.device
                    )
                    processing_time = time.time() - start_time_proc
                    total_time += processing_time
                    
                    # Process detections
                    detections = self.process_detections(results, detect_classes)
                    if detect_classes:
                        detections = self.filter_detections(detections, detect_classes)
                                        
                    # Calculate confidence
                    overall_confidence = (
                        sum(d['confidence'] for d in detections[:5]) / min(5, len(detections))
                        if detections else 0.0
                    )
                    
                    # Create vision output
                    vision_output = VisionOutput(
                        task_type=VisionTaskType.OBJECT_DETECTION,
                        results={
                            'detections': detections,
                            'num_detections': len(detections),
                            'detect_classes': detect_classes
                        },
                        confidence=overall_confidence,
                        processing_time=processing_time
                    )
                    frames_results.append(vision_output)
                    
                    # Write annotated frame if output path is provided
                    if output_path:
                        if detections:
                            annotated_frame = draw_detections(frame, detections)
                            out.write(annotated_frame)
                        else:
                            out.write(frame)
                    
                    frame_count += 1
                    pbar.update(1)
        
        finally:
            cap.release()
            if output_path and 'out' in locals():
                out.release()
        
        if frame_count == start_frame:
            raise RuntimeError("No frames were processed successfully")
        
        return VideoDetectionResult(
            video_path=video_path,
            frames_results=frames_results,
            fps=fps,
            total_time=total_time
        )
    
    def visualize_detections(self, image_path: str, detections: List[Dict], 
                           output_path: Optional[str] = None) -> np.ndarray:
        """Visualize detections on the image"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        annotated_image = draw_detections(image, detections)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        return annotated_image