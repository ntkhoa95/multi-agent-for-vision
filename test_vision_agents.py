import os
import torch
import logging
from pathlib import Path
import cv2
import urllib.request
from PIL import Image
import time
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
from vision_framework import VisionOrchestrator, VisionTaskType
from vision_framework.utils.logging import setup_logging

# Setup logging
logger = setup_logging()

class VisionFrameworkTester:
    def __init__(self):
        self.config = self.get_config()
        self.setup_directories()
        self.orchestrator = VisionOrchestrator(self.config)
        
    def get_config(self):
        """Get framework configuration"""
        return {
            'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
            'BATCH_SIZE': 32,
            'NUM_WORKERS': 4,
            'TEST_IMAGE_PATH': './test_assets/images/',
            'RESULTS_PATH': './test_results/',
            'VIDEO_SAVE_PATH': './test_results/videos/',
            'MODEL_NAME': 'mobilenetv3_large_100',
            'MODEL_PRETRAINED': True,
            'USE_FP16': False,
            'YOLO_MODEL_NAME': 'yolov8s.pt',
            'YOLO_CONFIDENCE_THRESHOLD': 0.25,
            'YOLO_IOU_THRESHOLD': 0.45,
            'DETECTION_IMAGE_SIZE': 640,
            'VIDEO_FPS': 30,
            'ENABLE_TRACK': True,
            'SAVE_CROPS': True,
        }
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config['TEST_IMAGE_PATH'],
            self.config['RESULTS_PATH'],
            self.config['VIDEO_SAVE_PATH'],
            './test_assets/videos/'
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def download_test_assets(self):
        """Download test images and prepare video paths"""
        # Test images
        image_urls = {
            'street.jpg': 'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg',
            'dog.jpg': 'https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg',
            'bus.jpg': 'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg'
        }
        
        for name, url in image_urls.items():
            path = Path(self.config['TEST_IMAGE_PATH']) / name
            if not path.exists():
                logger.info(f"Downloading {name}...")
                urllib.request.urlretrieve(url, path)
        
        # Use existing videos from test_videos folder
        video_dir = Path('./test_videos')
        video_paths = list(video_dir.glob('*.avi')) + list(video_dir.glob('*.mp4'))
        if not video_paths:
            logger.warning("No videos found in test_videos folder, creating synthetic video...")
            video_path = Path('./test_assets/videos/synthetic_test.mp4')
            self.create_test_video(str(video_path))
            video_paths = [video_path]
        else:
            logger.info(f"Found {len(video_paths)} videos in test_videos folder")
        
        return list(Path(self.config['TEST_IMAGE_PATH']).glob('*.jpg')), video_paths
    
    def create_test_video(self, output_path: str, duration: int = 5):
        """Create a synthetic test video"""
        width, height = 640, 480
        fps = 30
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Create moving objects for testing
        for i in range(fps * duration):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add moving rectangle (simulating person/object)
            x = int(width * (i / (fps * duration)))
            cv2.rectangle(frame, (x, 200), (x + 50, 250), (0, 255, 0), -1)
            
            # Add static rectangle (simulating background object)
            cv2.rectangle(frame, (width//2-25, height//2-25), 
                        (width//2+25, height//2+25), (0, 0, 255), -1)
            
            out.write(frame)
        
        out.release()
        logger.info(f"Created test video at {output_path}")
    
    def test_classification(self, image_paths: List[Path]):
        """Test classification agent"""
        logger.info("\n=== Testing Classification Agent ===")
        
        test_queries = [
            "classify this image",
            "what is in this image",
            "identify objects in this image"
        ]
        
        for image_path in image_paths:
            logger.info(f"\nProcessing image: {image_path.name}")
            
            for query in test_queries:
                try:
                    result = self.orchestrator.process_image(
                        image_path=str(image_path),
                        user_comment=query,
                        task_type=VisionTaskType.IMAGE_CLASSIFICATION  # Explicitly set task type
                    )
                    
                    logger.info(f"\nQuery: '{query}'")
                    
                    # Handle different result formats based on task type
                    if result.task_type == VisionTaskType.IMAGE_CLASSIFICATION:
                        if 'top_predictions' in result.results:
                            logger.info(f"Top 5 predictions:")
                            for pred in result.results['top_predictions']:
                                logger.info(f"- {pred['class']}: {pred['confidence']:.3f}")
                        else:
                            logger.info("No predictions found in classification results")
                            logger.debug(f"Available keys in results: {result.results.keys()}")
                    elif result.task_type == VisionTaskType.OBJECT_DETECTION:
                        logger.info(f"Detections:")
                        for det in result.results['detections']:
                            logger.info(f"- {det['class']}: {det['confidence']:.3f}")
                    
                    logger.info(f"Processing time: {result.processing_time:.3f}s")
                    logger.info(f"Task type: {result.task_type}")
                    
                except Exception as e:
                    logger.error(f"Error processing query '{query}' for image {image_path.name}: {str(e)}")
                    logger.debug("Result details:", exc_info=True)
                    continue
    
    def test_detection(self, image_paths: List[Path]):
        """Test detection agent with strict class filtering"""
        logger.info("\n=== Testing Detection Agent ===")
        
        test_queries = [
            # Format: (query, allowed_classes)
            ("detect person only", ["person"]),
            ("find people", ["person"]),
            ("detect all objects", None),  # None means no filtering
            ("find cars and people", ["person", "car"]),
        ]
        
        for image_path in image_paths:
            logger.info(f"\nProcessing image: {image_path.name}")
            
            for query, allowed_classes in test_queries:
                try:
                    # Create additional parameters for class filtering
                    additional_params = {'detect_classes': allowed_classes} if allowed_classes else None
                    
                    result = self.orchestrator.process_image(
                        image_path=str(image_path),
                        user_comment=query,
                        task_type=VisionTaskType.OBJECT_DETECTION,
                        additional_params=additional_params
                    )
                    
                    # Filter detections based on allowed classes
                    if allowed_classes:
                        filtered_detections = [
                            det for det in result.results['detections']
                            if det['class'].lower() in [cls.lower() for cls in allowed_classes]
                        ]
                        result.results['detections'] = filtered_detections
                        result.results['num_detections'] = len(filtered_detections)
                    
                    # Save visualization
                    output_path = os.path.join(
                        self.config['RESULTS_PATH'],
                        f'detection_{image_path.stem}_{query.replace(" ", "_")}.jpg'
                    )
                    
                    detection_agent = self.orchestrator.router.agents[VisionTaskType.OBJECT_DETECTION]
                    detection_agent.visualize_detections(
                        image_path=str(image_path),
                        detections=result.results['detections'],
                        output_path=output_path
                    )
                    
                    logger.info(f"\nQuery: '{query}'")
                    if allowed_classes:
                        logger.info(f"Filtering for classes: {allowed_classes}")
                    logger.info(f"Detections:")
                    for det in result.results['detections']:
                        logger.info(f"- {det['class']}: {det['confidence']:.3f}")
                    logger.info(f"Processing time: {result.processing_time:.3f}s")
                    logger.info(f"Visualization saved to: {output_path}")
                    
                except Exception as e:
                    logger.error(f"Error processing query '{query}' for image {image_path.name}: {str(e)}")
                    logger.debug("Result details:", exc_info=True)
                    continue
    
    def test_batch_processing(self, image_paths: List[Path]):
        """Test batch processing"""
        logger.info("\n=== Testing Batch Processing ===")
        
        results = self.orchestrator.process_batch(
            image_paths=[str(p) for p in image_paths],
            task_type=VisionTaskType.OBJECT_DETECTION,
            user_comment="detect objects"
        )
        
        logger.info(f"\nProcessed {len(results)} images in batch")
        for result in results:
            logger.info(f"\nResults for {Path(result.image_path).name}:")
            logger.info(f"Number of detections: {result.vision_output.results['num_detections']}")
            logger.info(f"Confidence: {result.vision_output.confidence:.3f}")

    def test_video_processing(self, video_paths: List[Path]):
        """Test video processing with specific queries"""
        logger.info("\n=== Testing Video Processing ===")
        
        test_queries = [
            # Format: (query, allowed_classes)
            ("detect person only in video", ["person"]),
            ("track people", ["person"]),
            ("detect all objects", None),
        ]
        
        for video_path in video_paths:
            logger.info(f"\nProcessing video: {video_path.name}")
            
            for query, allowed_classes in test_queries:
                try:
                    # Create additional parameters for class filtering
                    additional_params = {'detect_classes': allowed_classes} if allowed_classes else None
                    
                    output_path = os.path.join(
                        self.config['VIDEO_SAVE_PATH'],
                        f'output_{video_path.stem}_{query.replace(" ", "_")}.mp4'
                    )
                    
                    video_result = self.orchestrator.process_video(
                        video_path=str(video_path),
                        output_path=output_path,
                        user_comment=query,
                        start_time=0,
                        end_time=10,  # Process first 10 seconds
                        additional_params=additional_params
                    )
                    
                    logger.info(f"\nQuery: '{query}'")
                    if allowed_classes:
                        logger.info(f"Filtering for classes: {allowed_classes}")
                    logger.info(f"Processed {video_result.num_frames} frames")
                    logger.info(f"Average FPS: {video_result.num_frames / video_result.total_time:.2f}")
                    
                    # Calculate detection statistics
                    total_detections = sum(
                        len([d for d in frame.results['detections']
                            if not allowed_classes or d['class'].lower() in [c.lower() for c in allowed_classes]])
                        for frame in video_result.frames_results
                    )
                    avg_detections = total_detections / video_result.num_frames
                    
                    logger.info(f"Total detections: {total_detections}")
                    logger.info(f"Average detections per frame: {avg_detections:.2f}")
                    logger.info(f"Output saved to: {output_path}")
                    
                except Exception as e:
                    logger.error(f"Error processing query '{query}' for video {video_path.name}: {str(e)}")
                    logger.debug("Result details:", exc_info=True)
                    continue
    
    def validate_results(self):
        """Validate test results"""
        logger.info("\n=== Validating Test Results ===")
        
        # Check if result files exist
        result_files = list(Path(self.config['RESULTS_PATH']).rglob('*.jpg'))
        video_files = list(Path(self.config['VIDEO_SAVE_PATH']).glob('*.mp4'))
        
        logger.info(f"Generated {len(result_files)} detection visualizations")
        logger.info(f"Generated {len(video_files)} processed videos")
        
        # Validate detection results
        for file in result_files:
            img = Image.open(file)
            assert img.size[0] > 0 and img.size[1] > 0, f"Invalid image: {file}"
        
        # Validate video results
        for file in video_files:
            cap = cv2.VideoCapture(str(file))
            assert cap.isOpened(), f"Invalid video: {file}"
            cap.release()
        
        logger.info("All result files validated successfully")

    def debug_result(self, result):
        """Debug helper for examining result structure"""
        logger.debug("=== Result Debug Info ===")
        logger.debug(f"Task Type: {result.task_type}")
        logger.debug(f"Confidence: {result.confidence}")
        logger.debug(f"Processing Time: {result.processing_time}")
        logger.debug("Results keys: " + str(list(result.results.keys())))
        logger.debug("Results content: " + str(result.results))


def main():
    try:
        # Initialize tester
        logger.info("Initializing Vision Framework Tester...")
        tester = VisionFrameworkTester()
        
        # Download/prepare test assets
        logger.info("Preparing test assets...")
        image_paths, video_paths = tester.download_test_assets()
        
        # Run tests with proper error handling
        try:
            logger.info("Testing classification...")
            tester.test_classification(image_paths)
        except Exception as e:
            logger.error(f"Classification tests failed: {str(e)}")
        
        try:
            logger.info("Testing detection...")
            tester.test_detection(image_paths)
        except Exception as e:
            logger.error(f"Detection tests failed: {str(e)}")
        
        try:
            logger.info("Testing batch processing...")
            tester.test_batch_processing(image_paths)
        except Exception as e:
            logger.error(f"Batch processing tests failed: {str(e)}")
        
        try:
            logger.info("Testing video processing...")
            tester.test_video_processing(video_paths)
        except Exception as e:
            logger.error(f"Video processing tests failed: {str(e)}")
        
        # Validate results
        logger.info("Validating results...")
        tester.validate_results()
        
        logger.info("\nAll tests completed!")
        
    except Exception as e:
        logger.error(f"Critical error during testing: {str(e)}")
        logger.debug("Error details:", exc_info=True)
        raise

if __name__ == "__main__":
    main()