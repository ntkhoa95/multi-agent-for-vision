import pytest
import torch
from vision_framework import VisionOrchestrator

@pytest.fixture
def test_config():
    return {
        'DEVICE': 'cpu',  # Use CPU for testing
        'BATCH_SIZE': 2,
        'NUM_WORKERS': 0,
        'TEST_IMAGE_PATH': './test_images/',
        'RESULTS_PATH': './test_results/',
        'VIDEO_SAVE_PATH': './test_results/videos/',
        'MODEL_NAME': 'mobilenetv3_large_100',
        'MODEL_PRETRAINED': True,
        'USE_FP16': False,
        'YOLO_MODEL_NAME': 'yolov8n.pt',  # Use nano model for faster testing
        'YOLO_CONFIDENCE_THRESHOLD': 0.25,
        'YOLO_IOU_THRESHOLD': 0.45,
        'DETECTION_IMAGE_SIZE': 640,
        'VIDEO_FPS': 30,
        'ENABLE_TRACK': True,
    }

@pytest.fixture
def orchestrator(test_config):
    return VisionOrchestrator(test_config)

@pytest.fixture
def sample_image(tmp_path):
    """Create a sample test image"""
    import numpy as np
    from PIL import Image
    
    # Create a simple test image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[200:400, 200:400] = 255  # White square
    
    image_path = tmp_path / "test.jpg"
    Image.fromarray(img).save(image_path)
    
    return str(image_path)

@pytest.fixture
def sample_video(tmp_path):
    """Create a sample test video"""
    import cv2
    import numpy as np
    
    video_path = str(tmp_path / "test.mp4")
    fps = 30
    frame_size = (640, 480)
    
    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        frame_size
    )
    
    # Create 30 frames (1 second)
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add moving object
        x = int(640 * (i / 30))
        cv2.rectangle(frame, (x, 200), (x + 50, 250), (255, 255, 255), -1)
        out.write(frame)
    
    out.release()
    return video_path