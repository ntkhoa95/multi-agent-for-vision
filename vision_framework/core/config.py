import torch
from typing import Dict

DEFAULT_CONFIG = {
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'BATCH_SIZE': 32,
    'NUM_WORKERS': 4,
    'TEST_IMAGE_PATH': './test_images/',
    'RESULTS_PATH': './results/',
    'VIDEO_SAVE_PATH': './results/videos/',
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

def validate_config(config: Dict) -> Dict:
    """Validate and merge with default config"""
    final_config = DEFAULT_CONFIG.copy()
    final_config.update(config)
    return final_config