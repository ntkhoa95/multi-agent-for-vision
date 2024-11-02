import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple

def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw detection boxes and labels on image"""
    image_copy = image.copy()
    
    for det in detections:
        bbox = det['bbox']
        label = f"{det['class']} {det['confidence']:.2f}"
        if 'track_id' in det:
            label += f" ID:{det['track_id']}"
        
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_copy, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image_copy