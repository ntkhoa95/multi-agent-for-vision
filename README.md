# Vision Framework

A flexible and extensible computer vision framework supporting multiple vision tasks with natural language queries.

## Features

- Multiple vision tasks support:
  - Object Detection (YOLOv8)
  - Image Classification (MobileNetV3)
  - Natural language query understanding
  - Object tracking in videos
  - Batch processing capability

- Natural Language Processing:
  - Query understanding and task determination
  - Object extraction from queries
  - Dynamic class filtering

- Video Processing:
  - Real-time object detection and tracking
  - Progress visualization
  - Frame-by-frame processing
  - Output video generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vision-framework.git
cd vision-framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```python
from vision_framework import VisionOrchestrator

# Initialize framework
config = {
    'DEVICE': 'cuda',  # or 'cpu'
    'YOLO_MODEL_NAME': 'yolov8s.pt',
    'MODEL_NAME': 'mobilenetv3_large_100',
}
orchestrator = VisionOrchestrator(config)

# Process single image
result = orchestrator.process_image(
    image_path="images/test.jpg",
    user_comment="detect cats and dogs"
)

# Process video
result = orchestrator.process_video(
    video_path="videos/test.mp4",
    user_comment="find people and cars",
    output_path="results/output.mp4"
)
```

### Natural Language Queries

Examples of supported queries:
- "detect cats and dogs in this image"
- "find all people in the video"
- "classify this image"
- "locate cars and trucks"
- "find pedestrians crossing the street"

### Batch Processing

```python
# Process multiple images
results = orchestrator.process_batch(
    image_paths=["image1.jpg", "image2.jpg"],
    user_comment="detect objects",
)
```

## Configuration

Key configuration options:
```python
config = {
    'DEVICE': 'cuda',  # or 'cpu'
    'BATCH_SIZE': 32,
    'NUM_WORKERS': 4,
    'YOLO_MODEL_NAME': 'yolov8s.pt',
    'YOLO_CONFIDENCE_THRESHOLD': 0.25,
    'YOLO_IOU_THRESHOLD': 0.45,
    'MODEL_NAME': 'mobilenetv3_large_100',
    'MODEL_PRETRAINED': True,
    'ENABLE_TRACK': True,
}
```

## Project Structure

```
vision_framework/
├── core/
│   ├── types.py           # Core data types
│   └── config.py          # Configuration handling
├── agents/
│   ├── base.py           # Base agent class
│   ├── classification.py  # Classification agent
│   └── detection.py      # Detection agent
├── nlp/
│   └── processor.py      # NLP processing
├── utils/
│   ├── image.py          # Image utilities
│   └── video.py          # Video utilities
└── router/
    └── router.py         # Agent router
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_detection.py

# Install test dependencies
pip install pytest pytest-cov

# Run tests with coverage report
pytest --cov=vision_framework tests/
```

## Requirements

See `requirements.txt` for full list of dependencies.

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
