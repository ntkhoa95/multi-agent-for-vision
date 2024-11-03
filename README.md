# Vision Framework

[![CI](https://github.com/{username}/{repo}/actions/workflows/ci.yml/badge.svg)](https://github.com/{username}/{repo}/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/{username}/{repo}/branch/main/graph/badge.svg)](https://codecov.io/gh/{username}/{repo})
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
├── setup.py                # Package setup and dependencies
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── LICENSE                 # License file
├── .gitignore              # Git ignore file
│
├── vision_framework/       # Main package directory
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration management
│   ├── orchestrator.py     # Vision Orchestrator management
│   │
│   ├── core/               # Core functionality
│   │   ├── __init__.py
│   │   ├── types.py        # Data classes and type definitions
│   │   └── exceptions.py   # Custom exceptions
│   │
│   ├── agents/             # Vision agents
│   │   ├── __init__.py
│   │   ├── base.py         # Base agent class
│   │   ├── classification.py
│   │   └── detection.py
│   │
│   ├── nlp/                # NLP processing
│   │   ├── __init__.py
│   │   └── processor.py
│   │
│   ├── router/             # Request routing
│   │   ├── __init__.py
│   │   └── router.py
│   │
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── image.py
│       ├── video.py
│       └── logging.py
│
├── tests/                   # Test directory
│   ├── __init__.py
│   ├── conftest.py          # Pytest configuration and fixtures
│   │
│   ├── unit/               # Unit tests
│   │   ├── __init__.py
│   │   ├── test_base.py
│   │   ├── test_classification.py
│   │   ├── test_detection.py
│   │   ├── test_nlp.py
│   │   ├── test_router.py
│   │   └── test_orchestrator.py
│   │
│   ├── integration/        # Integration tests
│   │   ├── __init__.py
│   │   ├── test_image.py
│   │   └── test_video.py
│   │
│   └── data/               # Test data
│       ├── images/         # Test images
│       └── videos/         # Test videos
│
└── examples/               # Example scripts
    ├── classification_example.py
    ├── detection_example.py
    └── video_processing_example.py
```

## Install package in development mode
```
pip install -e .
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test types
pytest tests/unit
pytest tests/integration

# Run with coverage
pytest --cov=vision_framework

# Run specific test file
pytest tests/unit/test_detection.py
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
