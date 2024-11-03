# Vision Framework: Multi-Agent System for Computer Vision Tasks

[![CI](https://github.com/{username}/{repo}/actions/workflows/ci.yml/badge.svg)](https://github.com/{username}/{repo}/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/{username}/{repo}/branch/main/graph/badge.svg)](https://codecov.io/gh/{username}/{repo})
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A flexible and extensible multi-agent framework for computer vision tasks, supporting classification, object detection, and more. This framework provides a modular approach to handling various vision tasks through specialized agents.

## 🌟 Features

- **Multi-Agent Architecture**: Specialized agents for different vision tasks
  - Classification Agent (MobileNetV3)
  - Object Detection Agent (YOLOv8)
  - Easily extensible for new vision tasks

- **Natural Language Interface**: Process vision tasks using natural language queries
  - "What's in this image?"
  - "Detect objects in this scene"
  - "Classify this image"

- **Intelligent Task Routing**: Automatically determines the most appropriate agent based on user queries

- **Modern Deep Learning Models**:
  - MobileNetV3 for efficient classification
  - YOLOv8 for state-of-the-art object detection
  - Support for model customization and extension

- **Comprehensive Output Format**:
  - Detailed predictions with confidence scores
  - Processing time metrics
  - Model configuration details

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ntkhoa95/multi-agent-for-vision.git
cd multi-agent-for-vision

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Install in development mode
pip install -e .
```

## 🗂️ Available Examples

### 1. Classification Example (`classification_example.py`)

Demonstrates image classification capabilities using MobileNetV3.

```python
# Run the classification example
python examples/classification_example.py
```

#### Features:
- Image classification with detailed class predictions
- Confidence scores for top-5 predictions
- Model configuration display
- Processing time metrics

#### Example Output:
```
Processing image: tests/data/images/dog.jpg
Query: What is in this image?
Task Type: VisionTaskType.IMAGE_CLASSIFICATION
Classification Results:
  Golden retriever: 0.856
  Labrador retriever: 0.125
  Irish setter: 0.012
  Chesapeake Bay retriever: 0.004
  Greater Swiss Mountain dog: 0.003

Model Configuration:
  Model: mobilenetv3_large_100
  Input size: (3, 224, 224)
  Interpolation: bicubic
  Mean: (0.485, 0.456, 0.406)
  Std: (0.229, 0.224, 0.225)
Processing time: 0.064 seconds
```

### 2. Object Detection Example (`detection_example.py`)

Demonstrates object detection capabilities using YOLOv8.

```python
# Run the detection example
python examples/detection_example.py
```

#### Features:
- Multiple object detection
- Bounding box coordinates
- Class predictions with confidence scores
- Processing time metrics

### 3. Video Processing Example (`video_processing_example.py`)

Shows how to process video inputs for both classification and detection tasks.

```python
# Run the video processing example
python examples/video_processing_example.py
```

## Basic Usage

```python
from vision_framework import VisionOrchestrator
from vision_framework.core.types import VisionTaskType

# Initialize the framework
config = {
    'DEVICE': 'cuda',  # or 'cpu'
    'MODEL_NAME': 'mobilenetv3_large_100',
    'MODEL_PRETRAINED': True
}
orchestrator = VisionOrchestrator(config)

# Process an image
result = orchestrator.process_image(
    image_path="path/to/image.jpg",
    user_comment="What is in this image?"
)

# Print results
print(f"Task Type: {result.task_type}")
print(f"Confidence: {result.confidence}")
print("Predictions:", result.results)
```

## 🏗️ Architecture

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

### Components

1. **Vision Orchestrator**: Central component managing the interaction between agents
2. **Agents**: Specialized modules for specific vision tasks
3. **Router**: Determines appropriate agent based on user queries
4. **NLP Processor**: Interprets natural language queries

## 🔧 Configuration

Key configuration options:

```python
config = {
    'DEVICE': 'cuda',                    # Device for inference
    'MODEL_NAME': 'mobilenetv3_large_100', # Classification model
    'MODEL_PRETRAINED': True,            # Use pretrained weights
    'BATCH_SIZE': 1,                     # Batch size for inference
    'NUM_WORKERS': 0,                    # Workers for data loading
    'YOLO_MODEL_NAME': 'yolov8s.pt',     # Detection model
    'YOLO_CONFIDENCE_THRESHOLD': 0.25,    # Detection confidence
    'YOLO_IOU_THRESHOLD': 0.45,          # Detection IOU threshold
}
```

## 📊 Supported Tasks

1. **Image Classification**
   - Identifies main subjects in images
   - Returns top-5 predictions with confidence scores
   - Supports ImageNet classes

2. **Object Detection**
   - Locates and identifies multiple objects
   - Provides bounding boxes and confidence scores
   - Supports COCO classes

## 🛠️ Development

### Setting up development environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running tests

```bash
pytest tests/
```

### Code Style

This project follows:
- Black for code formatting
- isort for import sorting
- flake8 for code linting
- mypy for type checking

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [timm](https://github.com/rwightman/pytorch-image-models) for efficient model implementations
- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [PyTorch](https://pytorch.org/) for the deep learning framework

## 📧 Contact

Khoa Nguyen - [toankhoabk@gmail.com]

Project Link: https://github.com/ntkhoa95/multi-agent-for-vision.git