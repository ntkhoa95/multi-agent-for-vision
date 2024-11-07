# Vision Framework: Multi-Agent System for Computer Vision Tasks

<div>
<div>


[![codecov](https://codecov.io/gh/ntkhoa95/multi-agent-for-vision/branch/main/graph/badge.svg)](https://codecov.io/gh/ntkhoa95/multi-agent-for-vision)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img src="docs/demo-app-gradio.png" alt="Vision Framework Demo" width="800"/>

A flexible and extensible multi-agent framework for computer vision tasks, supporting classification, object detection, and more.

<details>
<summary>📋 Table of Contents</summary>

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Available Examples](#-available-examples)
- [Basic Usage](#-basic-usage)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [Supported Tasks](#-supported-tasks)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

</details>

## ✨ Features

<table>
<tr>
<td>

### 🤖 Multi-Agent Architecture
- Classification Agent (MobileNetV3)
- Object Detection Agent (YOLOv8)
- Easily extensible for new vision tasks

### 🗣️ Natural Language Interface
- "What's in this image?"
- "Detect objects in this scene"
- "Classify this image"
### 🗣️ Natural Language Interface
- "What's in this image?"
- "Detect objects in this scene"
- "Classify this image"

</td>
<td>

### 🎯 Intelligent Task Routing
- Automatic agent selection
- Query-based task determination
- Flexible routing system

### 📊 Comprehensive Output
- Detailed predictions with confidence scores
- Processing time metrics
- Model configuration details
</td>
<td>

### 🎯 Intelligent Task Routing
- Automatic agent selection
- Query-based task determination
- Flexible routing system

### 📊 Comprehensive Output
- Detailed predictions with confidence scores
- Processing time metrics
- Model configuration details

</td>
</tr>
</table>

</td>
</tr>
</table>

## 🚀 Quick Start

<details>
<summary>Installation Steps</summary>
<details>
<summary>Installation Steps</summary>

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
</details>

## 🎯 Demo Application

The Vision Framework includes a user-friendly web interface built with Gradio:

```bash
# Run the demo application
python examples/gradio_demo.py
```

<div>
<img src="docs/demo-app-gradio.png" alt="Demo Application Interface" width="800"/>
</div>

- Interactive web interface
- Real-time inference
- Support for both classification and detection
- Natural language query processing
- Visual results with bounding boxes and labels
</details>

## 🎯 Demo Application

The Vision Framework includes a user-friendly web interface built with Gradio:

```bash
# Run the demo application
python examples/gradio_demo.py
```

<div>
<img src="docs/demo-app-gradio.png" alt="Demo Application Interface" width="800"/>
</div>

- Interactive web interface
- Real-time inference
- Support for both classification and detection
- Natural language query processing
- Visual results with bounding boxes and labels

## 🗂️ Available Examples

<details>
<summary>1. Classification Example (classification_example.py)</summary>

```python
# Run the classification example
python examples/classification_example.py
```

### Features:
- Image classification with detailed class predictions
- Confidence scores for top-5 predictions
- Model configuration display
- Processing time metrics

### Example Output:
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
</details>
</details>

<details>
<summary>2. Object Detection Example (detection_example.py)</summary>

```python
# Run the detection example
python examples/detection_example.py
```

### Features:
- Multiple object detection
- Bounding box coordinates
- Class predictions with confidence scores
- Processing time metrics
</details>

<details>
<summary>3. Video Processing Example (video_processing_example.py)</summary>

```python
# Run the video processing example
python examples/video_processing_example.py
```
</details>

## 💻 Basic Usage

### Using the Gradio Interface

Run the demo application:
```bash
python app/gradio_demo.py
```

This launches a web interface where you can:
- Upload images
- Select vision tasks
- Enter natural language queries
- View results with visualizations

### Programmatic Usage

#### Image Classification
```python
from vision_framework import VisionOrchestrator, VisionTaskType

# Initialize orchestrator
config = {
    "MODEL_NAME": "mobilenetv3_large_100",
    "MODEL_PRETRAINED": True
}
orchestrator = VisionOrchestrator(config)

# Process image
result = orchestrator.process_image(
    image_path="path/to/image.jpg",
    task_type=VisionTaskType.IMAGE_CLASSIFICATION,
    user_comment="classify this image"
)
```

#### Object Detection
```python
# Configure detection
config = {
    "YOLO_MODEL_NAME": "yolov8s.pt",
    "YOLO_CONFIDENCE_THRESHOLD": 0.25,
    "ENABLE_TRACK": True
}
orchestrator = VisionOrchestrator(config)

# Detect objects
result = orchestrator.process_image(
    image_path="path/to/image.jpg",
    task_type=VisionTaskType.OBJECT_DETECTION,
    user_comment="detect people and cars"
)
```

#### Image Captioning
```python
# Enable captioning
config = {
    "ENABLE_CAPTIONING": True,
    "MAX_CAPTION_LENGTH": 50
}
orchestrator = VisionOrchestrator(config)

# Generate caption
result = orchestrator.process_image(
    image_path="path/to/image.jpg",
    task_type=VisionTaskType.IMAGE_CAPTIONING,
    user_comment="describe this image"
)
```

#### Video Processing
```python
# Process video with tracking
result = orchestrator.process_video(
    video_path="path/to/video.mp4",
    output_path="output.mp4",
    user_comment="track vehicles"
)
```

## 🏗️ Architecture

<details>
<summary>Project Structure</summary>

```
multi-agents/
├── app/
│   └── gradio_demo.py           # Interactive web interface
├── docs/
│   └── images/                  # Documentation images
├── examples/
│   ├── classification_example.py # Image classification demo
│   ├── detection_example.py     # Object detection demo
│   ├── video_processing_example.py  # Video analysis demo
│   ├── vision_captioning_example.py # Image captioning demo
│   └── video_vision_captioning_example.py
├── tests/
│   ├── data/
│   │   ├── images/             # Test images
│   │   └── videos/             # Test videos
│   ├── integration/
│   │   ├── test_image.py       # Image processing tests
│   │   └── test_video.py       # Video processing tests
│   └── unit/
│       ├── test_base.py        # Base agent tests
│       ├── test_classification.py
│       ├── test_detection.py
│       ├── test_nlp.py
│       ├── test_orchestrator.py
│       └── test_router.py
├── vision_framework/
│   ├── agents/
│   │   ├── base.py            # Base agent class
│   │   ├── classification.py  # MobileNet classifier
│   │   ├── detection.py      # YOLO detector
│   │   └── captioning.py     # VIT-GPT2 captioner
│   ├── core/
│   │   ├── config.py         # Configuration management
│   │   └── types.py          # Type definitions
│   ├── nlp/
│   │   └── processor.py      # Query processing
│   ├── router/
│   │   └── router.py         # Task routing
│   ├── utils/
│   │   ├── image.py          # Image utilities
│   │   ├── logging.py        # Logging setup
│   │   └── video.py          # Video utilities
│   └── orchestrator.py       # Main orchestrator
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
├── setup.py                 # Package setup
├── pytest.ini              # Test configuration
└── README.md               # This file
```
</details>

### Framework Workflow

```mermaid
flowchart TB
    subgraph Input
        A[Image/Video] --> C{VisionInput}
        B[User Comment] --> C
        P[Additional Params] -.-> C
    end

    subgraph Orchestrator
        C --> D[VisionOrchestrator]
        D --> E[AgentRouter]
        D --> NLP[NLP Processor]

        subgraph Router Logic
            E --> F{Task Type\nSpecified?}
            F -->|Yes| H[Get Registered Agent]
            F -->|No| G[Parse Query\nIntent]
            NLP --> G
            G --> H
        end
    end

    subgraph Agents
        H --> I{Select Agent}
        I -->|Object Detection| J[YOLO Detection\nAgent  ByteTrack]
        I -->|Classification| K[MobileNet\nClassification Agent]
        I -->|Captioning| L[VIT-GPT2\nCaptioning Agent]
    end

    subgraph Processing
        J --> O[Process Input]
        K --> O
        L --> O
        O --> V{Input Type}
        V -->|Image| Q1[Process Image]
        V -->|Video| Q2[Process Video]
        Q1 --> R[Create VisionOutput]
        Q2 --> R
    end

    R --> S[Return Results]
```

### Core Components

1. **Vision Orchestrator**: Central component managing the interaction between agents
2. **Agents**: Specialized modules for specific vision tasks
3. **Router**: Determines appropriate agent based on user queries
4. **NLP Processor**: Interprets natural language queries

## ⚙️ Configuration

### General Settings
```python
config = {
    "DEVICE": "cuda",  # or "cpu"
    "BATCH_SIZE": 32,
    "NUM_WORKERS": 4
}
```

### Classification Settings
```python
config = {
    "MODEL_NAME": "mobilenetv3_large_100",
    "MODEL_PRETRAINED": True
}
```

### Detection Settings
```python
config = {
    "YOLO_MODEL_NAME": "yolov8s.pt",
    "YOLO_CONFIDENCE_THRESHOLD": 0.25,
    "YOLO_IOU_THRESHOLD": 0.45,
    "DETECTION_IMAGE_SIZE": 640,
    "ENABLE_TRACK": True
}
```

### Captioning Settings
```python
config = {
    "ENABLE_CAPTIONING": True,
    "MAX_CAPTION_LENGTH": 50,
    "MIN_CAPTION_LENGTH": 5,
    "NUM_BEAMS": 3,
    "TEMPERATURE": 1.0
}
```

## 📊 Supported Tasks

### 1. Image Classification
- Identifies main subjects in images
- Returns top-5 predictions with confidence scores
- Supports ImageNet classes

### 2. Object Detection
- Locates and identifies multiple objects
- Provides bounding boxes and confidence scores
- Supports COCO classes

## 🛠️ Development

<details>
<summary>Setting Up Development Environment</summary>

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Download required NLTK data
python -c "import nltk; nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm

# Download required NLTK data
python -c "import nltk; nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm
```
</details>

### Running Tests
```bash
pytest tests/
```

## 🤝 Contributing

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
- [Gradio](https://gradio.app/) for the demo interface
- [Gradio](https://gradio.app/) for the demo interface

## 📧 Contact

<div>

**Khoa Nguyen**

[![Email](https://img.shields.io/badge/Email-toankhoabk%40gmail.com-blue?style=flat-square&logo=gmail)](mailto:toankhoabk@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-ntkhoa95-black?style=flat-square&logo=github)](https://github.com/ntkhoa95/multi-agent-for-vision)

</div>
