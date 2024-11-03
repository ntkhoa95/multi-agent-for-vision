from .core.config import DEFAULT_CONFIG
from .core.types import (
    BatchDetectionResult,
    VideoDetectionResult,
    VisionInput,
    VisionOutput,
    VisionTaskType,
)
from .orchestrator import VisionOrchestrator
from .utils.logging import setup_logging

__version__ = "1.0.0"

# Setup default logging
setup_logging()
