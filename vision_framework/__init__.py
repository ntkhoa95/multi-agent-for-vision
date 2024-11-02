from .core.types import (
    VisionTaskType,
    VisionInput,
    VisionOutput,
    BatchDetectionResult,
    VideoDetectionResult
)
from .orchestrator import VisionOrchestrator
from .core.config import DEFAULT_CONFIG
from .utils.logging import setup_logging

__version__ = "1.0.0"

# Setup default logging
setup_logging()
