import logging
import time
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

from ..core.types import VisionInput, VisionOutput, VisionTaskType
from .base import BaseVisionAgent

logger = logging.getLogger(__name__)


class CaptioningAgent(BaseVisionAgent):
    """Enhanced captioning agent using VIT-GPT2 model."""

    def __init__(self, config: Dict):
        """Initialize VIT-GPT2 captioning agent."""
        # Initialize base parameters
        self.device = torch.device(
            config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Model configuration
        self.model_name = config.get("GIT_MODEL_NAME", "nlpconnect/vit-gpt2-image-captioning")
        self.max_length = config.get("MAX_CAPTION_LENGTH", 50)
        self.min_length = config.get("MIN_CAPTION_LENGTH", 5)
        self.num_beams = config.get("NUM_BEAMS", 3)
        self.temperature = config.get("TEMPERATURE", 1.0)
        self.no_repeat_ngram_size = config.get("NO_REPEAT_NGRAM_SIZE", 3)
        self.do_sample = config.get("DO_SAMPLE", False)

        logger.info(f"Initializing CaptioningAgent with device: {self.device}")
        logger.info(f"Using model: {self.model_name}")

        # Initialize processors
        try:
            self.image_processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Processors initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing processors: {str(e)}")
            raise

        # Initialize model
        try:
            logger.info("Loading captioning model...")
            self.model = self.load_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

        # Configure generation settings
        self.generation_config = {
            "max_length": self.max_length,
            "min_length": self.min_length,
            "num_beams": self.num_beams,
            "temperature": self.temperature,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "do_sample": self.do_sample,
            "early_stopping": True,
        }

        logger.info("CaptioningAgent initialized successfully")

    def load_model(self) -> VisionEncoderDecoderModel:
        """Load and configure the model."""
        model = VisionEncoderDecoderModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        model.eval()
        return model

    def generate_caption(self, image: Union[Image.Image, str], prompt: Optional[str] = None) -> str:
        """Generate caption for an image with optional prompt."""
        try:
            # Load and verify image
            if isinstance(image, str):
                image = self.load_image(image)
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Process image
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

            # Generate caption
            with torch.no_grad():
                output_ids = self.model.generate(
                    pixel_values=inputs.pixel_values, **self.generation_config
                )

            caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logger.debug(f"Generated caption: {caption}")
            return caption

        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            raise

    def process(self, vision_input: VisionInput) -> VisionOutput:
        """Process single image and generate caption."""
        start_time = time.time()

        try:
            caption = self.generate_caption(
                image=vision_input.image, prompt=vision_input.user_comment
            )

            processing_time = time.time() - start_time

            return VisionOutput(
                task_type=VisionTaskType.IMAGE_CAPTIONING,
                results={
                    "caption": caption,
                    "model_config": {
                        "model_name": self.model_name,
                        "generation_config": self.generation_config,
                    },
                },
                confidence=1.0,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error in caption processing: {str(e)}")
            raise

    def process_batch(
        self, image_paths: List[str], batch_size: Optional[int] = None, prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Process a batch of images with memory-efficient batching."""
        if not image_paths:
            return []

        batch_size = batch_size or min(4, len(image_paths))
        results = []

        try:
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i : i + batch_size]

                # Process images with error handling
                batch_images = []
                valid_paths = []
                for path in batch:
                    try:
                        batch_images.append(self.load_image(path))
                        valid_paths.append(path)
                    except Exception as e:
                        logger.error(f"Error loading image {path}: {str(e)}")
                        continue

                if not batch_images:
                    continue

                # Generate captions
                inputs = self.image_processor(batch_images, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        pixel_values=inputs.pixel_values, **self.generation_config
                    )

                captions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                results.extend(
                    [
                        {"path": path, "caption": caption}
                        for path, caption in zip(valid_paths, captions)
                    ]
                )

                # Clear memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return results

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise

    def clear_memory(self):
        """Clear GPU memory if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU memory cache")

    @staticmethod
    def get_memory_usage() -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0
