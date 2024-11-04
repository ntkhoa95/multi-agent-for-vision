import logging
import time
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

from ..core.types import VisionInput, VisionOutput, VisionTaskType
from .base import BaseVisionAgent

logger = logging.getLogger(__name__)


class GITCaptioningAgent(BaseVisionAgent):
    """Custom captioning agent using Microsoft's GIT model."""

    def __init__(self, config: Dict):
        """Initialize GIT captioning agent."""
        # Set parameters
        self.max_length = config.get("MAX_CAPTION_LENGTH", 50)
        self.min_length = config.get("MIN_CAPTION_LENGTH", 5)
        self.num_beams = config.get("NUM_BEAMS", 3)
        self.model_name = config.get("GIT_MODEL_NAME", "microsoft/git-base-coco")

        # Initialize processor first
        logger.info(f"Loading GIT processor from {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # Call base class __init__
        super().__init__(config)
        logger.info("GITCaptioningAgent initialized successfully")

        # Set specific generation config
        self.generation_config = {
            "num_beams": self.num_beams,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "length_penalty": 1.0,
            "num_return_sequences": 1,
            "do_sample": False,
        }

    def load_model(self):
        """Load the GIT model."""
        try:
            logger.info(f"Loading GIT model from {self.model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device)
            model.eval()
            logger.info("GIT model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Error loading GIT model: {str(e)}")
            raise

    def generate_caption(self, image: Union[Image.Image, str], prompt: Optional[str] = None) -> str:
        """Generate caption for an image."""
        try:
            # Load and verify image
            if isinstance(image, str):
                logger.debug(f"Loading image from path: {image}")
                image = self.load_image(image)
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Prepare inputs
            logger.debug("Preparing inputs for caption generation")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(
                self.device
            )

            # Generate caption
            logger.debug("Generating caption")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=pixel_values, **self.generation_config
                )

            # Decode caption
            generated_caption = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            logger.info(f"Generated caption: {generated_caption}")
            return generated_caption

        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}", exc_info=True)
            raise

    def process(self, vision_input: VisionInput) -> VisionOutput:
        """Process single image and generate caption."""
        start_time = time.time()

        try:
            # Generate caption
            caption = self.generate_caption(
                image=vision_input.image, prompt=vision_input.user_comment
            )

            # Create output
            processing_time = time.time() - start_time
            output = VisionOutput(
                task_type=VisionTaskType.IMAGE_CAPTIONING,
                results={
                    "caption": caption,
                    "model_config": {
                        "model_name": self.model_name,
                        "max_length": self.max_length,
                        "min_length": self.min_length,
                        "num_beams": self.num_beams,
                    },
                },
                confidence=1.0,
                processing_time=processing_time,
            )

            logger.info(f"Caption generated in {processing_time:.2f} seconds")
            return output

        except Exception as e:
            logger.error(f"Error in caption processing: {str(e)}", exc_info=True)
            raise

    def process_batch(
        self, image_paths: List[str], batch_size: Optional[int] = None, prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Process a batch of images and generate captions."""
        if not image_paths:
            return []

        batch_size = batch_size or min(
            4, self.batch_size
        )  # Smaller batch size for memory efficiency
        results = []
        logger.info(f"Processing batch of {len(image_paths)} images")

        try:
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i : i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}")

                # Load images with error handling
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

                # Process batch
                pixel_values = self.processor(
                    images=batch_images, return_tensors="pt"
                ).pixel_values.to(self.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values=pixel_values, **self.generation_config
                    )

                captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

                # Store results
                results.extend(
                    [
                        {"path": path, "caption": caption.strip()}
                        for path, caption in zip(valid_paths, captions)
                    ]
                )

            logger.info(f"Successfully processed {len(results)} images")
            return results

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def get_memory_usage():
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2  # MB
        return 0


class CustomCaptioningAgent(BaseVisionAgent):
    """Custom captioning agent using VIT-GPT2 model."""

    def __init__(self, config: Dict):
        """Initialize VIT-GPT2 captioning agent."""
        # First initialize device
        self.device = torch.device(
            config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Using device: {self.device}")

        # Set parameters
        self.max_length = config.get("MAX_CAPTION_LENGTH", 50)
        self.min_length = config.get("MIN_CAPTION_LENGTH", 5)
        self.num_beams = config.get("NUM_BEAMS", 3)
        self.model_name = config.get("GIT_MODEL_NAME", "nlpconnect/vit-gpt2-image-captioning")

        # Initialize processors
        try:
            self.image_processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Processors initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing processors: {str(e)}")
            raise

        try:
            # Initialize model
            logger.info("Loading captioning model...")
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")

            logger.info("CustomCaptioningAgent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing CustomCaptioningAgent: {str(e)}")
            raise

        # Configure generation settings
        self.generation_config = {
            "max_length": self.max_length,
            "min_length": self.min_length,
            "num_beams": self.num_beams,
            "early_stopping": True,
        }

        # # Call parent class initialization
        # super().__init__(config)

    def load_model(self):
        """Return the already initialized model."""
        return self.model

    def generate_caption(self, image: Union[Image.Image, str], prompt: Optional[str] = None) -> str:
        """Generate caption for an image."""
        try:
            # Load and verify image
            if isinstance(image, str):
                logger.debug(f"Loading image from path: {image}")
                image = self.load_image(image)
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Prepare inputs
            logger.debug("Preparing inputs for caption generation")
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

            # Generate caption
            logger.debug("Generating caption")
            with torch.no_grad():
                output_ids = self.model.generate(
                    pixel_values=inputs.pixel_values, **self.generation_config
                )

            # Decode caption
            caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logger.info(f"Generated caption: {caption}")
            return caption

        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}", exc_info=True)
            raise

    def process(self, vision_input: VisionInput) -> VisionOutput:
        """Process single image and generate caption."""
        start_time = time.time()

        try:
            # Generate caption
            caption = self.generate_caption(
                image=vision_input.image, prompt=vision_input.user_comment
            )

            # Create output
            processing_time = time.time() - start_time
            output = VisionOutput(
                task_type=VisionTaskType.IMAGE_CAPTIONING,
                results={
                    "caption": caption,
                    "model_config": {
                        "model_name": self.model_name,
                        "max_length": self.max_length,
                        "min_length": self.min_length,
                        "num_beams": self.num_beams,
                    },
                },
                confidence=1.0,
                processing_time=processing_time,
            )

            logger.info(f"Caption generated in {processing_time:.2f} seconds")
            return output

        except Exception as e:
            logger.error(f"Error in caption processing: {str(e)}", exc_info=True)
            raise

    def process_batch(
        self, image_paths: List[str], batch_size: Optional[int] = None, prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Process a batch of images and generate captions."""
        if not image_paths:
            return []

        batch_size = batch_size or min(4, len(image_paths))
        results = []
        logger.info(f"Processing batch of {len(image_paths)} images")

        try:
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i : i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}")

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

                # Process batch with attention masks
                inputs = self.image_processor(batch_images, return_tensors="pt")
                pixel_values = inputs.pixel_values.to(self.device)

                # Create attention masks for batch
                attention_mask = torch.ones(
                    pixel_values.shape[0],
                    self.model.config.encoder_attention_heads,
                    dtype=torch.long,
                    device=self.device,
                )

                with torch.no_grad():
                    output_ids = self.model.generate(
                        pixel_values, attention_mask=attention_mask, **self.generation_config
                    )

                captions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                # Store results
                results.extend(
                    [
                        {"path": path, "caption": caption}
                        for path, caption in zip(valid_paths, captions)
                    ]
                )

            logger.info(f"Successfully processed {len(results)} images")
            return results

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}", exc_info=True)
            raise

    def clear_gpu_memory(self):
        """Clear GPU memory if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU memory cache")

    @staticmethod
    def get_memory_usage():
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2  # MB
        return 0
