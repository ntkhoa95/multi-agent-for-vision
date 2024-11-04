import logging
import time
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from ..core.types import VisionInput, VisionOutput, VisionTaskType
from .base import BaseVisionAgent

logger = logging.getLogger(__name__)


class BlipCaptioningAgent(BaseVisionAgent):
    def __init__(self, config: Dict):
        """Initialize BLIP captioning agent."""
        # Set parameters before initializing base class
        self.max_length = config.get("MAX_CAPTION_LENGTH", 32)
        self.min_length = config.get("MIN_CAPTION_LENGTH", 8)
        self.batch_size = config.get("BATCH_SIZE", 16)
        self.num_beams = config.get("NUM_BEAMS", 4)

        # Initialize processor first
        logger.info("Loading BLIP processor...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        # Call base class __init__ which will call load_model and set self.model
        super().__init__(config)
        self.model = self.load_model()
        logger.info("BlipCaptioningAgent initialized successfully")

    def load_model(self):
        """Load the BLIP model."""
        try:
            logger.info("Loading BLIP model...")
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device)
            model.eval()
            logger.info("BLIP model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading BLIP model: {str(e)}")
            raise

    def generate_caption(self, image: Union[Image.Image, str], prompt: Optional[str] = None) -> str:
        """Generate caption for an image."""
        try:
            if self.model is None:
                raise ValueError("Model is not loaded properly")

            # Convert string path to PIL Image if needed
            if isinstance(image, str):
                logger.debug(f"Loading image from path: {image}")
                image = self.load_image(image)
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")

            logger.debug("Preparing inputs for caption generation")
            # Prepare inputs
            inputs = self.processor(
                images=image, text=prompt, return_tensors="pt", padding=True
            ).to(self.device)

            logger.debug("Generating caption")
            # Generate caption using the model
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    min_length=self.min_length,
                    num_beams=self.num_beams,
                    repetition_penalty=1.0,
                )

            # Decode caption
            caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
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

        batch_size = batch_size or self.batch_size
        results = []
        logger.info(f"Processing batch of {len(image_paths)} images")

        try:
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i : i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}")

                batch_images = []
                for path in batch:
                    try:
                        batch_images.append(self.load_image(path))
                    except FileNotFoundError as e:
                        logger.error(e)
                        continue

                # Process batch
                inputs = self.processor(
                    images=batch_images,
                    text=[prompt] * len(batch) if prompt else None,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        min_length=self.min_length,
                        num_beams=self.num_beams,
                        repetition_penalty=1.0,
                    )

                captions = self.processor.batch_decode(output_ids, skip_special_tokens=True)

                # Store results
                results.extend(
                    [{"path": path, "caption": caption} for path, caption in zip(batch, captions)]
                )

            logger.info(f"Successfully processed {len(results)} images")
            return results

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}", exc_info=True)
            raise
