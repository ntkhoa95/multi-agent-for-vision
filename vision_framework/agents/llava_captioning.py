import logging
import time
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..core.types import VisionInput, VisionOutput, VisionTaskType
from .base import BaseVisionAgent

logger = logging.getLogger(__name__)


class LLaVaCaptioningAgent(BaseVisionAgent):
    def __init__(self, config: Dict):
        """Initialize LLaVa captioning agent."""
        # Set parameters before initializing base class
        self.max_length = config.get("MAX_CAPTION_LENGTH", 32)
        self.min_length = config.get("MIN_CAPTION_LENGTH", 8)
        self.batch_size = config.get("BATCH_SIZE", 16)
        self.num_beams = config.get("NUM_BEAMS", 4)

        # Initialize processor first
        logger.info("Loading LLaVa processor...")
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        super().__init__(config)
        self.model = self.load_model()
        logger.info("LLaVaCaptioningAgent initialized successfully")

    def load_model(self):
        """Load the LLaVa model."""
        try:
            logger.info("Loading LLaVa model...")
            model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                torch_dtype=torch.float16,  # if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            model.eval()
            logger.info("LLaVa model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading LLaVa model: {str(e)}")
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
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt or "What is shown in this image?"},
                    ],
                },
            ]
            prompt_template = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            inputs = self.processor(
                images=image, text=prompt_template, return_tensors="pt", padding=True
            ).to(self.device, torch.float16)

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
            caption = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
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

                # Prepare batch prompts
                conversations = [
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt or "What is shown in this image?"},
                            ],
                        },
                    ]
                    for _ in batch_images
                ]
                prompts = [
                    self.processor.apply_chat_template(conv, add_generation_prompt=True)
                    for conv in conversations
                ]

                # Process batch
                inputs = self.processor(
                    images=batch_images, text=prompts, return_tensors="pt", padding=True
                ).to(self.device, torch.float16)

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
