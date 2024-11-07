# examples/gradio_demo.py
import logging
import sys
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))
from vision_framework import VisionOrchestrator, VisionTaskType
from vision_framework.nlp.processor import NLPProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionDemo:
    def __init__(self):
        self.config = {
            "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
            "MODEL_NAME": "mobilenetv3_large_100",
            "MODEL_PRETRAINED": True,
            "BATCH_SIZE": 1,
            "NUM_WORKERS": 0,
            "YOLO_MODEL_NAME": "yolov8s.pt",
            "YOLO_CONFIDENCE_THRESHOLD": 0.25,
            "YOLO_IOU_THRESHOLD": 0.45,
            "ENABLE_CAPTIONING": True,
            "MAX_CAPTION_LENGTH": 50,
            "MIN_CAPTION_LENGTH": 10,
        }
        self.orchestrator = VisionOrchestrator(self.config)
        self.nlp_processor = NLPProcessor()
        logger.info(f"Using device: {self.config['DEVICE']}")

        # Define distinct colors for different classes (BGR format)
        self.colors = {
            "person": (0, 255, 0),  # Green
            "car": (255, 0, 0),  # Blue
            "truck": (0, 0, 255),  # Red
            "traffic light": (255, 255, 0),  # Cyan
            "bus": (255, 0, 255),  # Magenta
            "motorcycle": (0, 255, 255),  # Yellow
            "bicycle": (128, 0, 0),  # Dark blue
            "dog": (0, 128, 0),  # Dark green
            "cat": (0, 0, 128),  # Dark red
            # Furniture
            "chair": (120, 180, 0),  # Green
            "couch": (180, 120, 0),  # Brown
            "dining table": (140, 140, 0),  # Dark yellow
            "bed": (160, 100, 0),  # Dark brown
            # Indoor objects
            "potted plant": (0, 200, 0),  # Bright green
            "vase": (200, 200, 0),  # Yellow
            "lamp": (255, 255, 0),  # Bright yellow
            "backpack": (128, 0, 128),  # Purple
            # Default color for unspecified objects
            "default": (128, 128, 128),  # Gray
        }

    def filter_predictions(self, predictions, query, task_type):
        """Filter predictions based on NLP processed query"""
        # Use NLP processor to parse query
        parsed_task, target_objects = self.nlp_processor.parse_query(query)
        logger.info(f"Parsed task: {parsed_task}, Target objects: {target_objects}")

        # For general detection queries, return all predictions
        if query.lower() in [
            "detect objects",
            "detect all objects",
            "find objects",
            "detect object",
        ]:
            logger.info("General detection query, showing all detections")
            return predictions

        # Get available classes from predictions
        available_classes = {pred["class"].lower() for pred in predictions}
        logger.info(f"Available classes: {available_classes}")

        # Handle specific object queries
        query_lower = query.lower()
        specific_objects = set()

        # Map common query terms to object classes
        query_mappings = {
            "traffic light": ["traffic light", "traffic lights", "stoplight"],
            "person": ["person", "people", "human", "humans"],
            "car": ["car", "cars", "vehicle", "vehicles"],
            "truck": ["truck", "trucks"],
            "bus": ["bus", "buses"],
            "motorcycle": ["motorcycle", "motorcycles", "bike", "bikes"],
            "bicycle": ["bicycle", "bicycles", "bike", "bikes"],
            # Add animal categories
            "dog": ["dog", "dogs", "puppy", "puppies", "animal", "animals"],
            "cat": ["cat", "cats", "kitten", "kittens", "animal", "animals"],
            "bird": ["bird", "birds", "animal", "animals"],
            "horse": ["horse", "horses", "animal", "animals"],
            "sheep": ["sheep", "lamb", "lambs", "animal", "animals"],
            "cow": ["cow", "cows", "cattle", "animal", "animals"],
            # Furniture
            "chair": ["chair", "chairs", "seat", "seats", "furniture"],
            "couch": ["couch", "couches", "sofa", "sofas", "furniture"],
            "dining table": ["dining table", "table", "tables", "furniture"],
            "bed": ["bed", "beds", "furniture"],
            "desk": ["desk", "desks", "furniture"],
            # Indoor objects
            "plant": ["plant", "plants", "potted plant", "potted plants"],
            "vase": ["vase", "vases", "pot", "pots"],
            "lamp": ["lamp", "lamps", "light", "lights"],
            "backpack": ["backpack", "backpacks", "bag", "bags"],
            "book": ["book", "books"],
            "clock": ["clock", "clocks"],
            # Electronics
            "tv": ["tv", "television", "monitor", "screen"],
            "laptop": ["laptop", "laptops", "computer", "computers"],
            # Other indoor items
            "bottle": ["bottle", "bottles"],
            "cup": ["cup", "cups", "mug", "mugs"],
            "bowl": ["bowl", "bowls"],
        }

        # Check if query is about animals in general
        animal_terms = ["animal", "animals", "pet", "pets"]
        is_animal_query = any(term in query_lower for term in animal_terms)

        # Check if query is about furniture in general
        furniture_terms = ["furniture", "furnishing", "furnishings"]
        is_furniture_query = any(term in query_lower for term in furniture_terms)

        # Process query for specific objects or categories
        for class_name, terms in query_mappings.items():
            # If it's an animal query, include all animal classes
            if is_animal_query and any(animal_term in terms for animal_term in animal_terms):
                specific_objects.add(class_name)
            # Otherwise check for specific terms
            elif any(term in query_lower for term in terms):
                specific_objects.add(class_name)
            # If it's a furniture query, include all furniture classes

            if is_furniture_query and "furniture" in terms:
                specific_objects.add(class_name)
            # Check for specific terms in query
            elif any(term in query_lower for term in terms):
                specific_objects.add(class_name)
                logger.info(f"Added object class: {class_name} based on query terms")

        # Check for categorical queries
        if "plant" in query_lower or "plants" in query_lower:
            specific_objects.add("potted plant")

        if specific_objects:
            filtered_predictions = []
            for pred in predictions:
                class_name = pred["class"].lower()
                if class_name in specific_objects:
                    filtered_predictions.append(pred)
                    logger.info(f"Matched specific object: {class_name}")
            return filtered_predictions

        # If no specific objects found but query suggests general detection
        if any(term in query_lower for term in ["detect", "find", "show", "identify"]):
            logger.info("General detection terms found, showing all objects")
            return predictions

        # If nothing matches, return empty list
        logger.info("No matching objects found for query")
        return []

    def draw_predictions(self, image, predictions, task_type, caption=None):
        """Draw predictions and caption on image with better spacing and colors"""
        img_array = np.array(image)

        # Get image dimensions
        height, width = img_array.shape[:2]
        caption_height = 0

        # Add caption first if provided
        if caption:
            # Calculate caption space
            font_scale = 0.7
            thickness = 2
            margin = 10

            # Split caption into multiple lines
            words = caption.split()
            lines = []
            current_line = words[0]
            max_width = width - 2 * margin

            for word in words[1:]:
                test_line = current_line + " " + word
                (test_width, text_height), _ = cv2.getTextSize(
                    test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                if test_width <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            lines.append(current_line)

            # Calculate total caption height
            line_height = int(text_height * 1.5)
            caption_height = line_height * len(lines) + 2 * margin

            # Create a new image with extra space for caption
            new_height = height + caption_height
            new_img = np.zeros((new_height, width, 3), dtype=np.uint8)

            # Add black background for caption
            new_img[:caption_height] = (0, 0, 0)

            # Copy original image below caption
            new_img[caption_height:] = img_array

            # Draw caption text
            for i, line in enumerate(lines):
                y_position = margin + (i + 1) * line_height
                cv2.putText(
                    new_img,
                    line,
                    (margin, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )

            img_array = new_img

        if task_type == VisionTaskType.OBJECT_DETECTION:
            for pred in predictions:
                bbox = pred.get("bbox", None)
                if bbox:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    # Adjust y-coordinates to account for caption space
                    y1 += caption_height
                    y2 += caption_height

                    class_name = pred["class"].lower()
                    confidence = pred["confidence"]

                    # Get color for class
                    color = self.colors.get(class_name, self.colors["default"])

                    # Convert BGR to RGB for display
                    color = color[::-1]  # Reverse tuple for RGB

                    # Draw rectangle
                    cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)

                    # Add label with confidence
                    label = f"{class_name}: {confidence:.2f}"
                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                    # Ensure label background doesn't go outside image
                    y1_label = max(y1 - label_size[1] - 10, caption_height)

                    # Draw label background
                    cv2.rectangle(
                        img_array,
                        (x1, y1_label - baseline),
                        (x1 + label_size[0], y1_label + label_size[1]),
                        color,
                        -1,
                    )

                    # Draw label text
                    cv2.putText(
                        img_array,
                        label,
                        (x1, y1_label + label_size[1] - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

        return img_array

    def process_image(self, image, task_type, query):
        """Process image with specified task type and query"""
        try:
            temp_path = "temp_image.jpg"
            image.save(temp_path)

            logger.info(f"Processing image with task type: {task_type}, query: {query}")

            if task_type == "IMAGE_CAPTIONING":
                result = self.orchestrator.process_image(
                    image_path=temp_path,
                    user_comment=query,
                    task_type=VisionTaskType.IMAGE_CAPTIONING,
                )
                caption = result.results.get("caption", "No caption generated")
                visualized_image = self.draw_predictions(
                    image, [], VisionTaskType.IMAGE_CAPTIONING, caption=caption
                )
                return (
                    visualized_image,
                    f"Generated Caption:\n{caption}\n\nProcessing time: {result.processing_time:.3f}s",
                )

            elif task_type in ["OBJECT_DETECTION", "IMAGE_CLASSIFICATION"]:
                if task_type == "OBJECT_DETECTION":
                    detection_result, caption_result = self.orchestrator.process_image_with_caption(
                        image_path=temp_path,
                        user_comment=query,
                    )
                    result = detection_result
                else:
                    result = self.orchestrator.process_image(
                        image_path=temp_path,
                        user_comment=query,
                        task_type=VisionTaskType[task_type],
                    )
                    caption_result = self.orchestrator.process_image(
                        image_path=temp_path,
                        user_comment="Describe this image",
                        task_type=VisionTaskType.IMAGE_CAPTIONING,
                    )

                if result.task_type == VisionTaskType.IMAGE_CLASSIFICATION:
                    filtered_preds = self.filter_predictions(
                        result.results["top_predictions"], query, result.task_type
                    )
                    output = "Classification Results:\n\n"
                    if filtered_preds:
                        for pred in filtered_preds:
                            output += f"• {pred['class']}: {pred['confidence']:.3f}\n"
                    else:
                        output = f"No matching classification results found.\n"

                    caption = caption_result.results.get("caption") if caption_result else None
                    visualized_image = self.draw_predictions(
                        image, filtered_preds, result.task_type, caption=caption
                    )

                elif result.task_type == VisionTaskType.OBJECT_DETECTION:
                    detections = result.results.get("detections", [])
                    logger.info(f"Raw detections: {detections}")

                    filtered_dets = self.filter_predictions(detections, query, result.task_type)
                    logger.info(f"Filtered detections: {filtered_dets}")

                    output = "Detection Results:\n\n"
                    if filtered_dets:
                        for det in filtered_dets:
                            output += f"• {det['class']}: {det['confidence']:.3f}\n"

                        caption = caption_result.results.get("caption") if caption_result else None
                        visualized_image = self.draw_predictions(
                            image, filtered_dets, result.task_type, caption=caption
                        )
                    else:
                        output = f"No objects matching '{query}' were found.\n"
                        visualized_image = image

                    if caption_result and caption_result.results.get("caption"):
                        output += f"\nImage Caption:\n{caption_result.results['caption']}"

                output += f"\nProcessing time: {result.processing_time:.3f}s"
                return visualized_image, output

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(traceback.format_exc())
            return image, f"Error processing image: {str(e)}\n\nPlease check logs for details."
        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def create_interface(self):
        """Create Gradio interface"""
        demo = gr.Blocks(title="Vision Framework Demo")

        with demo:
            gr.Markdown(
                """
            # 🤖 Vision Framework Demo

            This demo showcases the multi-agent vision framework's capabilities. Upload an image and select a task type to analyze it.

            ### Supported Tasks:
            - **Image Classification**: Identifies the main subjects in the image
            - **Object Detection**: Locates and identifies multiple objects in the scene
            - **Image Captioning**: Generates natural language description of the image
            """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Upload Image")
                    task_type = gr.Dropdown(
                        choices=["IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "IMAGE_CAPTIONING"],
                        value="OBJECT_DETECTION",
                        label="Task Type",
                    )
                    query = gr.Textbox(
                        label="Query (Optional)",
                        placeholder="E.g., 'detect traffic lights', 'find vehicles', 'describe this image'",
                        value="detect objects",
                    )
                    submit_btn = gr.Button("Analyze", variant="primary")

                with gr.Column(scale=1):
                    output_image = gr.Image(label="Visualization")
                    output_text = gr.Textbox(label="Results", lines=10)

            # Example tabs
            with gr.Tabs():
                with gr.TabItem("Detection Examples"):
                    gr.Examples(
                        examples=[
                            [
                                "examples/data/street.jpg",
                                "OBJECT_DETECTION",
                                "detect traffic lights",
                            ],
                            ["examples/data/street.jpg", "OBJECT_DETECTION", "find vehicles"],
                            ["examples/data/street.jpg", "OBJECT_DETECTION", "detect people"],
                        ],
                        inputs=[input_image, task_type, query],
                    )

                with gr.TabItem("Classification Examples"):
                    gr.Examples(
                        examples=[
                            [
                                "examples/data/dog.jpg",
                                "IMAGE_CLASSIFICATION",
                                "What breed is this dog?",
                            ],
                            [
                                "examples/data/cat.jpg",
                                "IMAGE_CLASSIFICATION",
                                "Classify this image",
                            ],
                        ],
                        inputs=[input_image, task_type, query],
                    )

                with gr.TabItem("Captioning Examples"):
                    gr.Examples(
                        examples=[
                            ["examples/data/street.jpg", "IMAGE_CAPTIONING", "Describe this scene"],
                            [
                                "examples/data/living_room.jpg",
                                "IMAGE_CAPTIONING",
                                "What's in this image?",
                            ],
                        ],
                        inputs=[input_image, task_type, query],
                    )

            # Set up event handlers
            submit_btn.click(
                fn=self.process_image,
                inputs=[input_image, task_type, query],
                outputs=[output_image, output_text],
            )

        return demo


def main():
    """Main function to run the Gradio demo"""
    try:
        demo = VisionDemo()
        interface = demo.create_interface()
        interface.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
        )
    except Exception as e:
        logger.error(f"Error launching demo: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
