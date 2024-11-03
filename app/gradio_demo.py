# examples/gradio_demo.py
import logging
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

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
        }
        self.orchestrator = VisionOrchestrator(self.config)
        self.nlp_processor = NLPProcessor()
        logger.info(f"Using device: {self.config['DEVICE']}")

        # Define colors for visualization
        self.colors = {
            "human": (255, 0, 0),  # Red for persons
            "vehicle": (0, 255, 0),  # Green for vehicles
            "animal": (0, 0, 255),  # Blue for animals
            "default": (255, 255, 0),  # Yellow for other objects
        }

    def get_category_for_class(self, class_name):
        """Get the category for a given class name"""
        class_name = class_name.lower()
        for category, terms in self.category_mappings.items():
            if any(term in class_name for term in terms):
                return category
        return "default"

    def filter_predictions(self, predictions, query, task_type):
        """Filter predictions based on NLP processed query"""
        # Use NLP processor to parse query
        parsed_task, target_objects = self.nlp_processor.parse_query(query)
        logger.info(f"Parsed task: {parsed_task}, Target objects: {target_objects}")

        # For general queries without specific targets, return all predictions
        if not target_objects:
            if parsed_task == "classification" or query.lower() in [
                "what is in this image",
                "classify this image",
            ]:
                return predictions

        # Get available classes from predictions
        available_classes = {pred["class"].lower() for pred in predictions}

        # Validate and normalize target objects
        valid_objects = self.nlp_processor.validate_objects(target_objects, available_classes)
        logger.info(f"Valid target objects: {valid_objects}")

        if not valid_objects:
            if parsed_task == "classification":
                return predictions
            return []

        # Filter predictions based on validated objects
        filtered_predictions = []
        for pred in predictions:
            class_name = pred["class"].lower()
            normalized_class = self.nlp_processor.normalize_object_name(class_name)

            if normalized_class in valid_objects:
                filtered_predictions.append(pred)
                logger.info(f"Matched prediction: {class_name}")

        return filtered_predictions

    def draw_predictions(self, image, predictions, task_type):
        """Draw predictions on image"""
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        if task_type == VisionTaskType.OBJECT_DETECTION:
            for pred in predictions:
                bbox = pred.get("bbox", None)
                if bbox:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    class_name = pred["class"].lower()
                    confidence = pred["confidence"]

                    # Get color based on normalized class name
                    normalized_class = self.nlp_processor.normalize_object_name(class_name)
                    category = next(
                        (
                            cat
                            for cat, terms in self.nlp_processor.object_synonyms.items()
                            if normalized_class in terms
                        ),
                        "default",
                    )
                    color = self.colors.get(category, self.colors["default"])

                    # Draw rectangle
                    cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)

                    # Add label with confidence
                    label = f"{class_name}: {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(
                        img_array,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        img_array,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

        elif task_type == VisionTaskType.IMAGE_CLASSIFICATION:
            for i, pred in enumerate(predictions[:3]):
                class_name = pred["class"]
                confidence = pred["confidence"]
                label = f"{class_name}: {confidence:.2f}"
                y_pos = 30 + (i * 30)
                cv2.putText(
                    img_array,
                    label,
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    3,
                )
                cv2.putText(
                    img_array,
                    label,
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    1,
                )

        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return img_array

    def process_image(self, image, task_type, query):
        """Process image with specified task type and query"""
        try:
            temp_path = "temp_image.jpg"
            image.save(temp_path)

            logger.info(f"Processing image with task type: {task_type}, query: {query}")

            # Process image
            result = self.orchestrator.process_image(
                image_path=temp_path,
                user_comment=query,
                task_type=VisionTaskType[task_type],
            )

            if result.task_type == VisionTaskType.IMAGE_CLASSIFICATION:
                # Filter and process classification results
                filtered_preds = self.filter_predictions(
                    result.results["top_predictions"], query, result.task_type
                )

                output = "Classification Results:\n\n"
                if filtered_preds:
                    for pred in filtered_preds:
                        output += f"• {pred['class']}: {pred['confidence']:.3f}\n"
                else:
                    output = f"No matching classification results found.\n"
                visualized_image = self.draw_predictions(image, filtered_preds, result.task_type)

            elif result.task_type == VisionTaskType.OBJECT_DETECTION:
                # Filter and process detection results
                detections = result.results.get("detections", [])
                logger.info(f"Raw detections: {detections}")

                filtered_dets = self.filter_predictions(detections, query, result.task_type)
                logger.info(f"Filtered detections: {filtered_dets}")

                output = "Detection Results:\n\n"
                if filtered_dets:
                    for det in filtered_dets:
                        output += f"• {det['class']}: {det['confidence']:.3f}\n"
                    visualized_image = self.draw_predictions(image, filtered_dets, result.task_type)
                else:
                    output = f"No objects matching '{query}' were found.\n"
                    visualized_image = image

            output += f"\nProcessing time: {result.processing_time:.3f}s"
            return visualized_image, output

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(traceback.format_exc())
            return (
                image,
                f"Error processing image: {str(e)}\n\nPlease check logs for details.",
            )
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
            """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    # Input components
                    input_image = gr.Image(type="pil", label="Upload Image")
                    task_type = gr.Dropdown(
                        choices=["IMAGE_CLASSIFICATION", "OBJECT_DETECTION"],
                        value="OBJECT_DETECTION",
                        label="Task Type",
                    )
                    query = gr.Textbox(
                        label="Query (Optional)",
                        placeholder="E.g., 'detect human in this image', 'find vehicles'",
                        value="detect objects",
                    )
                    submit_btn = gr.Button("Analyze", variant="primary")

                with gr.Column(scale=1):
                    # Output components
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
                                "detect humans",
                            ],
                            [
                                "examples/data/street.jpg",
                                "OBJECT_DETECTION",
                                "find vehicles",
                            ],
                            [
                                "examples/data/living_room.jpg",
                                "OBJECT_DETECTION",
                                "detect all objects",
                            ],
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

            # Set up event handlers
            submit_btn.click(
                fn=self.process_image,
                inputs=[input_image, task_type, query],
                outputs=[output_image, output_text],
            )

            # Add usage instructions
            gr.Markdown(
                """
            ### 📝 Instructions

            1. Upload an image or use one of the examples
            2. Select the task type:
               - Use **Object Detection** to find and locate objects
               - Use **Classification** to identify the main subject
            3. Enter a query (optional):
               - For detection: "detect humans", "find vehicles", etc.
               - For classification: "what is this?", "classify this image"
            4. Click "Analyze" to process the image

            ### 💡 Tips

            - For detection queries, you can specify object types (humans, vehicles, animals)
            - Use clear images with good lighting for best results
            - Try different queries to filter specific objects of interest
            """
            )

        return demo


def main():
    demo = VisionDemo()
    interface = demo.create_interface()
    interface.launch(share=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
