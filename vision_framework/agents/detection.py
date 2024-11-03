import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ..core.types import (
    BatchDetectionResult,
    VideoDetectionResult,
    VisionInput,
    VisionOutput,
    VisionTaskType,
)
from ..utils.image import draw_detections
from ..utils.video import get_video_properties
from .base import BaseVisionAgent

logger = logging.getLogger(__name__)


class YOLODetectionAgent(BaseVisionAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        self.conf_threshold = config.get("YOLO_CONFIDENCE_THRESHOLD", 0.25)
        self.iou_threshold = config.get("YOLO_IOU_THRESHOLD", 0.45)
        self.image_size = config.get("DETECTION_IMAGE_SIZE", 640)
        self.enable_tracking = config.get("ENABLE_TRACK", True)

        # Initialize model
        self.model = self.load_model()

        # Initialize tracking if enabled
        if self.enable_tracking:
            try:
                self.model.tracker = "bytetrack.yaml"
                logger.info("Tracking enabled with ByteTrack")
            except Exception as e:
                logger.warning(f"Failed to enable tracking: {str(e)}")
                self.enable_tracking = False

    def get_available_classes(self) -> Set[str]:
        """Get all available classes in the model"""
        return {name.lower() for name in self.model.names.values()}

    def validate_classes(self, requested_classes: List[str]) -> List[str]:
        """Validate and normalize class names"""
        available = self.get_available_classes()
        normalized = []
        invalid_classes = []

        for cls in requested_classes:
            cls_lower = cls.lower()
            if cls_lower in available:
                normalized.append(cls_lower)
            else:
                invalid_classes.append(cls)

        if invalid_classes:
            logger.warning(f"Classes not available in model: {invalid_classes}")
            logger.info(f"Available classes: {sorted(list(available))}")

        return normalized

    def get_class_indices(self, class_names: List[str]) -> List[int]:
        """Convert class names to model indices with validation"""
        indices = []
        for name in class_names:
            name_lower = name.lower()
            if name_lower in self.class_mapping:
                indices.append(self.class_mapping[name_lower])
            else:
                logger.warning(f"Class '{name}' not found in model classes")
        return indices

    def load_model(self):
        """Load YOLOv8 model"""
        logger.info(f"Loading YOLOv8 model: {self.config['YOLO_MODEL_NAME']}")
        model = YOLO(self.config["YOLO_MODEL_NAME"])
        return model

    def process_image(
        self,
        image: np.ndarray,
        detect_classes: Optional[List[str]] = None,
        user_comment: Optional[str] = None,
    ) -> Dict:
        """Process a single image for detection."""
        vision_input = VisionInput(
            image=image,
            additional_params={"detect_classes": detect_classes},
            user_comment=user_comment,
        )
        return self.process(vision_input)

    def process_detections(
        self, results, allowed_classes: Optional[List[str]] = None
    ) -> List[Dict]:
        """Process YOLO results into standardized detection format with strict filtering"""
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_name = result.names[int(box.cls)]

                # Apply strict class filtering
                if allowed_classes and class_name.lower() not in [
                    cls.lower() for cls in allowed_classes
                ]:
                    continue

                detection = {
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf),
                    "class": class_name,
                    "class_id": int(box.cls),
                }

                # Add tracking ID if available
                if hasattr(box, "id") and box.id is not None:
                    try:
                        detection["track_id"] = int(box.id.item())
                    except:
                        detection["track_id"] = str(box.id)

                detections.append(detection)

        return detections

    def filter_detections(
        self, detections: List[Dict], allowed_classes: Optional[List[str]] = None
    ) -> List[Dict]:
        """Filter detections based on allowed classes"""
        if not allowed_classes:
            return detections

        # Convert to lower case for case-insensitive comparison
        allowed_classes = [cls.lower() for cls in allowed_classes]

        return [det for det in detections if det["class"].lower() in allowed_classes]

    def process(self, vision_input: VisionInput) -> VisionOutput:
        """Process single image detection with strict filtering"""
        start_time = time.time()

        # Get specific classes to detect from additional parameters
        detect_classes = None
        if vision_input.additional_params and "detect_classes" in vision_input.additional_params:
            detect_classes = vision_input.additional_params["detect_classes"]
            logger.info(f"Filtering detections for classes: {detect_classes}")

        # Run inference
        results = self.model(
            vision_input.image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
        )

        # Process and filter detections
        detections = self.process_detections(results)
        if detect_classes:
            detections = self.filter_detections(detections, detect_classes)

        # Calculate confidence
        overall_confidence = (
            sum(d["confidence"] for d in detections[:5]) / min(5, len(detections))
            if detections
            else 0.0
        )

        return VisionOutput(
            task_type=VisionTaskType.OBJECT_DETECTION,
            results={
                "detections": detections,
                "num_detections": len(detections),
                "detect_classes": detect_classes,
            },
            confidence=overall_confidence,
            processing_time=time.time() - start_time,
        )

    def process_batch(
        self, image_paths: List[str], batch_size: int = 32
    ) -> List[BatchDetectionResult]:
        """Process a batch of images"""
        results = []

        # Process images in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i : i + batch_size]

            # Load batch images
            batch_images = [self.load_image(path) for path in batch_paths]

            # Run inference on batch
            batch_results = self.model(
                batch_images,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
            )

            # Process each result
            for img_path, result in zip(batch_paths, batch_results):
                detections = self.process_detections([result])

                # Calculate confidence
                overall_confidence = (
                    sum(d["confidence"] for d in detections[:5]) / min(5, len(detections))
                    if detections
                    else 0.0
                )

                # Create vision output
                vision_output = VisionOutput(
                    task_type=VisionTaskType.OBJECT_DETECTION,
                    results={
                        "detections": detections,
                        "num_detections": len(detections),
                    },
                    confidence=overall_confidence,
                    processing_time=0.0,
                )

                results.append(BatchDetectionResult(img_path, vision_output))

        return results

    def process_video(
        self,
        vision_input: VisionInput,
        output_path: Optional[str] = None,
        start_time: float = 0,
        end_time: Optional[float] = None,
    ) -> VideoDetectionResult:
        """Process video with strict class filtering"""
        video_path = str(vision_input.image)
        allowed_classes = None
        class_indices = None

        logger.info(f"Starting video processing: {video_path}")

        # Setup class filtering with validation
        if vision_input.additional_params and "detect_classes" in vision_input.additional_params:
            requested_classes = vision_input.additional_params["detect_classes"]
            allowed_classes = self.validate_classes(requested_classes)

            if not allowed_classes:
                logger.warning(f"No valid classes found among {requested_classes}")
                allowed_classes = None
            else:
                # Get model class indices for filtering
                class_mapping = {name.lower(): idx for idx, name in self.model.names.items()}
                class_indices = [class_mapping[cls] for cls in allowed_classes]
                logger.info(f"Filtering for classes: {allowed_classes}")

        # First, get video properties and count frames if needed
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Count frames if total_frames is 0 or invalid
        if total_frames <= 0:
            logger.warning("Invalid frame count, counting frames manually...")
            total_frames = 0
            while True:
                ret = cap.grab()
                if not ret:
                    break
                total_frames += 1
            logger.info(f"Found {total_frames} frames")

        # Close the first capture after counting
        cap.release()

        # Calculate frame range
        start_frame = int(start_time * fps) if start_time > 0 else 0
        end_frame = min(int(end_time * fps), total_frames) if end_time else total_frames

        if start_frame >= end_frame or start_frame >= total_frames:
            raise ValueError(
                f"Invalid frame range: {start_frame} to {end_frame} (total frames: {total_frames})"
            )

        logger.debug(f"Processing frames {start_frame} to {end_frame} of {total_frames}")

        # Initialize results storage
        frames_results = []
        total_time = 0
        frame_count = start_frame

        try:
            # Open a fresh video capture for processing
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file for processing: {video_path}")

            # Set starting frame
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Setup video writer if output path is provided
            out = None
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Process frames
            with tqdm(total=end_frame - start_frame, desc="Processing video") as pbar:
                while frame_count < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Failed to read frame {frame_count}")
                        break

                    start_time_proc = time.time()

                    try:
                        # Run prediction with class filtering
                        if allowed_classes:
                            results = self.model.predict(
                                source=frame,
                                classes=class_indices,
                                conf=self.conf_threshold,
                                iou=self.iou_threshold,
                                device=self.device,
                                verbose=False,
                                stream=True,
                            )
                        else:
                            results = self.model.predict(
                                source=frame,
                                conf=self.conf_threshold,
                                iou=self.iou_threshold,
                                device=self.device,
                                verbose=False,
                                stream=True,
                            )

                        # Process detections
                        detections = []
                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                class_name = r.names[int(box.cls)].lower()

                                # Skip if not in allowed classes
                                if allowed_classes and class_name not in allowed_classes:
                                    continue

                                detection = {
                                    "bbox": box.xyxy[0].tolist(),
                                    "confidence": float(box.conf),
                                    "class": class_name,
                                    "class_id": int(box.cls),
                                }

                                # Add tracking ID if available
                                if hasattr(box, "id") and box.id is not None:
                                    try:
                                        detection["track_id"] = int(box.id.item())
                                    except:
                                        detection["track_id"] = str(box.id)

                                detections.append(detection)

                        processing_time = time.time() - start_time_proc
                        total_time += processing_time

                        # Create vision output
                        confidence = sum(d["confidence"] for d in detections[:5]) / max(
                            len(detections[:5]), 1
                        )

                        vision_output = VisionOutput(
                            task_type=VisionTaskType.OBJECT_DETECTION,
                            results={
                                "detections": detections,
                                "num_detections": len(detections),
                                "detect_classes": allowed_classes,
                            },
                            confidence=confidence,
                            processing_time=processing_time,
                        )
                        frames_results.append(vision_output)

                        # Write annotated frame if requested
                        if out is not None:
                            annotated_frame = frame.copy()
                            if detections:
                                for det in detections:
                                    bbox = det["bbox"]
                                    x1, y1, x2, y2 = map(int, bbox)
                                    label = f"{det['class']} {det['confidence']:.2f}"
                                    if "track_id" in det:
                                        label += f" ID:{det['track_id']}"

                                    cv2.rectangle(
                                        annotated_frame,
                                        (x1, y1),
                                        (x2, y2),
                                        (0, 255, 0),
                                        2,
                                    )
                                    cv2.putText(
                                        annotated_frame,
                                        label,
                                        (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 255, 0),
                                        2,
                                    )
                            out.write(annotated_frame)

                        # Log progress periodically
                        if frame_count % 100 == 0:
                            logger.debug(f"Processed frame {frame_count}/{end_frame}")

                    except Exception as e:
                        logger.error(f"Error processing frame {frame_count}: {str(e)}")

                    frame_count += 1
                    pbar.update(1)

        except Exception as e:
            logger.error(f"Error during video processing: {str(e)}")
            raise

        finally:
            logger.debug("Cleaning up resources...")
            if "cap" in locals() and cap is not None:
                cap.release()
            if "out" in locals() and out is not None:
                out.release()

        # Verify processing results
        processed_frames = len(frames_results)
        if processed_frames == 0:
            raise RuntimeError("No frames were processed successfully")

        logger.info(f"Successfully processed {processed_frames} frames")

        # Calculate statistics
        all_detected_classes = set()
        total_detections = 0
        for frame in frames_results:
            frame_classes = {det["class"] for det in frame.results["detections"]}
            all_detected_classes.update(frame_classes)
            total_detections += frame.results["num_detections"]

        logger.info(f"Total detections: {total_detections}")
        logger.info(f"Average detections per frame: {total_detections/processed_frames:.2f}")
        logger.info(f"Detected classes: {all_detected_classes}")

        if allowed_classes:
            logger.info(f"Filtering was active for classes: {allowed_classes}")

        return VideoDetectionResult(
            video_path=video_path,
            frames_results=frames_results,
            fps=fps,
            total_time=total_time,
        )

    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        if isinstance(image, str):
            annotated_image = cv2.imread(image)
            # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        else:
            """Visualize detections on the image."""
            annotated_image = image.copy()  # Work with a copy of the image

        for det in detections:
            bbox = det["bbox"]
            label = f"{det['class']} {det['confidence']:.2f}"
            if len(bbox) != 4:
                logger.error(f"Invalid bounding box: {bbox}")
                continue
            x1, y1, x2, y2 = map(int, bbox)  # Ensure these are integers
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        if output_path:
            cv2.imwrite(output_path, annotated_image)

        return annotated_image

    # def visualize_detections(self, image_path: str, detections: List[Dict],
    #                        output_path: Optional[str] = None) -> np.ndarray:
    #     """Visualize detections on the image"""
    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     annotated_image = draw_detections(image, detections)

    #     if output_path:
    #         os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #         cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    #     return annotated_image
