import logging
import os
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from vision_framework import VisionInput, VisionOrchestrator, VisionTaskType


def download_test_video():
    """Download a test video if not exists."""
    import urllib.request

    video_dir = Path("tests/data/videos")
    video_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / "pedestrians.mp4"
    if not video_path.exists():
        logger.info(f"Downloading test video to {video_path}")
        # Replace with your video URL
        url = "https://example.com/sample_video.mp4"
        urllib.request.urlretrieve(url, video_path)
    return video_path


def print_video_info(video_path: str):
    """Print video information"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info("Video Information:")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Frame count: {frame_count}")
    logger.info(f"  Duration: {frame_count/fps:.2f} seconds")
    cap.release()


def process_video_with_caption(
    orchestrator: VisionOrchestrator,
    video_path: str,
    output_path: str,
    target_classes: list = None,
    process_duration: float = None,
    caption_interval: int = 30,
):
    """Process video with detection and frame-by-frame captioning."""
    try:
        import PIL.Image as Image

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info("Processing video with detection and tracking...")
        query = f"detect {' and '.join(target_classes)}" if target_classes else "detect all objects"

        # First pass: Get detections
        video_result, _ = orchestrator.process_video(
            video_path=video_path,
            output_path=None,
            user_comment=query,
            start_time=0,
            end_time=process_duration,
        )

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        current_caption = ""

        logger.info("Processing frames with captions...")

        # Get captioning agent
        captioning_agent = orchestrator.router.agents.get(VisionTaskType.IMAGE_CAPTIONING)

        for frame_result in tqdm(video_result.frames_results, desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break

            # Generate new caption periodically
            if frame_idx % caption_interval == 0:
                try:
                    # Convert frame to PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    # Get detections for current frame
                    detections = frame_result.results["detections"]
                    if target_classes:
                        detections = [
                            det
                            for det in detections
                            if det["class"].lower() in [cls.lower() for cls in target_classes]
                        ]

                    # Create prompt from detections
                    objects = [f"{d['class']}" for d in detections]
                    if objects:
                        prompt = f"A scene containing {', '.join(objects)}"
                    else:
                        prompt = "Describe this scene"

                    # Generate caption
                    vision_input = VisionInput(
                        image=pil_image,
                        user_comment=prompt,
                        task_type=VisionTaskType.IMAGE_CAPTIONING,
                    )
                    caption_output = captioning_agent.process(vision_input)
                    current_caption = caption_output.results.get("caption", "")
                    logger.debug(f"Frame {frame_idx} caption: {current_caption}")
                except Exception as e:
                    logger.warning(f"Failed to generate caption for frame {frame_idx}: {str(e)}")

            # Draw detections
            detections = frame_result.results["detections"]
            if target_classes:
                detections = [
                    det
                    for det in detections
                    if det["class"].lower() in [cls.lower() for cls in target_classes]
                ]

            # Draw bounding boxes
            for det in detections:
                bbox = det["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                label = f"{det['class']} {det['confidence']:.2f}"
                if "track_id" in det:
                    label += f" ID:{det['track_id']}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    (0, 255, 0),
                    -1,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            # Add caption to frame
            if current_caption:
                # Calculate position for caption
                margin = 10
                font_scale = 0.7
                thickness = 2

                # Split caption into multiple lines
                words = current_caption.split()
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

                # Draw caption background
                line_height = int(text_height * 1.5)
                total_height = line_height * len(lines) + 2 * margin
                cv2.rectangle(frame, (0, 0), (width, total_height), (0, 0, 0), -1)

                # Draw caption text
                for i, line in enumerate(lines):
                    y_position = margin + (i + 1) * line_height
                    cv2.putText(
                        frame,
                        line,
                        (margin, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                    )

            # Write the frame
            out.write(frame)
            frame_idx += 1

        # Release resources
        cap.release()
        out.release()

        # Print statistics
        logger.info(f"\nProcessed {frame_idx} frames")
        logger.info(f"Output saved to: {output_path}")

        return video_result, None

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise


def main():
    # Configuration
    config = {
        "ENABLE_CAPTIONING": True,
        "YOLO_MODEL_NAME": "yolov8s.pt",
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.45,
        "DETECTION_IMAGE_SIZE": 640,
        "ENABLE_TRACK": True,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "MAX_CAPTION_LENGTH": 50,
        "MIN_CAPTION_LENGTH": 10,
        "VIDEO_FPS": 30,
    }

    # Initialize orchestrator
    logger.info("Initializing VisionOrchestrator...")
    orchestrator = VisionOrchestrator(config)

    # Process video examples
    video_path = "tests/data/videos/crosswalk.avi"  # Replace with your video path
    logger.info(f"\nProcessing video: {video_path}")
    print_video_info(video_path)

    # Example 1: Track and caption people
    logger.info("\nExample 1: Track and caption people")
    logger.info("-" * 50)

    try:
        output_path = "examples/output/people_detection.mp4"
        result, captions = process_video_with_caption(
            orchestrator=orchestrator,
            video_path=video_path,
            output_path=output_path,
            target_classes=["person"],
            process_duration=10.0,  # Process first 10 seconds
            caption_interval=15,  # Caption every 15 frames
        )

    except Exception as e:
        logger.error(f"Error in people tracking: {str(e)}")

    # Example 2: Track multiple object types
    logger.info("\nExample 2: Track multiple objects")
    logger.info("-" * 50)

    try:
        output_path = "examples/output/multi_object_detection.mp4"
        result, captions = process_video_with_caption(
            orchestrator=orchestrator,
            video_path=video_path,
            output_path=output_path,
            target_classes=["person", "car", "bicycle"],
            process_duration=10.0,
            caption_interval=15,
        )

    except Exception as e:
        logger.error(f"Error in multi-object tracking: {str(e)}")


if __name__ == "__main__":
    main()
