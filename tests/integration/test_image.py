import cv2
import numpy as np

from vision_framework.utils.image import draw_detections


def test_draw_detections():
    # Create a blank image (black background) for testing
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Define a list of detection dictionaries
    detections = [
        {
            "bbox": [10, 10, 50, 50],
            "class": "cat",
            "confidence": 0.85,
        },  # x1, y1, x2, y2
        {"bbox": [60, 60, 90, 90], "class": "dog", "confidence": 0.95, "track_id": 42},
    ]

    # Draw detections on the image
    output_image = draw_detections(image, detections)

    # Verify the output is still the correct shape
    assert output_image.shape == image.shape, "Output image shape mismatch"

    # Check that some drawing was actually done (output is not identical to the input)
    assert not np.array_equal(output_image, image), "No drawings were made on the image"

    # Optionally save or display the image for manual inspection
    # cv2.imwrite("output_test.jpg", output_image)  # Save to verify visually, if needed

    # Check that bounding box areas are not all black, indicating that drawing occurred
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        drawn_area = output_image[y1:y2, x1:x2]

        # Ensure the region within the bounding box is not all black
        assert drawn_area.sum() > 0, f"Bounding box for {det['class']} not drawn properly"

    print("All assertions passed for draw_detections test.")
