import unittest
import os
import cv2
import numpy as np
from vision_framework.utils.video import get_video_properties

class TestVideoProperties(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create a temporary video file for testing
        self.test_video_path = 'test_video.mp4'
        self.create_test_video(self.test_video_path)

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_video_path):
            os.remove(self.test_video_path)

    def create_test_video(self, video_path: str):
        """Helper function to create a simple test video."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

        # Create a blank frame (black)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(10):  # 10 frames
            out.write(frame)

        out.release()

    def test_get_video_properties(self):
        """Test retrieving properties of a valid video."""
        self.assertTrue(os.path.exists(self.test_video_path), "Test video file does not exist.")
        fps, total_frames, width, height = get_video_properties(self.test_video_path)

        self.assertEqual(fps, 20.0, "FPS should be 20.0")
        self.assertEqual(total_frames, 10, "Total frames should be 10")
        self.assertEqual(width, 640, "Width should be 640 pixels")
        self.assertEqual(height, 480, "Height should be 480 pixels")

    def test_invalid_video_path(self):
        """Test handling of an invalid video path."""
        with self.assertRaises(ValueError) as context:
            get_video_properties("invalid_path.mp4")
        
        self.assertEqual(str(context.exception), "Error opening video file: invalid_path.mp4")

    def test_empty_video_file(self):
        """Test handling of an empty video file."""
        empty_video_path = 'empty_video.mp4'
        open(empty_video_path, 'a').close()  # Create an empty file

        with self.assertRaises(ValueError):
            get_video_properties(empty_video_path)

        os.remove(empty_video_path)

if __name__ == '__main__':
    unittest.main()
