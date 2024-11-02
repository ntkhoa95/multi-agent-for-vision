import pytest
from vision_framework import VisionTaskType, VisionOutput

def test_object_detection(orchestrator, sample_image="test_assets/images/street.jpg"):
    result = orchestrator.process_image(
        image_path=sample_image,
        user_comment="detect objects"
    )
    
    assert result.task_type == VisionTaskType.OBJECT_DETECTION
    assert 'detections' in result.results
    assert isinstance(result.results['detections'], list)
    assert result.confidence >= 0.0
    assert result.processing_time > 0.0

def test_specific_object_detection(orchestrator, sample_image):
    result = orchestrator.process_image(
        image_path=sample_image,
        user_comment="find people",
        task_type=VisionTaskType.OBJECT_DETECTION,
        additional_params={'detect_classes': ['person']}  # Explicitly set allowed classes
    )
    
    assert result.task_type == VisionTaskType.OBJECT_DETECTION
    assert 'detect_classes' in result.results
    assert result.results['detect_classes'] == ['person']
    
    # Verify all detections are persons
    for det in result.results['detections']:
        assert det['class'].lower() == 'person'

def test_batch_detection(orchestrator, sample_image="test_assets/images/street.jpg"):
    # Use same image twice for batch
    image_paths = [sample_image, sample_image]
    
    results = orchestrator.process_batch(
        image_paths=image_paths,
        task_type=VisionTaskType.OBJECT_DETECTION,
    )
    
    assert len(results) == 2
    for result in results:
        assert isinstance(result.vision_output, VisionOutput)
        assert result.vision_output.task_type == VisionTaskType.OBJECT_DETECTION

def test_video_detection(orchestrator, sample_video="test_videos/crosswalk.avi"):
    result = orchestrator.process_video(
        video_path=sample_video,
        user_comment="detect objects",
        output_path=None
    )
    
    assert result.num_frames > 0
    assert result.fps > 0
    assert result.total_time > 0
    assert len(result.frames_results) > 0