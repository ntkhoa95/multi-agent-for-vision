import pytest
from vision_framework import VisionTaskType

def test_image_classification(orchestrator, sample_image="test_assets/images/street.jpg"):
    result = orchestrator.process_image(
        image_path=sample_image,
        user_comment="classify this image"
    )
    
    assert result.task_type == VisionTaskType.IMAGE_CLASSIFICATION
    assert 'top_predictions' in result.results
    assert len(result.results['top_predictions']) == 5
    assert result.confidence > 0.0
    assert result.processing_time > 0.0

def test_classification_results_format(orchestrator, sample_image="test_assets/images/street.jpg"):
    result = orchestrator.process_image(
        image_path=sample_image,
        user_comment="what is in this image"
    )

    assert result.task_type == VisionTaskType.IMAGE_CLASSIFICATION
    assert 'top_predictions' in result.results
    assert len(result.results['top_predictions']) == 5
    assert result.confidence > 0.0
    assert result.processing_time > 0.0

    # Print the result to understand its structure
    print("Result:", result.results)
    
    # Check if 'top_predictions' key exists in the result
    if 'top_predictions' in result.results:
        predictions = result.results['top_predictions']
        for pred in predictions:
            assert 'class' in pred
            assert 'confidence' in pred
            assert isinstance(pred['confidence'], float)
            assert 0 <= pred['confidence'] <= 1
    else:
        print("Key 'top_predictions' not found in the result.")
        # Handle the case where 'top_predictions' is not present
        # For example, you can raise an error or fail the test
        assert False, "Key 'top_predictions' not found in the result."
