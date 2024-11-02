import pytest
from vision_framework.nlp.processor import NLPProcessor

@pytest.fixture
def nlp_processor():
    return NLPProcessor()

def test_query_preprocessing(nlp_processor):
    query = "Find Dogs and Cats in the Image!"
    processed = nlp_processor.preprocess_query(query)
    assert processed == "find dogs and cats in the image"

def test_task_type_extraction(nlp_processor):
    queries = {
        "detect cats": "detection",
        "classify this image": "classification",
        "find people": "detection",
        "what is in this image": "classification",
        "identify objects": "classification"
    }
    
    for query, expected_task in queries.items():
        task_type, _ = nlp_processor.parse_query(query)
        assert task_type == expected_task, f"Failed for query: {query}"

def test_object_extraction(nlp_processor):
    queries = {
        "detect cats and dogs": {"cat", "dog"},
        "find people walking": {"person"},
        "locate cars and trucks": {"car", "truck"}
    }
    
    for query, expected_objects in queries.items():
        _, objects = nlp_processor.parse_query(query)
        normalized_objects = {nlp_processor.normalize_object_name(obj) for obj in objects}
        assert normalized_objects == expected_objects, f"Failed for query: {query}"