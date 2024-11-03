from unittest.mock import MagicMock

import pytest

from vision_framework.core.types import VisionInput, VisionOutput, VisionTaskType
from vision_framework.router.router import AgentRouter


@pytest.fixture
def agent_router():
    """Fixture to create an instance of AgentRouter for testing."""
    return AgentRouter()


def test_register_agent(agent_router):
    """Test registering an agent."""
    mock_agent = MagicMock()
    agent_router.register_agent(VisionTaskType.OBJECT_DETECTION, mock_agent)

    assert VisionTaskType.OBJECT_DETECTION in agent_router.agents
    assert agent_router.agents[VisionTaskType.OBJECT_DETECTION] == mock_agent


def test_determine_task_type_detection(agent_router):
    """Test task type determination for detection."""
    user_comment = "Find a person"
    agent_router.nlp_processor.parse_query = MagicMock(
        return_value=("detection", ["person"])
    )
    agent_router.agents[VisionTaskType.OBJECT_DETECTION] = MagicMock()
    agent_router.agents[VisionTaskType.OBJECT_DETECTION].model = MagicMock(
        names={0: "person"}
    )

    task_type, additional_params = agent_router.determine_task_type(user_comment)

    assert task_type == VisionTaskType.OBJECT_DETECTION
    assert additional_params == {"detect_classes": ["person"]}


def test_determine_task_type_default(agent_router):
    """Test default task type determination when no valid task is found."""
    user_comment = "Do something"
    agent_router.nlp_processor.parse_query = MagicMock(
        return_value=("unknown_task", [])
    )

    task_type, additional_params = agent_router.determine_task_type(user_comment)

    assert task_type == VisionTaskType.OBJECT_DETECTION
    assert additional_params == {}


def test_process_request(agent_router):
    """Test processing a request."""
    mock_agent = MagicMock()
    mock_agent.process = MagicMock(
        return_value=VisionOutput(
            task_type=VisionTaskType.OBJECT_DETECTION,
            results={},
            confidence=0.0,
            processing_time=0.0,
        )
    )
    agent_router.register_agent(VisionTaskType.OBJECT_DETECTION, mock_agent)

    # Providing a mock image since VisionInput requires it
    mock_image = MagicMock()  # Replace with a valid image if needed
    vision_input = VisionInput(
        image=mock_image,
        task_type=VisionTaskType.OBJECT_DETECTION,
        user_comment="Detect objects",
    )

    result = agent_router.process_request(vision_input)

    assert result is not None
    mock_agent.process.assert_called_once_with(vision_input)


def test_process_request_no_agent(agent_router):
    """Test processing a request when no agent is registered for the task type."""
    mock_image = MagicMock()  # Replace with a valid image if needed
    vision_input = VisionInput(
        image=mock_image, task_type=VisionTaskType.OCR, user_comment="Read text"
    )

    with pytest.raises(
        ValueError, match="No agent registered for task type: VisionTaskType.OCR"
    ):
        agent_router.process_request(vision_input)


def test_process_request_with_user_comment(agent_router):
    """Test processing a request when task type is None and user comment is given."""
    mock_agent = MagicMock()
    agent_router.register_agent(VisionTaskType.OBJECT_DETECTION, mock_agent)
    mock_agent.model = MagicMock(names={0: "person"})

    # Providing a mock image since VisionInput requires it
    mock_image = MagicMock()  # Replace with a valid image if needed
    vision_input = VisionInput(
        image=mock_image, task_type=None, user_comment="detect a person"
    )
    agent_router.nlp_processor.parse_query = MagicMock(
        return_value=("detection", ["person"])
    )

    result = agent_router.process_request(vision_input)

    assert result is not None
    assert vision_input.task_type == VisionTaskType.OBJECT_DETECTION
    assert vision_input.additional_params == {"detect_classes": ["person"]}
    mock_agent.process.assert_called_once_with(vision_input)


def test_task_mapping(agent_router):
    """Test task mapping correctness."""
    assert agent_router.task_mapping["detection"] == VisionTaskType.OBJECT_DETECTION
    assert (
        agent_router.task_mapping["classification"]
        == VisionTaskType.IMAGE_CLASSIFICATION
    )
    assert agent_router.task_mapping["segmentation"] == VisionTaskType.SEGMENTATION
    assert agent_router.task_mapping["ocr"] == VisionTaskType.OCR
    assert agent_router.task_mapping["face"] == VisionTaskType.FACE_DETECTION


def test_determine_task_type_with_invalid_target(agent_router):
    """Test target object validation with an invalid class."""
    user_comment = "Find a unicorn"
    agent_router.nlp_processor.parse_query = MagicMock(
        return_value=("detection", ["unicorn"])
    )
    agent_router.agents[VisionTaskType.OBJECT_DETECTION] = MagicMock()
    agent_router.agents[VisionTaskType.OBJECT_DETECTION].model = MagicMock(
        names={0: "person"}
    )

    task_type, additional_params = agent_router.determine_task_type(user_comment)

    assert task_type == VisionTaskType.OBJECT_DETECTION
    assert additional_params == {}
