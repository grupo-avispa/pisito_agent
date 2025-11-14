import pytest
import logging
from unittest.mock import Mock, AsyncMock, patch
from pisito_agent.langgraph_home_assistant import LangGraphManager
from langgraph_base_ros.ollama_utils import Ollama, Messages, Message


# --------------------------------------------------
# FIXTURE: Mock ROS2 logger
# --------------------------------------------------
@pytest.fixture
def mock_ros_logger():
    """Mock ROS2 logger for testing."""
    logger = Mock()
    logger.info = Mock()
    return logger


# --------------------------------------------------
# FIXTURE: Mock Ollama agent
# --------------------------------------------------
@pytest.fixture
def mock_ollama_agent():
    """Mock Ollama agent for testing."""
    agent = Mock(spec=Ollama)
    agent.invoke = AsyncMock()
    agent.reset_memory = Mock()
    agent.create_message = Mock(side_effect=lambda **kwargs: Message(**kwargs))
    return agent


# --------------------------------------------------
# FIXTURE: LangGraphManager with mocked dependencies
# --------------------------------------------------
@pytest.fixture
def graph_manager(mock_ros_logger, mock_ollama_agent):
    """Create LangGraphManager instance with mocked dependencies."""
    return LangGraphManager(
        logger=mock_ros_logger,
        ollama_agent=mock_ollama_agent,
        max_steps=5
    )


# --------------------------------------------------
# TEST 1: Initialization
# --------------------------------------------------
def test_init_without_ollama_raises_error():
    """LangGraphManager should raise ValueError if Ollama agent is not provided."""
    with pytest.raises(ValueError, match="Ollama agent instance must be provided"):
        LangGraphManager(ollama_agent=None)


def test_init_with_ollama_agent(graph_manager, mock_ollama_agent):
    """LangGraphManager should initialize correctly with Ollama agent."""
    assert graph_manager.ollama_agent == mock_ollama_agent
    assert graph_manager.max_steps == 5
    assert graph_manager.steps == 0
    assert graph_manager.messages_count == 0
    assert graph_manager.graph is None


def test_init_with_ros_logger(mock_ros_logger, mock_ollama_agent):
    """LangGraphManager should accept ROS2 logger."""
    manager = LangGraphManager(
        logger=mock_ros_logger,
        ollama_agent=mock_ollama_agent
    )
    assert manager.logger == mock_ros_logger


def test_init_without_logger(mock_ollama_agent):
    """LangGraphManager should work without logger (using Python logging)."""
    manager = LangGraphManager(ollama_agent=mock_ollama_agent)
    assert manager.logger is None


# --------------------------------------------------
# TEST 2: _log method
# --------------------------------------------------
def test_log_with_ros_logger(graph_manager, mock_ros_logger):
    """_log should use ROS2 logger if available."""
    graph_manager._log("Test message")
    mock_ros_logger.info.assert_called_once_with("Test message")


def test_log_without_ros_logger(mock_ollama_agent):
    """_log should use Python logging if ROS2 logger is not available."""
    manager = LangGraphManager(logger=None, ollama_agent=mock_ollama_agent)
    
    with patch('logging.info') as mock_logging:
        manager._log("Test message")
        # assert_called_once_with verifies that logging.info was called once with "Test message"
        mock_logging.assert_called_once_with("Test message")


# --------------------------------------------------
# TEST 3: query_response
# --------------------------------------------------
@pytest.mark.asyncio
async def test_query_response_invokes_ollama(graph_manager, mock_ollama_agent):
    """query_response should invoke Ollama agent and return updated state."""
    initial_state = Messages(messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello")
    ])
    
    expected_state = Messages(messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!")
    ])
    
    mock_ollama_agent.invoke.return_value = expected_state
    
    result = await graph_manager.query_response(initial_state)
    
    mock_ollama_agent.invoke.assert_called_once_with(state=initial_state)
    assert result == expected_state


# --------------------------------------------------
# TEST 4: manage_steps - Tool call detected
# --------------------------------------------------
def test_manage_steps_with_tool_call(graph_manager, mock_ros_logger):
    """manage_steps should return 'agent' when tool call is detected and steps < max_steps."""
    state = Messages(messages=[
        Message(role="user", content="Turn on lights"),
        Message(role="assistant", tool_calls=[{"name": "light_control", "arguments": {}}]),
        Message(role="tool", content="Light turned on")
    ])
    
    result = graph_manager.manage_steps(state)
    
    assert result == "agent"
    assert graph_manager.steps == 1
    assert graph_manager.messages_count == 3


def test_manage_steps_max_steps_reached(graph_manager, mock_ros_logger):
    """manage_steps should return 'finish' when max steps is reached."""
    graph_manager.steps = 4  # One step before max
    
    state = Messages(messages=[
        Message(role="tool", content="Action completed")
    ])
    
    result = graph_manager.manage_steps(state)
    
    assert result == "finish"
    assert graph_manager.steps == 5
    mock_ros_logger.info.assert_any_call("Maximum steps reached, finishing interaction.")


# --------------------------------------------------
# TEST 5: manage_steps - No tool call
# --------------------------------------------------
def test_manage_steps_no_tool_call(graph_manager, mock_ros_logger):
    """manage_steps should return 'finish' when no tool call is detected."""
    state = Messages(messages=[
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!")
    ])
    
    result = graph_manager.manage_steps(state)
    
    assert result == "finish"
    assert graph_manager.steps == 1
    mock_ros_logger.info.assert_any_call("No tool call detected, finishing interaction.")


def test_manage_steps_empty_messages(graph_manager):
    """manage_steps should handle empty messages list."""
    state = Messages(messages=[])
    
    result = graph_manager.manage_steps(state)
    
    assert result == "finish"
    assert graph_manager.steps == 1


# --------------------------------------------------
# TEST 6: finish_ollama_interaction
# --------------------------------------------------
@pytest.mark.asyncio
async def test_finish_ollama_interaction(graph_manager, mock_ollama_agent, mock_ros_logger):
    """finish_ollama_interaction should reset memory and return state."""
    state = Messages(messages=[
        Message(role="user", content="Test"),
        Message(role="assistant", content="Response")
    ])
    
    result = await graph_manager.finish_ollama_interaction(state)
    
    assert result == state
    mock_ollama_agent.reset_memory.assert_called_once()
    mock_ros_logger.info.assert_any_call("Finalizing Ollama interaction.")


@pytest.mark.asyncio
async def test_finish_ollama_interaction_max_steps(graph_manager, mock_ros_logger):
    """finish_ollama_interaction should log when max steps is reached."""
    graph_manager.steps = 5
    state = Messages(messages=[Message(role="assistant", content="Done")])
    
    await graph_manager.finish_ollama_interaction(state)
    
    mock_ros_logger.info.assert_any_call("Maximum steps reached during finalization.")


@pytest.mark.asyncio
async def test_finish_ollama_interaction_before_max_steps(graph_manager, mock_ros_logger):
    """finish_ollama_interaction should log when finishing before max steps."""
    graph_manager.steps = 2
    state = Messages(messages=[Message(role="assistant", content="Done")])
    
    await graph_manager.finish_ollama_interaction(state)
    
    mock_ros_logger.info.assert_any_call("Agent reached final state before maximum steps.")


# --------------------------------------------------
# TEST 7: make_graph
# --------------------------------------------------
@pytest.mark.asyncio
async def test_make_graph_creates_workflow(graph_manager):
    """make_graph should create and compile the LangGraph workflow."""
    await graph_manager.make_graph()
    
    assert graph_manager.graph is not None
    # Verify that graph has the expected structure
    assert hasattr(graph_manager.graph, 'ainvoke')


@pytest.mark.asyncio
async def test_make_graph_workflow_execution(graph_manager, mock_ollama_agent):
    """make_graph should create a workflow that can be executed."""
    # Setup mock to return state without tool calls to finish immediately
    async def mock_invoke(state):
        return Messages(messages=[
            *state['messages'],
            Message(role="assistant", content="Response")
        ])
    
    mock_ollama_agent.invoke = AsyncMock(side_effect=mock_invoke)
    
    await graph_manager.make_graph()
    
    initial_state = Messages(messages=[
        Message(role="system", content="System prompt"),
        Message(role="user", content="Hello")
    ])
    
    result = await graph_manager.graph.ainvoke(initial_state)
    
    assert "messages" in result
    assert len(result["messages"]) >= 3


# --------------------------------------------------
# TEST 8: Integration test - Full workflow with tool calls
# --------------------------------------------------
@pytest.mark.asyncio
async def test_full_workflow_with_tool_calls(graph_manager, mock_ollama_agent):
    """Test complete workflow with multiple tool calls."""
    call_count = 0
    
    async def mock_invoke_with_tools(state):
        nonlocal call_count
        call_count += 1
        
        if call_count == 1:
            # First call: return tool call
            return Messages(messages=[
                *state['messages'],
                Message(role="assistant", tool_calls=[{"name": "test_tool", "arguments": {}}]),
                Message(role="tool", content="Tool result")
            ])
        else:
            # Second call: return final response
            return Messages(messages=[
                *state['messages'],
                Message(role="assistant", content="Final response")
            ])
    
    mock_ollama_agent.invoke = AsyncMock(side_effect=mock_invoke_with_tools)
    
    await graph_manager.make_graph()
    
    initial_state = Messages(messages=[
        Message(role="system", content="System"),
        Message(role="user", content="Query")
    ])
    
    result = await graph_manager.graph.ainvoke(initial_state)
    
    assert call_count == 2
    assert graph_manager.steps == 2
    mock_ollama_agent.reset_memory.assert_called_once()


# --------------------------------------------------
# TEST 9: Edge cases
# --------------------------------------------------
def test_manage_steps_updates_message_count(graph_manager):
    """manage_steps should correctly update messages_count."""
    state = Messages(messages=[
        Message(role="user", content="1"),
        Message(role="assistant", content="2"),
        Message(role="user", content="3")
    ])
    
    graph_manager.manage_steps(state)
    
    assert graph_manager.messages_count == 3


def test_manage_steps_increments_correctly(graph_manager):
    """manage_steps should increment steps counter correctly."""
    state = Messages(messages=[Message(role="assistant", content="Test")])
    
    initial_steps = graph_manager.steps
    graph_manager.manage_steps(state)
    
    assert graph_manager.steps == initial_steps + 1