"""
Unit tests for the LangGraph Agent — Phase 4.

Verifies the state schema, node functions, and graph assembly.
Mocking out the LLM calls to test the routing logic and node behaviour.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.nodes import should_continue, _is_brief_requested
from src.agent.graph import copilot_app
from config.settings import settings


# ═══════════════════════════════════════════════════════════════════════════════
# AgentState schema tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_agent_state_schema():
    """Verify that the AgentState TypedDict allows required fields."""
    state: AgentState = {
        "messages": [HumanMessage(content="Hello")],
        "client_id": "C-204",
        "rm_tier": "premium",
        "current_step": 0,
        "final_brief": None,
        "error": None,
    }
    assert state["client_id"] == "C-204"
    assert len(state["messages"]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# should_continue routing tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_should_continue_routes_to_tools():
    """If the LLM returns tool calls, should_continue routes to 'tools'."""
    ai_message_with_tools = AIMessage(
        content="",
        tool_calls=[{"name": "portfolio_lookup_tool", "args": {"client_id": "C-204"}, "id": "call_123"}],
    )
    state: AgentState = {
        "messages": [ai_message_with_tools],
        "client_id": "C-204",
        "rm_tier": "premium",
        "current_step": 1,
        "final_brief": None,
        "error": None,
    }
    assert should_continue(state) == "tools"


def test_should_continue_routes_to_synthesize_brief_when_done():
    """If the LLM returns plain text, should_continue routes to 'synthesize_brief'."""
    ai_message = AIMessage(content="Here is the client's portfolio...")
    state: AgentState = {
        "messages": [ai_message],
        "client_id": "C-204",
        "rm_tier": "premium",
        "current_step": 1,
        "final_brief": None,
        "error": None,
    }
    assert should_continue(state) == "synthesize_brief"


def test_should_continue_enforces_max_steps():
    """If the agent exceeds max steps, should_continue routes to '__end__' to prevent loops."""
    ai_message_with_tools = AIMessage(
        content="",
        tool_calls=[{"name": "market_data_tool", "args": {"ticker": "VTI"}, "id": "call_456"}],
    )
    state: AgentState = {
        "messages": [ai_message_with_tools],
        "client_id": "C-204",
        "rm_tier": "premium",
        "current_step": settings.agent_max_steps,  # Reached limit
        "final_brief": None,
        "error": None,
    }
    # Even though there are tool calls, the step limit forces termination
    assert should_continue(state) == "__end__"


# ═══════════════════════════════════════════════════════════════════════════════
# call_model node tests (Fix 7)
# ═══════════════════════════════════════════════════════════════════════════════

@patch("src.agent.nodes._llm_with_tools", None)
@patch("src.agent.nodes._llm_base", None)
@patch("src.agent.nodes.ChatOpenAI")
def test_call_model_injects_system_prompt(mock_chat_openai):
    """System prompt is injected as the first message when none exists."""
    from src.agent.nodes import call_model

    mock_response = AIMessage(content="Portfolio analysis complete.")
    mock_llm_instance = MagicMock()
    mock_llm_instance.bind_tools.return_value = MagicMock(
        invoke=MagicMock(return_value=mock_response)
    )
    mock_chat_openai.return_value = mock_llm_instance

    state: AgentState = {
        "messages": [HumanMessage(content="How is C-204's portfolio?")],
        "client_id": "C-204",
        "rm_tier": "premium",
        "current_step": 0,
        "final_brief": None,
        "error": None,
    }

    result = call_model(state)

    # The LLM should have been called
    mock_llm_instance.bind_tools.return_value.invoke.assert_called_once()
    invoked_messages = mock_llm_instance.bind_tools.return_value.invoke.call_args[0][0]

    # First message must be a SystemMessage
    assert isinstance(invoked_messages[0], SystemMessage), (
        "First message should be SystemMessage (system prompt)"
    )
    assert "C-204" in invoked_messages[0].content
    assert "premium" in invoked_messages[0].content

    # Result should contain the response and incremented step
    assert result["messages"] == [mock_response]
    assert result["current_step"] == 1


@patch("src.agent.nodes._llm_with_tools", None)
@patch("src.agent.nodes._llm_base", None)
@patch("src.agent.nodes.ChatOpenAI")
def test_call_model_increments_step(mock_chat_openai):
    """call_model always increments current_step by 1."""
    from src.agent.nodes import call_model

    mock_response = AIMessage(content="Done.")
    mock_llm_instance = MagicMock()
    mock_llm_instance.bind_tools.return_value = MagicMock(
        invoke=MagicMock(return_value=mock_response)
    )
    mock_chat_openai.return_value = mock_llm_instance

    for start_step in [0, 2, 5]:
        state: AgentState = {
            "messages": [HumanMessage(content="Query")],
            "client_id": "C-204",
            "rm_tier": "standard",
            "current_step": start_step,
            "final_brief": None,
            "error": None,
        }
        result = call_model(state)
        assert result["current_step"] == start_step + 1, (
            f"Expected step {start_step + 1}, got {result['current_step']}"
        )


@patch("src.agent.nodes._llm_with_tools", None)
@patch("src.agent.nodes._llm_base", None)
@patch("src.agent.nodes.ChatOpenAI")
def test_call_model_does_not_duplicate_system_prompt(mock_chat_openai):
    """If a SystemMessage already exists, call_model should NOT prepend another one."""
    from src.agent.nodes import call_model

    mock_response = AIMessage(content="Updated.")
    mock_llm_instance = MagicMock()
    mock_llm_instance.bind_tools.return_value = MagicMock(
        invoke=MagicMock(return_value=mock_response)
    )
    mock_chat_openai.return_value = mock_llm_instance

    existing_system_prompt = SystemMessage(content="Existing system prompt.")
    state: AgentState = {
        "messages": [existing_system_prompt, HumanMessage(content="Follow-up")],
        "client_id": "C-204",
        "rm_tier": "premium",
        "current_step": 2,
        "final_brief": None,
        "error": None,
    }

    call_model(state)

    invoked_messages = mock_llm_instance.bind_tools.return_value.invoke.call_args[0][0]
    system_messages = [m for m in invoked_messages if isinstance(m, SystemMessage)]
    assert len(system_messages) == 1, (
        f"Expected exactly 1 SystemMessage, found {len(system_messages)}"
    )
    # Should be the existing one, not a newly injected one
    assert system_messages[0].content == "Existing system prompt."


# ═══════════════════════════════════════════════════════════════════════════════
# Brief request detection tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_brief_requested_detected():
    """_is_brief_requested returns True when query contains brief keywords."""
    for phrase in ["generate brief", "prepare a brief", "create a brief", "client brief"]:
        state: AgentState = {
            "messages": [HumanMessage(content=f"Please {phrase} for this client.")],
            "client_id": "C-204",
            "rm_tier": "premium",
            "current_step": 0,
            "final_brief": None,
            "error": None,
        }
        assert _is_brief_requested(state) is True, f"Expected brief detected for phrase: '{phrase}'"


def test_brief_not_requested_for_normal_query():
    """_is_brief_requested returns False for a normal portfolio query."""
    state: AgentState = {
        "messages": [HumanMessage(content="How is C-204's equity allocation?")],
        "client_id": "C-204",
        "rm_tier": "premium",
        "current_step": 0,
        "final_brief": None,
        "error": None,
    }
    assert _is_brief_requested(state) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Graph compilation tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_graph_compilation():
    """Verify that the StateGraph compiles without errors."""
    assert copilot_app is not None

    # Verify the structure using the compiled graph's metadata
    nodes = set(copilot_app.nodes.keys())
    assert "agent" in nodes
    assert "tools" in nodes
    assert "synthesize_brief" in nodes


def test_graph_has_synthesize_brief_node():
    """Verify synthesize_brief node is part of the compiled graph."""
    nodes = set(copilot_app.nodes.keys())
    assert "synthesize_brief" in nodes, (
        "synthesize_brief node missing from compiled graph"
    )
