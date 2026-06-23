"""
End-to-End Smoke Tests — Fix 9.

Tests the full copilot_app.invoke() pipeline with a mocked LLM
(no real OpenAI API call made). Verifies the graph executes without errors,
produces the expected message structure, and that the synthesize_brief node
runs correctly.

Run: pytest tests/test_e2e_smoke.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.graph import copilot_app
from src.agent.state import AgentState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_initial_state(prompt: str, client_id: str = "C-204", rm_tier: str = "standard") -> AgentState:
    return {
        "messages": [HumanMessage(content=prompt)],
        "client_id": client_id,
        "rm_tier": rm_tier,
        "current_step": 0,
        "final_brief": None,
        "error": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Structure Tests (no LLM needed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphStructure:
    def test_graph_compiled_successfully(self):
        """Verify the graph compiled without errors."""
        assert copilot_app is not None

    def test_graph_has_required_nodes(self):
        """All expected nodes are present in the compiled graph."""
        nodes = set(copilot_app.nodes.keys())
        assert "agent" in nodes, "Missing 'agent' node"
        assert "tools" in nodes, "Missing 'tools' node"
        assert "synthesize_brief" in nodes, "Missing 'synthesize_brief' node"

    def test_graph_input_schema_matches_agent_state(self):
        """The graph's input schema includes the AgentState fields."""
        # LangGraph wraps the schema under $defs in Pydantic v2
        try:
            schema = copilot_app.input_schema.model_json_schema()
        except AttributeError:
            schema = copilot_app.input_schema.schema()  # Pydantic v1 fallback

        # Properties may be at top level or nested under $defs → AgentState
        props = schema.get("properties", {})
        if not props:
            defs = schema.get("$defs", {})
            for _name, defn in defs.items():
                candidate = defn.get("properties", {})
                if "messages" in candidate:
                    props = candidate
                    break

        for field in ["messages", "client_id", "rm_tier", "current_step"]:
            assert field in props, f"Missing field '{field}' in graph input schema"




# ═══════════════════════════════════════════════════════════════════════════════
# End-to-End Smoke Tests (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestE2ESmokeInvoke:
    """
    Full invoke() tests with ChatOpenAI mocked to return a plain-text
    response (no tool calls). This exercises the full graph flow:
    START → agent → synthesize_brief → END
    """

    @patch("src.agent.nodes._llm_with_tools", None)
    @patch("src.agent.nodes._llm_base", None)
    @patch("src.agent.nodes.ChatOpenAI")
    def test_smoke_invoke_no_tool_calls(self, mock_chat_openai):
        """
        Full graph run with mocked LLM that returns plain text (no tool calls).
        Verifies the conversation completes without exceptions.
        """
        final_ai_text = "The portfolio is well-diversified across equity and fixed income."
        mock_ai_response = AIMessage(content=final_ai_text)

        mock_llm_instance = MagicMock()
        mock_llm_with_tools = MagicMock(invoke=MagicMock(return_value=mock_ai_response))
        mock_llm_instance.bind_tools.return_value = mock_llm_with_tools
        mock_llm_instance.bind.return_value = MagicMock(
            invoke=MagicMock(return_value=AIMessage(content='{"client_id": "C-204"}'))
        )
        mock_chat_openai.return_value = mock_llm_instance

        initial_state = _make_initial_state("Summarise the portfolio.")
        result = copilot_app.invoke(initial_state)

        # Verify the result state has messages
        assert "messages" in result
        messages = result["messages"]
        assert len(messages) >= 2, (
            f"Expected at least HumanMessage + AIMessage, got {len(messages)} messages"
        )

        # Verify one of the messages is the AI's response
        ai_contents = [
            m.content for m in messages
            if isinstance(m, AIMessage) and m.content
        ]
        assert any(final_ai_text in c for c in ai_contents), (
            f"Expected AI response in messages; found: {ai_contents}"
        )

    @patch("src.agent.nodes._llm_with_tools", None)
    @patch("src.agent.nodes._llm_base", None)
    @patch("src.agent.nodes.ChatOpenAI")
    def test_smoke_invoke_final_brief_populated_on_request(self, mock_chat_openai):
        """
        When the query explicitly requests a brief, final_brief should be
        populated in the state after graph execution.
        """
        import json

        sample_brief = {
            "client_id": "C-204",
            "client_name": "Sarah Chen",
            "risk_profile": "balanced",
            "portfolio_summary": "60% equity, 30% bonds, 10% cash.",
            "portfolio_risk_assessment": "Aligned with balanced profile.",
            "total_aum": 2400000,
            "allocation_breakdown": {"equity": 60, "fixed_income": 30, "cash": 10},
            "recommendations": [],
            "compliance_status": "cleared",
            "compliance_notes": "",
            "talking_points": ["Portfolio is on track.", "Consider reviewing bonds."],
        }

        # First LLM call (agent node): returns plain text
        agent_response = AIMessage(content="Here is the portfolio overview.")
        # Second LLM call (synthesize_brief node): returns JSON brief
        synthesis_response = AIMessage(content=json.dumps(sample_brief))

        mock_llm_instance = MagicMock()
        # bind_tools().invoke() → agent response
        mock_llm_instance.bind_tools.return_value = MagicMock(
            invoke=MagicMock(return_value=agent_response)
        )
        # bind(response_format=...).invoke() → synthesis response
        mock_llm_instance.bind.return_value = MagicMock(
            invoke=MagicMock(return_value=synthesis_response)
        )
        mock_chat_openai.return_value = mock_llm_instance

        initial_state = _make_initial_state(
            "Please generate a brief for C-204.",
            client_id="C-204",
            rm_tier="premium",
        )
        result = copilot_app.invoke(initial_state)

        # final_brief should be populated
        assert result.get("final_brief") is not None, (
            "final_brief should be set when brief is explicitly requested"
        )
        assert result["final_brief"]["client_id"] == "C-204"

    @patch("src.agent.nodes._llm_with_tools", None)
    @patch("src.agent.nodes._llm_base", None)
    @patch("src.agent.nodes.ChatOpenAI")
    def test_smoke_invoke_final_brief_not_set_for_regular_query(self, mock_chat_openai):
        """
        For a regular query (no brief keyword), final_brief remains None.
        """
        mock_ai_response = AIMessage(content="The equity allocation looks good.")

        mock_llm_instance = MagicMock()
        mock_llm_instance.bind_tools.return_value = MagicMock(
            invoke=MagicMock(return_value=mock_ai_response)
        )
        mock_llm_instance.bind.return_value = MagicMock(
            invoke=MagicMock(return_value=AIMessage(content="{}"))
        )
        mock_chat_openai.return_value = mock_llm_instance

        initial_state = _make_initial_state("What is the equity allocation?")
        result = copilot_app.invoke(initial_state)

        # final_brief should remain unset / None for a regular query
        assert result.get("final_brief") is None, (
            "final_brief should be None for non-brief queries"
        )

    @patch("src.agent.nodes._llm_with_tools", None)
    @patch("src.agent.nodes._llm_base", None)
    @patch("src.agent.nodes.ChatOpenAI")
    def test_smoke_invoke_does_not_raise_on_synthesis_failure(self, mock_chat_openai):
        """
        If brief synthesis raises an exception, the graph should still complete
        gracefully (synthesis failure is non-fatal).
        """
        agent_response = AIMessage(content="Analysis complete.")

        mock_llm_instance = MagicMock()
        mock_llm_instance.bind_tools.return_value = MagicMock(
            invoke=MagicMock(return_value=agent_response)
        )
        # Simulate synthesis LLM failure
        mock_llm_instance.bind.return_value = MagicMock(
            invoke=MagicMock(side_effect=RuntimeError("OpenAI timeout"))
        )
        mock_chat_openai.return_value = mock_llm_instance

        initial_state = _make_initial_state(
            "Please generate a brief for C-204.",
            rm_tier="premium",
        )

        # Should not raise — synthesis failure is caught inside synthesize_brief
        result = copilot_app.invoke(initial_state)
        assert "messages" in result
