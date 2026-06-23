"""
Agent Nodes — Phase 4: LangGraph Agents.

Contains the individual functions (nodes) that make up the LangGraph state machine.
"""

import json
from typing import Literal, Optional

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import settings
from src.agent.prompts import SYSTEM_PROMPT_TEMPLATE, BRIEF_SYNTHESIS_PROMPT
from src.agent.state import AgentState
from src.guardrails.citation_validator import validate_and_patch_brief
from src.tools import (
    get_market_data_tool,
    get_portfolio_tool,
    get_rag_retriever_tool,
    get_suitability_checker_tool,
)

# ── Tool Registry ─────────────────────────────────────────────────────────────

def get_agent_tools():
    """Return the suite of tools available to the agent."""
    return [
        get_portfolio_tool(),
        get_market_data_tool(),
        get_rag_retriever_tool(),
        get_suitability_checker_tool(),
    ]


def _get_llm_with_tools():
    """Return the ChatOpenAI instance with tools bound."""
    llm_base = ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        api_key=settings.openai_api_key,
    )
    return llm_base.bind_tools(get_agent_tools())


def _get_llm_base() -> ChatOpenAI:
    """Return the base ChatOpenAI instance (no tools bound)."""
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        api_key=settings.openai_api_key,
    )


# ── Brief-request keyword detection ──────────────────────────────────────────

_BRIEF_REQUEST_KEYWORDS = [
    "generate brief", "prepare brief", "create brief",
    "generate a brief", "prepare a brief", "create a brief",
    "draft brief", "draft a brief", "write a brief",
    "meeting brief", "client brief", "produce brief",
    "compile brief", "summarize brief", "make a brief",
]


def _is_brief_requested(state: AgentState) -> bool:
    """
    Return True if any user message in the conversation explicitly
    requests a ClientBrief to be generated.
    """
    from langchain_core.messages import HumanMessage
    messages = state.get("messages", [])
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content_lower = (msg.content or "").lower()
            if any(kw in content_lower for kw in _BRIEF_REQUEST_KEYWORDS):
                return True
    return False


# ── Nodes ─────────────────────────────────────────────────────────────────────

def call_model(state: AgentState) -> dict:
    """
    Invokes the core LLM to reason, answer questions, or decide to call tools.
    Uses the module-level LLM singleton to avoid repeated client instantiation.
    """
    messages = state.get("messages", [])
    current_step = state.get("current_step", 0)

    # Inject dynamic system prompt if it's the first step
    if not messages or not isinstance(messages[0], SystemMessage):
        sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            client_id=state.get("client_id", "Unknown"),
            rm_tier=state.get("rm_tier", "standard"),
        )
        messages = [SystemMessage(content=sys_prompt)] + list(messages)

    # Use singleton LLM with tools bound
    llm_with_tools = _get_llm_with_tools()

    with open("agent_input.log", "a") as f:
        f.write(f"--- AGENT CALLED ---\n")
        for m in messages:
            f.write(f"ROLE: {m.type}\n")
            f.write(f"CONTENT: {m.content}\n")
            if hasattr(m, 'tool_calls'):
                f.write(f"TOOL_CALLS: {m.tool_calls}\n")
        f.write("--------------------\n")

    # Call LLM
    response = llm_with_tools.invoke(messages)

    # Return partial state update (LangGraph's add_messages will append)
    return {
        "messages": [response],
        "current_step": current_step + 1,
    }


def synthesize_brief(state: AgentState) -> dict:
    """
    Optional post-processing node that generates a structured ClientBrief.

    Runs only when the RM explicitly requested a brief (Option B).
    Extracts context from the full conversation history and uses a
    second LLM call in JSON mode to populate AgentState.final_brief.

    If brief synthesis fails or was not requested, final_brief is left as None.
    """
    # Only synthesize if explicitly requested
    if not _is_brief_requested(state):
        return {}

    messages = state.get("messages", [])
    client_id = state.get("client_id", "Unknown")
    rm_tier = state.get("rm_tier", "standard")

    # Build conversation history summary for the synthesis prompt
    conversation_text = []
    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        content = getattr(msg, "content", "") or ""
        if content.strip():
            conversation_text.append(f"{role}: {content.strip()[:1000]}")

    conversation_str = "\n\n".join(conversation_text[-20:])  # Last 20 messages max

    synthesis_prompt = BRIEF_SYNTHESIS_PROMPT.format(
        client_id=client_id,
        rm_tier=rm_tier,
        conversation=conversation_str,
    )

    try:
        llm = _get_llm_base()
        # Use JSON mode for structured output
        llm_json = llm.bind(response_format={"type": "json_object"})
        response = llm_json.invoke([SystemMessage(content=synthesis_prompt)])

        raw = response.content or "{}"
        brief_dict = json.loads(raw)

        # Stamp metadata
        brief_dict["client_id"] = client_id
        brief_dict["rm_tier"] = rm_tier

        # Feature 3 — Citation Hallucination Validation
        # Cross-check every cited chunk_id against what was actually retrieved.
        # Patches brief_dict in-place if hallucinations are detected.
        messages_list = state.get("messages", [])
        validate_and_patch_brief(brief_dict, messages_list)

        return {"final_brief": brief_dict}

    except Exception as exc:
        # Log but don't crash — brief synthesis failure is non-fatal
        print(f"\n  [WARN] Brief synthesis failed: {exc}. final_brief will be None.")
        return {}


def should_continue(state: AgentState) -> Literal["tools", "synthesize_brief", "__end__"]:
    """
    Conditional edge routing logic.
    Decides whether to:
     - Route to 'tools' if the LLM made tool calls
     - Route to 'synthesize_brief' if the conversation is done
     - Force '__end__' if max steps are exceeded
    """
    messages = state.get("messages", [])
    current_step = state.get("current_step", 0)

    # Failsafe: prevent infinite loops
    if current_step >= settings.agent_max_steps:
        print(f"\n  [WARN] Agent reached max steps ({settings.agent_max_steps}). Forcing exit.")
        return "__end__"

    last_message = messages[-1]

    # If the LLM made tool calls, route to the 'tools' node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, conversation is done — attempt brief synthesis then end
    return "synthesize_brief"
