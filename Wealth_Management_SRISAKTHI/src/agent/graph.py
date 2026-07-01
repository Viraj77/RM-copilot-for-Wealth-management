"""LangGraph StateGraph assembly."""

import os
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from src.agent.nodes import (
    check_suitability_node,
    gather_market_node,
    gather_portfolio_node,
    gather_research_node,
    plan_node,
    react_tools_node,
    review_gate_node,
    synthesize_node,
    _steps_exceeded,
)
from src.agent.state import AgentState
from src.config import settings
from src.guardrails.compliance import apply_review_gate
from src.models import ClientBrief


def _route_after_plan(state: AgentState) -> str:
    if _steps_exceeded(state):
        return "synthesize"
    query = state["query"].lower()
    has_client = bool(state.get("client_id")) or "client c-" in query
    if ("suitable" in query or "pg-" in query) and not has_client:
        return "suitability_path"
    return "gather_portfolio"


def _route_after_portfolio(state: AgentState) -> str:
    if _steps_exceeded(state):
        return "synthesize"
    query = state["query"].lower()
    if "portfolio risk" in query or (
        "summarize" in query and "prepare" not in query and "talking" not in query
    ):
        return "gather_market"
    return "gather_research"


def build_graph():
    """Build the LangGraph agent pipeline with ReAct tools and HITL interrupt."""
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("react_tools", react_tools_node)
    graph.add_node("gather_portfolio", gather_portfolio_node)
    graph.add_node("gather_research", gather_research_node)
    graph.add_node("gather_market", gather_market_node)
    graph.add_node("check_suitability", check_suitability_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("review_gate", review_gate_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "react_tools")

    graph.add_conditional_edges(
        "react_tools",
        _route_after_plan,
        {
            "gather_portfolio": "gather_portfolio",
            "suitability_path": "gather_research",
            "synthesize": "synthesize",
        },
    )

    graph.add_conditional_edges(
        "gather_portfolio",
        _route_after_portfolio,
        {
            "gather_research": "gather_research",
            "gather_market": "gather_market",
            "synthesize": "synthesize",
        },
    )

    graph.add_edge("gather_research", "gather_market")
    graph.add_edge("gather_market", "check_suitability")
    graph.add_edge("check_suitability", "synthesize")
    graph.add_edge("synthesize", "review_gate")

    def _route_after_review(state: AgentState) -> str:
        """Conditional compliance gate: cleared briefs finish; escalations use HITL."""
        brief = state.get("final_brief") or state.get("draft_brief")
        if state.get("requires_escalation"):
            return "escalated"
        if brief and getattr(brief, "compliance_status", None):
            from src.models import ComplianceStatus
            if brief.compliance_status != ComplianceStatus.CLEARED:
                return "escalated"
        return "cleared"

    graph.add_conditional_edges(
        "review_gate",
        _route_after_review,
        {"cleared": END, "escalated": END},
    )

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def _setup_langsmith() -> None:
    if settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        if settings.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key


def _initial_state(
    query: str,
    rm_id: str,
    rm_entitlements: list[str] | None,
    client_id: str | None,
) -> AgentState:
    return {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "client_id": client_id,
        "rm_id": rm_id,
        "rm_entitlements": rm_entitlements or ["public", "internal"],
        "plan": "",
        "portfolio": None,
        "portfolio_summary": "",
        "retrieved_chunks": [],
        "research_context": "",
        "market_context": "",
        "suitability_results": [],
        "draft_brief": None,
        "final_brief": None,
        "step_count": 0,
        "requires_escalation": False,
        "human_approved": None,
        "tools_trace": [],
        "error": None,
    }


def run_agent(
    query: str,
    rm_id: str = "RM-001",
    rm_entitlements: list[str] | None = None,
    client_id: str | None = None,
    thread_id: str = "default",
    resume: dict[str, Any] | None = None,
) -> ClientBrief | dict[str, Any]:
    """Run agent pipeline. Returns ClientBrief or interrupt dict for HITL resume."""
    _setup_langsmith()
    agent = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    if resume is not None:
        result = agent.invoke(Command(resume=resume), config=config)
    else:
        result = agent.invoke(
            _initial_state(query, rm_id, rm_entitlements, client_id),
            config=config,
        )

    snapshot = agent.get_state(config)
    if snapshot.next:
        values = snapshot.values or result
        return {
            "interrupted": True,
            "thread_id": thread_id,
            "next": snapshot.next,
            "draft_brief": values.get("draft_brief"),
            "requires_escalation": values.get("requires_escalation", False),
            "interrupt_payload": values,
        }

    if result.get("final_brief"):
        return result["final_brief"]

    if result.get("draft_brief"):
        return apply_review_gate(
            result["draft_brief"],
            query,
            requires_escalation=result.get("requires_escalation", False),
        )

    raise RuntimeError("Agent did not produce a ClientBrief")


def resume_agent(thread_id: str, approved: bool = True) -> ClientBrief | dict[str, Any]:
    """Resume after human-in-the-loop interrupt."""
    return run_agent(
        query="",
        thread_id=thread_id,
        resume={"approved": approved},
    )
