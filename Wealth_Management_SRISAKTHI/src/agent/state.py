"""LangGraph agent state definition."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from src.models import ClientBrief, ClientPortfolio, RetrievedChunk


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    client_id: str | None
    rm_id: str
    rm_entitlements: list[str]
    plan: str
    portfolio: ClientPortfolio | None
    portfolio_summary: str
    retrieved_chunks: list[RetrievedChunk]
    research_context: str
    market_context: str
    suitability_results: list[str]
    draft_brief: ClientBrief | None
    final_brief: ClientBrief | None
    step_count: int
    requires_escalation: bool
    human_approved: bool | None
    tools_trace: list[str]
    error: str | None
