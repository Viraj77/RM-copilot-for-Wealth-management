"""
Agent State Schema — Phase 4: LangGraph Agents.

Defines the TypedDict state that flows through the LangGraph nodes.
Tracks conversation history, context variables, and final outputs.
"""

from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State dictionary for the Wealth Management Agent.
    
    Attributes:
        messages: The chat history, including user messages, AI responses, and tool calls.
                  The `add_messages` reducer appends new messages to the list.
        client_id: The ID of the client being discussed (e.g., 'C-204').
        rm_tier: The entitlement tier of the RM ('standard', 'premium', 'institutional').
        current_step: Counter to prevent infinite loops.
        final_brief: Populated when the agent decides to generate a formal ClientBrief.
        error: Used to track critical failures.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    client_id: str
    rm_tier: str
    
    current_step: int
    final_brief: Optional[dict]
    error: Optional[str]
