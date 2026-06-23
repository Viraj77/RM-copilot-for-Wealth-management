"""
LangGraph Assembly — Phase 4: LangGraph Agents.

Wires the state, nodes, and tools together into a compiled StateGraph.
This is the main entry point for running the Copilot agent.

Graph topology:
    START → agent → (tools → agent)* → synthesize_brief → END
"""

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.nodes import call_model, get_agent_tools, should_continue, synthesize_brief
from src.agent.state import AgentState


def build_graph():
    """
    Constructs and compiles the Wealth Manager Copilot graph.

    The graph loop:
    START -> agent -> (tools or synthesize_brief) -> agent -> ...
                                                  -> synthesize_brief -> END

    The 'synthesize_brief' node is always visited before END. It is a
    no-op (returns {}) if the RM did not explicitly request a ClientBrief,
    so it adds negligible latency in the normal conversational case.
    """
    class SafeToolNode(ToolNode):
        def __call__(self, input, config=None, **kwargs):
            try:
                result = super().__call__(input, config, **kwargs)
                with open("tool_call.log", "a") as f:
                    f.write(f"TOOL RETURNED: {result}\n")
                return result
            except Exception as e:
                import traceback
                with open("tool_call.log", "a") as f:
                    f.write(f"TOOL EXCEPTION: {e}\n")
                    traceback.print_exc(file=f)
                raise

    # 1. Initialise the graph with our state schema
    workflow = StateGraph(AgentState)

    # 2. Add nodes
    # 'agent' node invokes the LLM
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", SafeToolNode(get_agent_tools()))
    workflow.add_node("synthesize_brief", synthesize_brief)

    # 3. Define edges and control flow
    workflow.add_edge(START, "agent")

    # Conditional edge: after 'agent' runs, decide whether to use tools,
    # run brief synthesis, or stop (max-steps failsafe only → __end__)
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "synthesize_brief": "synthesize_brief",
            "__end__": END,
        },
    )

    # After tools execute, always return to the agent to interpret results
    workflow.add_edge("tools", "agent")

    # After brief synthesis, always end
    workflow.add_edge("synthesize_brief", END)

    # 4. Compile into a runnable application
    return workflow.compile()


# Create a singleton instance for easy import
copilot_app = build_graph()
