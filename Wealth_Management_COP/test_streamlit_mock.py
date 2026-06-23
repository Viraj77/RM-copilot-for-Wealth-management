import asyncio
from langchain_core.messages import HumanMessage
from src.agent.graph import build_graph

def run():
    client_id = "C-301"
    rm_tier = "institutional"
    prompt = "Search for exact guidelines on maximum equity allocation for conservative clients."
    
    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "client_id": client_id,
        "rm_tier": rm_tier,
        "current_step": 0,
    }

    copilot_app = build_graph()
    
    print("STARTING STREAM...")
    for event in copilot_app.stream(initial_state):
        for node_name, state_update in event.items():
            if node_name == "agent":
                msg = state_update["messages"][-1]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print("TOOL CALLS:", msg.tool_calls)
                elif hasattr(msg, "content") and msg.content:
                    print("AI MSG:", msg.content)

if __name__ == "__main__":
    run()
