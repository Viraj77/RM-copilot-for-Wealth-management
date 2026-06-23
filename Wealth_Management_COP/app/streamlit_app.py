"""
Main Streamlit Application — Phase 6: Frontend.

Wires together the LangGraph agent, tools, guardrails, and UI components.
"""

import os
import sys
from pathlib import Path

# Ensure the src module can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# FORCE RELOAD CACHED MODULES
import sys
for key in list(sys.modules.keys()):
    if key.startswith("src.") or key.startswith("config."):
        del sys.modules[key]

from src.agent.graph import build_graph
from app.components.client_selector import render_sidebar_selector
from app.components.brief_renderer import render_brief
from app.components.escalation_view import render_tool_calls, render_escalation_panel


# ── Configuration & Setup ─────────────────────────────────────────────────────

load_dotenv()

st.set_page_config(
    page_title="Wealth Manager Copilot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply some custom CSS for that "wow" factor
st.markdown("""
<style>
    /* Sleek gradient header */
    .css-10trblm {
        color: #ffffff;
    }
    .stApp > header {
        background-color: transparent;
    }
    /* Dynamic button hover */
    .stButton>button {
        transition: all 0.2s ease-in-out;
        border-radius: 8px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    /* Clean expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 8px;
    }
    /* Beautiful alert boxes */
    div[data-testid="stAlert"] {
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("💼 Wealth Manager Copilot")
st.markdown(
    "Your AI assistant for portfolio analysis, policy retrieval, and compliance-checked recommendations."
)

# Render Sidebar
client_id, rm_tier = render_sidebar_selector()

# Chat Interface
st.subheader("Query the Copilot")

# Initialise chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input(f"Ask about {client_id} (e.g. 'How is their portfolio allocated?')"):
    
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run the Agent Graph
    with st.chat_message("assistant"):
        st.markdown("*Thinking and using tools...*")
        
        # We use a status container to show tool calls dynamically
        status_container = st.empty()
        
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "client_id": client_id,
            "rm_tier": rm_tier,
            "current_step": 0,
        }

        final_response_text = ""
        agent_tool_calls = []

        try:
            with open("streamlit_trace.log", "a") as f:
                f.write(f"USER SUBMITTED PROMPT: {prompt}\n")

            # Stream the LangGraph execution
            copilot_app = build_graph()
            for event in copilot_app.stream(initial_state):
                with open("streamlit_trace.log", "a") as f:
                    f.write(f"EVENT: {event.keys()}\n")
                for node_name, state_update in event.items():
                    if node_name == "agent":
                        msg = state_update["messages"][-1]
                        
                        # Catch tool calls for UI rendering
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            agent_tool_calls.extend(msg.tool_calls)
                            with status_container.container():
                                render_tool_calls(agent_tool_calls)
                        
                        # Catch the final response
                        elif hasattr(msg, "content") and msg.content:
                            final_response_text = msg.content
                            
        except Exception as e:
            st.error(f"Agent execution failed: {str(e)}")
            st.stop()

        status_container.empty() # Clear the "Thinking" placeholder
        
        # 3. Guardrails & Rendering
        if final_response_text:
            render_brief(final_response_text, rm_tier)
            st.session_state.messages.append({"role": "assistant", "content": final_response_text})
