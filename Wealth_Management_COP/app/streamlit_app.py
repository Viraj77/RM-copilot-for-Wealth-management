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
    if key.startswith("src.") or key.startswith("config.") or key.startswith("app."):
        del sys.modules[key]

from src.agent.graph import build_graph
from app.components.client_selector import render_sidebar_selector
from app.components.brief_renderer import render_brief
from app.components.escalation_view import render_tool_calls, render_escalation_panel
from src.tools.portfolio_tool import portfolio_lookup


from app.components.upload_view import render_upload_view
from app.components.client_info_view import render_client_info_view
from app.components.market_data_view import render_market_data_view
from app.components.evaluation_view import render_evaluation_view


# ── Configuration & Setup ─────────────────────────────────────────────────────

load_dotenv()

st.set_page_config(
    page_title="Wealth Manager Copilot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Remove the huge empty space at the bottom of the page
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 1rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# Apply some custom CSS for that "wow" factor
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dynamic button hover */
    .stButton>button {
        transition: all 0.3s ease;
        border-radius: 8px;
        background-color: #00adb5;
        color: #ffffff;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 173, 181, 0.4);
        background-color: #008c93;
        color: #ffffff;
    }
    
    /* Glassmorphism sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(30, 30, 30, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Clean expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 8px;
        background-color: rgba(255,255,255,0.02);
    }
    
    /* Beautiful alert boxes */
    div[data-testid="stAlert"] {
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        background-color: rgba(40,40,40,0.8);
    }
    
    /* Chat bubbles basic animation */
    .stChatMessage {
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        animation: fadeIn 0.4s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# ── Main UI ───────────────────────────────────────────────────────────────────

# Render Sidebar (Always visible so client selection persists)
client_id, rm_tier = render_sidebar_selector()

# Navigation
st.sidebar.markdown("### 🧭 Navigation")
nav_selection = st.sidebar.radio("Go to", [
    "📊 Client Dashboard", 
    "📈 Market Data", 
    "📂 Knowledge Base Upload",
    "🧪 RAGAS Evaluation"
])
st.sidebar.divider()

st.sidebar.info(
    "**Tip**: Try changing your tier to 'standard' while viewing a 'premium' client "
    "to test the Copilot's entitlement guardrails."
)

if nav_selection == "📂 Knowledge Base Upload":
    render_upload_view()
elif nav_selection == "📈 Market Data":
    render_market_data_view()
elif nav_selection == "🧪 RAGAS Evaluation":
    render_evaluation_view()
else:
    st.title("💼 AI Wealth Manager Dashboard")
    st.markdown(
        "Your unified command center for portfolio analysis, client sentiment, and AI assistance."
    )
    st.divider()
    
    col_left, col_right = st.columns([1.1, 0.9], gap="large")
    
    with col_left:
        # Render Client Information on the left in a scrollable container
        info_container = st.container(height=735)
        with info_container:
            render_client_info_view(client_id, rm_tier, embedded=True)
        
    with col_right:
        # Chat Interface
        st.subheader("💬 Copilot Assistant")
    
        # Initialise chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
        # Display chat history inside a fixed-height container
        chat_container = st.container(height=600)
        
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        render_brief(msg["content"], rm_tier)
                    else:
                        st.markdown(msg["content"])
    
        # Chat input
        if prompt := st.chat_input(f"Ask about {client_id} (e.g. 'How is their portfolio allocated?')"):
        
            # 1. Display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
    
            # 2. Run the Agent Graph
            with chat_container:
                with st.chat_message("assistant"):
                    # We use a status container to show tool calls dynamically
                    status_container = st.empty()
            
                with st.spinner("🧠 Copilot is analyzing your request..."):
                    langchain_messages = []
                    for m in st.session_state.messages:
                        if m["role"] == "user":
                            langchain_messages.append(HumanMessage(content=m["content"]))
                        elif m["role"] == "assistant":
                            langchain_messages.append(AIMessage(content=m["content"]))
                        
                    initial_state = {
                        "messages": langchain_messages,
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
