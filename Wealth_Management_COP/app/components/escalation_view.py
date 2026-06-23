import streamlit as st
import json

def render_tool_calls(tool_calls: list) -> None:
    """
    Renders an expandable view of the tools the agent is using.
    """
    for call in tool_calls:
        tool_name = call.get("name", "unknown_tool")
        args = call.get("args", {})
        
        with st.expander(f"🛠️ Agent called tool: **{tool_name}**"):
            st.code(json.dumps(args, indent=2), language="json")


def render_escalation_panel(violations: list[str]) -> None:
    """
    Renders an alert panel if the suitability checker flagged violations.
    """
    if not violations:
        return
        
    st.error("🚨 **Compliance Escalation Required**")
    st.markdown(
        "The Copilot caught the following policy violations during its suitability checks. "
        "These recommendations were blocked from the final brief:"
    )
    for v in violations:
        st.markdown(f"- ❌ {v}")
