import streamlit as st

# pyrefly: ignore [missing-import]
from src.models.client import EntitlementTier
from src.tools.portfolio_tool import get_all_client_ids, get_client_display_names


@st.dialog("⚠️ Change Active Client?")
def confirm_client_change():
    st.write(f"Switching to **{st.session_state.pending_client}**")
    
    has_history = len(st.session_state.get("messages", [])) > 0
    if has_history:
        st.error("This will clear your current chat history. Proceed?")
    
    c1, c2 = st.columns(2)
    if c1.button("Yes", type="primary", use_container_width=True):
        st.session_state.confirmed_client = st.session_state.pending_client
        if has_history:
            st.session_state.messages = []  # Clear history
        st.rerun()
    if c2.button("No", use_container_width=True):
        st.session_state.pending_client = st.session_state.confirmed_client
        st.session_state.client_dropdown = st.session_state.confirmed_client
        st.rerun()

def render_sidebar_selector() -> tuple[str, str]:
    """
    Renders the sidebar for selecting the RM's entitlement tier and the active client.
    
    Returns:
        (client_id, rm_tier)
    """
    # Centered Premium Avatar (Adjusted size)
    col1, col2, col3 = st.sidebar.columns([3, 4, 3])
    with col2:
        # Use the newly generated local asset
        st.image("app/assets/copilot_avatar.png", use_container_width=True)
    
    st.sidebar.markdown(
        "<h4 style='text-align: center; margin-top: -15px; margin-bottom: 10px;'>AI Wealth Manager</h4>", 
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

    # RM Entitlement Tier
    st.sidebar.markdown("#### 🛡️ Entitlement Tier")
    rm_tier = st.sidebar.selectbox(
        "Select your access level:",
        options=[tier.value for tier in EntitlementTier],
        index=0,
        help="Determines which clients and knowledge documents you can access."
    )

    # Client Selection
    st.sidebar.markdown("#### 👤 Active Client")
    client_ids = get_all_client_ids()
    display_names = get_client_display_names()
    options = [display_names[cid] for cid in client_ids]

    if "confirmed_client" not in st.session_state:
        st.session_state.confirmed_client = options[0]
        st.session_state.pending_client = options[0]

    def on_client_change():
        st.session_state.pending_client = st.session_state.client_dropdown

    st.sidebar.selectbox(
        "Select your active client:",
        options=options,
        key="client_dropdown",
        on_change=on_client_change,
        help="The client you are currently assisting or reviewing."
    )

    # Check if a client change was requested
    if st.session_state.pending_client != st.session_state.confirmed_client:
        # Show the true modal popup
        confirm_client_change()

    # Extract the actual client_id (the part before the em dash) from the confirmed client
    # E.g., "C-204 — Sarah Chen" -> "C-204"
    client_id = st.session_state.confirmed_client.split(" — ")[0]

    return client_id, rm_tier
