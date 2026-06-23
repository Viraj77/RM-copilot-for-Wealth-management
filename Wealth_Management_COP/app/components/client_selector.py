import streamlit as st

from src.models.client import EntitlementTier
from src.tools.portfolio_tool import get_all_client_ids, get_client_display_names


def render_sidebar_selector() -> tuple[str, str]:
    """
    Renders the sidebar for selecting the RM's entitlement tier and the active client.
    
    Returns:
        (client_id, rm_tier)
    """
    st.sidebar.title("⚙️ Copilot Settings")
    st.sidebar.markdown("---")

    # RM Entitlement Tier
    st.sidebar.subheader("Your Entitlement Tier")
    rm_tier = st.sidebar.selectbox(
        "Select Tier",
        options=[tier.value for tier in EntitlementTier],
        index=0,
        help="Determines which clients and knowledge documents you can access."
    )

    # Client Selection
    st.sidebar.subheader("Active Client")
    client_ids = get_all_client_ids()
    display_names = get_client_display_names()

    # Create a reverse mapping to get ID from display name
    # e.g., "C-204 — Sarah Chen (Balanced)" -> "C-204"
    options = [display_names[cid] for cid in client_ids]
    
    selected_display = st.sidebar.selectbox(
        "Select Client",
        options=options,
        index=0,
        help="The client you are currently assisting or reviewing."
    )
    
    # Extract the actual client_id (the part before the em dash)
    # E.g., "C-204 — Sarah Chen" -> "C-204"
    client_id = selected_display.split(" — ")[0]

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Tip**: Try changing your tier to 'standard' while viewing a 'premium' client "
        "to test the Copilot's entitlement guardrails."
    )

    return client_id, rm_tier
