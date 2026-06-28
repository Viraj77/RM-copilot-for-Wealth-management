import streamlit as st
import pandas as pd
from src.tools.portfolio_tool import portfolio_lookup
from app.components.sentiment_view import render_sentiment_view

def render_client_info_view(client_id: str, rm_tier: str, embedded: bool = False):
    if not embedded:
        st.title("👤 Client Information")
    
    # Fetch client data
    client_data = portfolio_lookup(client_id, rm_tier)
    
    if not client_data.get("success"):
        st.error(f"🔒 Access Restricted: {client_data.get('error', 'Unknown Error')}")
        st.info("Please adjust your Entitlement Tier in the sidebar to view this client's details.")
        return

    # Header section
    st.markdown(f"## {client_data.get('client_name', 'Unknown Client')} ({client_id})")
    st.divider()
    
    # Create Tabs
    tab1, tab2 = st.tabs(["📊 Portfolio & Details", "🧠 Sentiment & Mood"])
    
    with tab1:
        # Top-level metrics
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        aum = client_data.get("total_aum_usd", 0)
        aum_str = f"${aum / 1_000_000:.1f}M" if aum >= 1_000_000 else f"${aum:,.0f}"
        
        m1.metric(label="Total AUM", value=aum_str)
        m2.metric(label="Risk Profile", value=client_data.get("risk_profile", "Unknown").title())
        m3.metric(label="Investment Horizon", value=client_data.get("investment_horizon", "Unknown"))
        m4.metric(label="Client Tier", value=client_data.get("entitlement_tier", "Unknown").title())
        
        # Notes / Overview
        st.markdown("### Overview")
        notes = client_data.get("rm_notes")
        st.info(notes if notes else "No notes available for this client.")
        
        # Holdings Table
        st.markdown("### Portfolio Holdings")
        holdings = client_data.get("holdings", [])
        
        if not holdings:
            st.warning("No holdings found for this client.")
        else:
            # Convert to DataFrame for nice rendering
            df = pd.DataFrame(holdings)
            
            # Format the dataframe
            if not df.empty:
                # Reorder and rename columns for display
                display_df = df.copy()
                
                # Formatting functions
                def format_currency(x):
                    return f"${x:,.2f}"
                    
                def format_pct(x):
                    return f"{x:.2f}%"
                    
                # Apply formatting
                display_df['allocation_pct'] = display_df['allocation_pct'].apply(format_pct)
                display_df['current_value_usd'] = display_df['current_value_usd'].apply(format_currency)
                
                # Styling functions
                def highlight_negative_returns(val):
                    if isinstance(val, (int, float)) and val < 0:
                        return 'color: #ff4b4b;'
                    return ''
                    
                def highlight_underperforming_funds(row):
                    is_negative = row['gain_loss_pct'] < 0
                    return ['color: #ff4b4b;' if is_negative and col in ['ticker', 'name'] else '' for col in row.index]
        
                styled_df = display_df.style.apply(highlight_underperforming_funds, axis=1)
                styled_df = styled_df.map(highlight_negative_returns, subset=["gain_loss_pct"])
                        
                # Display as an interactive dataframe
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "ticker": st.column_config.TextColumn("Ticker"),
                        "name": st.column_config.TextColumn("Name"),
                        "asset_class": st.column_config.TextColumn("Asset Class"),
                        "allocation_pct": st.column_config.TextColumn("Allocation"),
                        "current_value_usd": st.column_config.TextColumn("Current Value"),
                        "gain_loss_pct": st.column_config.NumberColumn(
                            "Gain/Loss (%)",
                            format="%.2f%%"
                        ),
                        "sector": st.column_config.TextColumn("Sector"),
                        "currency": st.column_config.TextColumn("Currency"),
                    }
                )

    with tab2:
        render_sentiment_view(client_id, rm_tier, embedded=True)
