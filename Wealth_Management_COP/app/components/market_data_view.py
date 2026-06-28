import streamlit as st
import pandas as pd
from src.ingestion.loader import load_market_data
from config.settings import settings

def render_market_data_view():
    st.title("📈 Market Data")
    
    try:
        data = load_market_data(settings.market_data_file)
    except Exception as e:
        st.error(f"Failed to load market data: {str(e)}")
        return
        
    metadata = data.get("metadata", {})
    market_data = data.get("market_data", {})
    
    if not market_data:
        st.warning("No market data available.")
        return
        
    st.markdown(f"**Last Updated:** {metadata.get('last_updated', 'Unknown')}")
    st.caption(metadata.get('disclaimer', ''))
    st.divider()
    
    # Parse market data into a tabular format
    rows = []
    for ticker, info in market_data.items():
        returns = info.get("returns", {})
        row = {
            "Ticker": ticker,
            "Name": info.get("name", ""),
            "Current Price": info.get("current_price", 0.0),
            "1W Return (%)": returns.get("1W", 0.0),
            "1M Return (%)": returns.get("1M", 0.0),
            "3M Return (%)": returns.get("3M", 0.0),
            "6M Return (%)": returns.get("6M", 0.0),
            "1Y Return (%)": returns.get("1Y", 0.0),
            "Volatility": info.get("volatility_annualised", 0.0),
            "Dividend Yield": info.get("dividend_yield", 0.0),
            "Asset Class": info.get("asset_class", "Unknown").replace("_", " ").title()
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # Styling functions
    def highlight_negative_returns(val):
        """Color negative numbers red."""
        if isinstance(val, (int, float)) and val < 0:
            return 'color: #ff4b4b;'
        return ''
        
    def highlight_underperforming_funds(row):
        """Color the Ticker and Name red if the 1Y Return is negative."""
        is_negative = row['1Y Return (%)'] < 0
        return ['color: #ff4b4b;' if is_negative and col in ['Ticker', 'Name'] else '' for col in row.index]

    return_cols = ["1W Return (%)", "1M Return (%)", "3M Return (%)", "6M Return (%)", "1Y Return (%)"]
    
    styled_df = df.style.apply(highlight_underperforming_funds, axis=1)
    styled_df = styled_df.map(highlight_negative_returns, subset=return_cols)
    
    # Interactive Dataframe
    st.markdown("### Global Market Overview")
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Name": st.column_config.TextColumn("Asset Name", width="medium"),
            "Current Price": st.column_config.NumberColumn(
                "Price (USD)",
                format="$%.2f"
            ),
            "1W Return (%)": st.column_config.NumberColumn("1W", format="%.2f%%"),
            "1M Return (%)": st.column_config.NumberColumn("1M", format="%.2f%%"),
            "3M Return (%)": st.column_config.NumberColumn("3M", format="%.2f%%"),
            "6M Return (%)": st.column_config.NumberColumn("6M", format="%.2f%%"),
            "1Y Return (%)": st.column_config.NumberColumn("1Y", format="%.2f%%"),
            "Volatility": st.column_config.NumberColumn("Volatility", format="%.1f%%"),
            "Dividend Yield": st.column_config.NumberColumn("Div Yield", format="%.2f%%"),
        }
    )
    
    st.info("Market data is synthetic and used for demonstration purposes within the Wealth Manager Copilot.")
