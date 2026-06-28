import json
from typing import Any

def get_live_price_data(ticker: str) -> dict[str, Any]:
    """
    Fetch live market data for a given ticker using yfinance.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {
            "success": False,
            "error": "yfinance library is not installed. Please run 'pip install yfinance'."
        }

    ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(ticker)
        # Fetch data for the last 1 day to get current price, high, low, volume
        hist = stock.history(period="1d")
        
        if hist.empty:
            return {
                "success": False,
                "error": f"No live data found for ticker '{ticker}'."
            }
            
        current_price = hist['Close'].iloc[-1]
        high_price = hist['High'].iloc[-1]
        low_price = hist['Low'].iloc[-1]
        volume = hist['Volume'].iloc[-1]
        
        # Optionally get some fast info if available
        info = {}
        try:
            info = stock.fast_info
        except:
            pass
            
        return {
            "success": True,
            "ticker": ticker,
            "live_price_usd": float(current_price),
            "today_high_usd": float(high_price),
            "today_low_usd": float(low_price),
            "volume": int(volume),
            "currency": info.get("currency", "USD"),
            "market_status": "Data fetched successfully from live market"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to fetch live data for '{ticker}': {str(e)}"
        }

def get_live_price_tool():
    """Return a LangChain-compatible tool for fetching live market prices."""
    try:
        from langchain_core.tools import tool
        
        @tool
        def live_price_tool(ticker: str) -> str:
            """
            Fetch LIVE, real-time market data for a given stock or ETF ticker.
            
            Use this tool specifically when the user asks for the "current", "live", or "latest" 
            price of a specific ticker. For historical returns or volatility, use the standard 
            market_data_tool instead.
            
            Args:
                ticker: Instrument ticker symbol (e.g. 'AAPL', 'VTI', 'MSFT').
                
            Returns:
                JSON string with the live market data.
            """
            result = get_live_price_data(ticker)
            return json.dumps(result, indent=2, default=str)
            
        return live_price_tool
    except ImportError:
        pass
