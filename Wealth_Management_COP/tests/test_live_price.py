"""
Unit tests for the Live Pricing Tool.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from src.tools.live_price_tool import get_live_price_data, get_live_price_tool

class TestLivePriceTool:
    @patch("yfinance.Ticker")
    def test_live_price_success(self, mock_ticker):
        import pandas as pd
        
        # Mock yfinance response
        mock_instance = MagicMock()
        
        # Setup mock pandas dataframe row for 'Close', 'High', 'Low', 'Volume'
        mock_hist = pd.DataFrame({
            "Close": [150.0],
            "High": [155.0],
            "Low": [149.0],
            "Volume": [1000000]
        })
        
        mock_instance.history.return_value = mock_hist
        mock_instance.fast_info = {"currency": "USD"}
        mock_ticker.return_value = mock_instance
        
        result = get_live_price_data("AAPL")
        
        assert result["success"] is True
        assert result["ticker"] == "AAPL"
        assert result["live_price_usd"] == 150.0
        assert result["today_high_usd"] == 155.0
        assert result["today_low_usd"] == 149.0
        assert result["volume"] == 1000000
        assert result["currency"] == "USD"

    @patch("yfinance.Ticker")
    def test_live_price_empty_history(self, mock_ticker):
        import pandas as pd
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_instance
        
        result = get_live_price_data("INVALID_TICKER")
        
        assert result["success"] is False
        assert "No live data found" in result["error"]

    def test_live_price_tool_wrapper(self):
        tool = get_live_price_tool()
        assert tool is not None
        assert tool.name == "live_price_tool"
