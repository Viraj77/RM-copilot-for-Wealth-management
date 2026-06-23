"""
Market Data Tool — Phase 3: Tools.

Retrieves simulated market data (price, returns, volatility) for
investment instruments. Uses a local JSON data store for reproducibility.
"""

import json
from pathlib import Path
from typing import Any, Optional

from config.settings import settings

# ── Valid periods ─────────────────────────────────────────────────────────────

VALID_PERIODS = {"1W", "1M", "3M", "6M", "1Y"}

# ── Market data cache ─────────────────────────────────────────────────────────

_MARKET_CACHE: dict[str, Any] = {}


def _load_market_db() -> dict[str, Any]:
    """Load and cache market data from JSON file."""
    global _MARKET_CACHE
    if _MARKET_CACHE:
        return _MARKET_CACHE

    path = settings.market_data_file
    if not path.exists():
        raise FileNotFoundError(f"Market data file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    _MARKET_CACHE = data.get("market_data", {})
    return _MARKET_CACHE


# ── Core market data function ─────────────────────────────────────────────────

def market_data(ticker: str, period: str = "1M") -> dict[str, Any]:
    """
    Retrieve market data for a given instrument ticker.

    Args:
        ticker: Instrument ticker (e.g. "VTI", "AGG", "QQQ").
        period: Lookback period — one of "1W", "1M", "3M", "6M", "1Y".

    Returns:
        Dict with current_price, period_return, volatility, 52w_high,
        52w_low, asset_class, sector, expense_ratio, dividend_yield.
    """
    ticker = ticker.strip().upper()
    period = period.strip().upper()

    if period not in VALID_PERIODS:
        return {
            "error": f"Invalid period '{period}'. Must be one of: {sorted(VALID_PERIODS)}",
            "success": False,
        }

    db = _load_market_db()

    if ticker not in db:
        available = sorted(db.keys())
        return {
            "error": f"Ticker '{ticker}' not found in market data.",
            "available_tickers": available,
            "success": False,
            "note": "This is a simulated market data store. Only known holdings are available.",
        }

    raw = db[ticker]
    returns = raw.get("returns", {})

    result: dict[str, Any] = {
        "success": True,
        "ticker": ticker,
        "name": raw.get("name", ticker),
        "current_price_usd": raw.get("current_price"),
        "period": period,
        "period_return_pct": returns.get(period),
        "volatility_annualised_pct": raw.get("volatility_annualised"),
        "52w_high": raw.get("52w_high"),
        "52w_low": raw.get("52w_low"),
        "asset_class": raw.get("asset_class"),
        "sector": raw.get("sector"),
        "expense_ratio_pct": raw.get("expense_ratio"),
        "dividend_yield_pct": raw.get("dividend_yield"),
        "all_period_returns": returns,
    }

    # Attach instrument-specific fields
    if "duration_years" in raw:
        result["duration_years"] = raw["duration_years"]
    if "credit_quality" in raw:
        result["credit_quality"] = raw["credit_quality"]
    if "inflation_linked" in raw:
        result["inflation_linked"] = raw["inflation_linked"]
    if "note" in raw:
        result["note"] = raw["note"]

    return result


def market_data_bulk(tickers: list[str], period: str = "1M") -> dict[str, Any]:
    """
    Retrieve market data for multiple tickers at once.

    Args:
        tickers: List of ticker symbols.
        period: Lookback period for all tickers.

    Returns:
        Dict mapping ticker → market_data result.
    """
    results = {}
    for ticker in tickers:
        results[ticker] = market_data(ticker, period)
    return results


def get_available_tickers() -> list[str]:
    """Return all tickers in the market data store."""
    db = _load_market_db()
    return sorted(db.keys())


# ── LangChain Tool wrapper ────────────────────────────────────────────────────

def get_market_data_tool():
    """Return a LangChain-compatible tool for market data retrieval."""
    try:
        from langchain_core.tools import tool  # type: ignore

        @tool
        def market_data_tool(ticker: str, period: str = "1M") -> str:
            """
            Retrieve current market data for an investment instrument.

            Use this tool to get price, returns, volatility, and other market
            metrics for a specific ticker or fund. Useful for understanding
            current market performance of client holdings.

            Args:
                ticker: Instrument ticker symbol (e.g. 'VTI', 'AGG', 'QQQ', 'TIPS').
                period: Return period — one of '1W', '1M', '3M', '6M', '1Y'.

            Returns:
                JSON string with market data including price, returns, and volatility.
            """
            result = market_data(ticker, period)
            return json.dumps(result, indent=2, default=str)

        return market_data_tool

    except ImportError:
        raise ImportError(
            "langchain-core is required. Install with: pip install langchain-core"
        )
