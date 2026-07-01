"""Market data tool (mock data for demo)."""

from langchain_core.tools import tool

MARKET_SNAPSHOT = {
    "SPY": {"price": 582.40, "change_pct": 0.32, "ytd_return": 4.2},
    "AGG": {"price": 98.15, "change_pct": -0.08, "ytd_return": 1.1},
    "VTI": {"price": 285.60, "change_pct": 0.45, "ytd_return": 4.8},
    "VNQ": {"price": 88.30, "change_pct": -0.15, "ytd_return": -1.2},
    "US-TBILL": {"price": 100.0, "change_pct": 0.01, "ytd_return": 4.5},
    "BND": {"price": 72.50, "change_pct": 0.02, "ytd_return": 1.5},
    "GLD": {"price": 242.10, "change_pct": 0.55, "ytd_return": 6.1},
    # Horizon product guides (PG-002 / PG-003)
    "PG-002": {"price": 11.42, "change_pct": 0.05, "ytd_return": 2.8},
    "HCIF": {"price": 11.42, "change_pct": 0.05, "ytd_return": 2.8},
    "PG-003": {"price": 24.85, "change_pct": 0.62, "ytd_return": 8.5},
    "HAEF": {"price": 24.85, "change_pct": 0.62, "ytd_return": 8.5},
}

MACRO_CONTEXT = """
Q2 2026 Macro Snapshot (RN-001):
- Fed Funds Rate: 4.25-4.50% (hold expected through Q2)
- 10Y Treasury Yield: 4.05%
- S&P 500 YTD: +4.2%
- VIX: 16.8 (moderate)
- Key theme: Soft landing base case; conservative income (PG-002/HCIF) favored for
  capital preservation; growth equity (PG-003/HAEF) for aggressive satellite allocations
"""


@tool
def market_data(symbols: str = "SPY,AGG,VTI") -> str:
    """Get current market data and macro context for given comma-separated symbols."""
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    lines = ["Market Data Snapshot:", ""]
    for sym in symbol_list:
        if sym in MARKET_SNAPSHOT:
            d = MARKET_SNAPSHOT[sym]
            lines.append(
                f"  {sym}: ${d['price']:.2f} ({d['change_pct']:+.2f}% today, "
                f"YTD {d['ytd_return']:+.1f}%)"
            )
        else:
            lines.append(f"  {sym}: data not available")
    lines.append("")
    lines.append(MACRO_CONTEXT.strip())
    return "\n".join(lines)
