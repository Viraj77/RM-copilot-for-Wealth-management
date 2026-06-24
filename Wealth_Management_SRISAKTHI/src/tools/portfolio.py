"""Portfolio lookup tool."""

import json
from pathlib import Path

from langchain_core.tools import tool

from src.config import CLIENTS_DIR
from src.models import ClientPortfolio, PortfolioHolding, RiskProfile, Sensitivity


def _load_client(client_id: str) -> ClientPortfolio:
    filepath = CLIENTS_DIR / f"{client_id}.json"
    if not filepath.exists():
        raise ValueError(f"Client {client_id} not found")
    data = json.loads(filepath.read_text(encoding="utf-8"))
    holdings = [PortfolioHolding(**h) for h in data["holdings"]]
    return ClientPortfolio(
        client_id=data["client_id"],
        client_name=data["client_name"],
        risk_profile=RiskProfile(data["risk_profile"]),
        total_value_usd=data["total_value_usd"],
        holdings=holdings,
        rm_id=data.get("rm_id", "RM-001"),
        rm_entitlements=[Sensitivity(e) for e in data.get("rm_entitlements", ["public", "internal"])],
    )


def get_portfolio_summary(client: ClientPortfolio) -> str:
    """Generate a human-readable portfolio summary."""
    lines = [
        f"Client: {client.client_name} ({client.client_id})",
        f"Risk Profile: {client.risk_profile.value}",
        f"Total Portfolio Value: ${client.total_value_usd:,.0f}",
        "",
        "Holdings:",
    ]
    allocation: dict[str, float] = {}
    for h in client.holdings:
        lines.append(
            f"  - {h.symbol} ({h.name}): {h.weight_pct:.1f}% "
            f"[{h.asset_class}] ${h.value_usd:,.0f}"
        )
        allocation[h.asset_class] = allocation.get(h.asset_class, 0) + h.weight_pct

    lines.append("")
    lines.append("Asset Allocation:")
    for asset_class, pct in sorted(allocation.items(), key=lambda x: -x[1]):
        lines.append(f"  - {asset_class}: {pct:.1f}%")

    equity_pct = allocation.get("Equity", 0) + allocation.get("Real Estate", 0)
    lines.append(f"\nEquity + Real Estate Exposure: {equity_pct:.1f}%")
    return "\n".join(lines)


@tool
def portfolio_lookup(client_id: str) -> str:
    """Fetch client portfolio holdings, allocation, and risk profile by client ID."""
    client = _load_client(client_id)
    return get_portfolio_summary(client)


def load_client_portfolio(client_id: str) -> ClientPortfolio:
    """Programmatic access to client portfolio data."""
    return _load_client(client_id)
