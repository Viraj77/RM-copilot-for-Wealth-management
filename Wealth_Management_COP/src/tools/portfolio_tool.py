"""
Portfolio Lookup Tool — Phase 3: Tools.

Fetches client portfolio holdings, risk profile, and allocation summary
from the local client database (data/clients/clients.json).
Used as a LangChain tool by the LangGraph agent.
"""

import json
from pathlib import Path
from typing import Any

from config.settings import settings
from src.models.client import ClientProfile, EntitlementTier


# ── Client database loader ────────────────────────────────────────────────────

_CLIENT_CACHE: dict[str, ClientProfile] = {}


def _load_client_db() -> dict[str, ClientProfile]:
    """Load and cache all clients from the JSON database."""
    global _CLIENT_CACHE
    if _CLIENT_CACHE:
        return _CLIENT_CACHE

    path = settings.client_data_file
    if not path.exists():
        raise FileNotFoundError(f"Client database not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for raw_client in data.get("clients", []):
        try:
            client = ClientProfile.model_validate(raw_client)
            _CLIENT_CACHE[client.client_id] = client
        except Exception as exc:
            print(f"  [WARN] Could not parse client '{raw_client.get('client_id')}': {exc}")

    return _CLIENT_CACHE


def _check_entitlement(client: ClientProfile, rm_tier: str) -> bool:
    """
    Check if the RM's tier allows access to this client's data.
    Simple rule: RM must have tier >= client's entitlement_tier.
    """
    tier_order = {
        EntitlementTier.STANDARD: 0,
        EntitlementTier.PREMIUM: 1,
        EntitlementTier.INSTITUTIONAL: 2,
    }
    rm_level = tier_order.get(EntitlementTier(rm_tier), 0)
    client_level = tier_order.get(client.entitlement_tier, 0)
    return rm_level >= client_level


# ── Core lookup function ──────────────────────────────────────────────────────

def portfolio_lookup(client_id: str, rm_tier: str = "standard") -> dict[str, Any]:
    """
    Fetch client portfolio holdings and risk profile.

    Args:
        client_id: Client identifier in format "C-NNN" (e.g. "C-204").
        rm_tier: RM's entitlement tier ("standard" | "premium" | "institutional").

    Returns:
        Dict with client_id, client_name, risk_profile, total_aum, investment_horizon,
        holdings, allocation_breakdown, last_review_date, and entitlement_tier.

    Raises:
        ValueError: If client_id not found or RM is not entitled to access the client.
    """
    client_id = client_id.strip().upper()

    db = _load_client_db()

    if client_id not in db:
        available = sorted(db.keys())
        return {
            "error": f"Client '{client_id}' not found.",
            "available_clients": available,
            "success": False,
        }

    client = db[client_id]

    # Entitlement check
    if not _check_entitlement(client, rm_tier):
        return {
            "error": (
                f"RM with tier '{rm_tier}' is not authorised to access "
                f"client '{client_id}' (requires '{client.entitlement_tier.value}' tier)."
            ),
            "success": False,
        }

    # Build allocation breakdown
    allocation = client.allocation_by_asset_class

    # Format holdings for the agent
    holdings_list = [
        {
            "ticker": h.ticker,
            "name": h.name,
            "asset_class": h.asset_class.value,
            "allocation_pct": h.allocation_pct,
            "current_value_usd": h.current_value,
            "gain_loss_pct": h.gain_loss_pct,
            "sector": h.sector,
        }
        for h in client.holdings
    ]

    return {
        "success": True,
        "client_id": client.client_id,
        "client_name": client.name,
        "risk_profile": client.risk_profile.value,
        "investment_horizon": client.investment_horizon,
        "total_aum_usd": client.total_aum,
        "holdings": holdings_list,
        "allocation_breakdown": allocation,
        "last_review_date": client.last_review_date.isoformat() if client.last_review_date else None,
        "entitlement_tier": client.entitlement_tier.value,
        "rm_notes": client.notes,
    }


# ── LangChain Tool wrapper ────────────────────────────────────────────────────

def get_portfolio_tool():
    """Return a LangChain-compatible tool for portfolio lookup."""
    try:
        from langchain_core.tools import tool  # type: ignore

        @tool
        def portfolio_lookup_tool(client_id: str, rm_tier: str = "standard") -> str:
            """
            Fetch a client's portfolio holdings and risk profile from the database.

            Use this tool when you need to retrieve information about a specific client's
            investments, risk profile, AUM, or allocation breakdown.

            Args:
                client_id: Client identifier in format 'C-NNN' (e.g. 'C-204').
                rm_tier: RM entitlement tier ('standard', 'premium', 'institutional').

            Returns:
                JSON string with client portfolio data.
            """
            result = portfolio_lookup(client_id, rm_tier)
            return json.dumps(result, indent=2, default=str)

        return portfolio_lookup_tool

    except ImportError:
        raise ImportError(
            "langchain-core is required for tool wrapping. "
            "Install with: pip install langchain-core"
        )


def get_all_client_ids() -> list[str]:
    """Return all available client IDs (for UI dropdowns)."""
    db = _load_client_db()
    return sorted(db.keys())


def get_client_display_names() -> dict[str, str]:
    """Return {client_id: display_name} mapping for UI."""
    db = _load_client_db()
    return {cid: f"{cid} — {c.name} ({c.risk_profile.value.title()})" for cid, c in db.items()}
