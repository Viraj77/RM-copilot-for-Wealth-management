# Mock Client Database for Wealth Management Copilot

CLIENT_DB = {
    "C-204": {
        "client_id": "C-204",
        "name": "Eleanor Vance",
        "risk_profile": "Balanced",
        "rm_name": "John Doe",
        "rm_research_tier": 2,  # Entitled to Restricted (Tier 2) research like RN-002
        "rm_access_to_private": True,
        "holdings": [
            {
                "product_code": "PG-001",
                "product_name": "Horizon Balanced Growth Fund (HBGF)",
                "allocation_amount": 150000.0,
                "asset_class": "Mutual Fund - Hybrid"
            },
            {
                "product_code": "CASH",
                "product_name": "Cash Equivalents",
                "allocation_amount": 50000.0,
                "asset_class": "Cash"
            }
        ]
    },
    "C-101": {
        "client_id": "C-101",
        "name": "Arthur Pendleton",
        "risk_profile": "Conservative",
        "rm_name": "Jane Smith",
        "rm_research_tier": 1,  # Entitled to Public (Tier 1) research only
        "rm_access_to_private": False,
        "holdings": [
            {
                "product_code": "PG-002",
                "product_name": "Horizon Conservative Income Fund (HCIF)",
                "allocation_amount": 80000.0,
                "asset_class": "Mutual Fund - Hybrid"
            },
            {
                "product_code": "CASH",
                "product_name": "Cash Equivalents",
                "allocation_amount": 20000.0,
                "asset_class": "Cash"
            }
        ]
    },
    "C-302": {
        "client_id": "C-302",
        "name": "Marcus Vance",
        "risk_profile": "Aggressive",
        "rm_name": "John Doe",
        "rm_research_tier": 2,
        "rm_access_to_private": True,
        "holdings": [
            {
                "product_code": "PG-003",
                "product_name": "Horizon Aggressive Equity Fund (HAEF)",
                "allocation_amount": 300000.0,
                "asset_class": "Mutual Fund - Equity"
            },
            {
                "product_code": "CASH",
                "product_name": "Cash Equivalents",
                "allocation_amount": 50000.0,
                "asset_class": "Cash"
            }
        ]
    }
}

def get_client_profile(client_id: str):
    """
    Fetches the profile for a client ID.
    Returns dict or None.
    """
    return CLIENT_DB.get(client_id.upper())

def get_all_clients():
    return list(CLIENT_DB.values())
