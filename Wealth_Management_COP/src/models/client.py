"""
Client data models — ClientProfile, PortfolioHolding, and related enums.
"""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class RiskProfile(str, Enum):
    """Client risk tolerance levels (conservative → aggressive)."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    GROWTH = "growth"
    AGGRESSIVE = "aggressive"


class AssetClass(str, Enum):
    """Major asset class categories."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    ALTERNATIVES = "alternatives"
    CASH = "cash"
    REAL_ESTATE = "real_estate"
    COMMODITY = "commodity"


class EntitlementTier(str, Enum):
    """RM entitlement tier — controls which knowledge documents are accessible."""
    STANDARD = "standard"
    PREMIUM = "premium"
    INSTITUTIONAL = "institutional"


# ── Portfolio Holding ─────────────────────────────────────────────────────────

class PortfolioHolding(BaseModel):
    """A single instrument/holding within a client's portfolio."""

    ticker: str = Field(..., description="Instrument ticker or fund ID (e.g. 'SPY', 'VBMFX')")
    name: str = Field(..., description="Human-readable instrument name")
    asset_class: AssetClass
    allocation_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Percentage of total portfolio value"
    )
    current_value: float = Field(..., ge=0.0, description="Current market value in USD")
    gain_loss_pct: float = Field(..., description="Unrealized P&L percentage")
    sector: Optional[str] = Field(None, description="Sector classification (for equities)")
    currency: str = Field(default="USD")

    model_config = {"str_strip_whitespace": True}


# ── Client Profile ────────────────────────────────────────────────────────────

class ClientProfile(BaseModel):
    """
    Full client profile including risk preference, investment horizon, and holdings.
    Used by the portfolio tool and agent state.
    """

    client_id: str = Field(
        ...,
        pattern=r"^C-\d{3,}$",
        description="Client identifier in format C-NNN (e.g. 'C-204')"
    )
    name: str = Field(..., description="Client full name")
    risk_profile: RiskProfile
    investment_horizon: str = Field(
        ...,
        description="Investment horizon description (e.g. '5-10 years')"
    )
    total_aum: float = Field(..., ge=0.0, description="Total Assets Under Management in USD")
    holdings: list[PortfolioHolding] = Field(default_factory=list)
    last_review_date: Optional[date] = Field(None, description="Date of last RM review meeting")
    rm_id: Optional[str] = Field(None, description="Assigned Relationship Manager ID")
    entitlement_tier: EntitlementTier = Field(
        default=EntitlementTier.STANDARD,
        description="Client-level data entitlement (controls RM's access to client data)"
    )
    notes: Optional[str] = Field(None, description="Free-text RM notes on client")

    @field_validator("holdings")
    @classmethod
    def validate_allocations(cls, holdings: list[PortfolioHolding]) -> list[PortfolioHolding]:
        """Warn if total allocation deviates significantly from 100%."""
        if holdings:
            total = sum(h.allocation_pct for h in holdings)
            if abs(total - 100.0) > 2.0:  # Allow ±2% rounding tolerance
                raise ValueError(
                    f"Holdings allocations sum to {total:.1f}%, expected ~100%"
                )
        return holdings

    @property
    def allocation_by_asset_class(self) -> dict[str, float]:
        """Aggregate allocation percentage grouped by asset class."""
        result: dict[str, float] = {}
        for h in self.holdings:
            key = h.asset_class.value
            result[key] = result.get(key, 0.0) + h.allocation_pct
        return result

    model_config = {"str_strip_whitespace": True}


# ── RM Profile ────────────────────────────────────────────────────────────────

class RMProfile(BaseModel):
    """Relationship Manager identity — used for entitlement checks."""
    rm_id: str = Field(..., description="RM employee ID")
    name: str
    entitlement_tier: EntitlementTier = Field(
        default=EntitlementTier.STANDARD,
        description="RM's knowledge access tier"
    )
    licensed: bool = Field(
        default=False,
        description="Whether the RM holds a financial advice licence"
    )
