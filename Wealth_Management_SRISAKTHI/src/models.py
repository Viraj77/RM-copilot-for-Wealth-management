"""Pydantic data models for the RM Copilot."""

from enum import Enum

from pydantic import BaseModel, Field


class RiskProfile(str, Enum):
    CONSERVATIVE = "Conservative"
    BALANCED = "Balanced"
    GROWTH = "Growth"
    AGGRESSIVE = "Aggressive"


class ComplianceStatus(str, Enum):
    CLEARED = "Cleared"
    NEEDS_REVIEW = "Needs Review"
    BLOCKED = "Blocked"


class DocumentType(str, Enum):
    PRODUCT = "product"
    POLICY = "policy"
    RESEARCH = "research"


class Sensitivity(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


class Reco(BaseModel):
    idea: str = Field(..., description="Recommended action or product idea")
    rationale: str = Field(..., description="Grounded rationale for the recommendation")
    suitability: str = Field(..., description="Suitability assessment for the client")
    citations: list[str] = Field(
        default_factory=list,
        description="Source citations supporting this recommendation",
    )


class ClientBrief(BaseModel):
    client_id: str
    risk_profile: RiskProfile
    portfolio_summary: str
    recommendations: list[Reco] = Field(default_factory=list)
    compliance_status: ComplianceStatus
    talking_points: list[str] = Field(default_factory=list)
    escalation_reason: str | None = Field(
        default=None,
        description="Reason for human-in-the-loop escalation, if any",
    )
    disclaimer: str = Field(
        default=(
            "This brief is decision support for relationship managers only. "
            "It does not constitute personalized investment advice or an "
            "automated recommendation. All suggestions require RM review and "
            "appropriate licensing before client communication."
        )
    )


class PortfolioHolding(BaseModel):
    symbol: str
    name: str
    asset_class: str
    weight_pct: float
    value_usd: float


class ClientPortfolio(BaseModel):
    client_id: str
    client_name: str
    risk_profile: RiskProfile
    total_value_usd: float
    holdings: list[PortfolioHolding]
    rm_id: str = "RM-001"
    rm_entitlements: list[Sensitivity] = Field(
        default_factory=lambda: [Sensitivity.PUBLIC, Sensitivity.INTERNAL]
    )


class SuitabilityResult(BaseModel):
    suitable: bool
    fund_or_product: str
    client_risk_profile: RiskProfile
    assessment: str
    citations: list[str] = Field(default_factory=list)
    requires_licensed_advice: bool = False
    blocked_reason: str | None = None


class RetrievedChunk(BaseModel):
    doc_id: str
    content: str
    doc_type: DocumentType
    source: str
    date: str
    sensitivity: Sensitivity
    score: float = 0.0
