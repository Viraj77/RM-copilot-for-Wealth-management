"""
ClientBrief and related output models — Recommendation, Citation, ComplianceStatus.
These are the structured outputs produced by the LangGraph synthesis node.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from src.models.client import RiskProfile


# ── Enums ─────────────────────────────────────────────────────────────────────

class ComplianceStatus(str, Enum):
    """Overall compliance gate result for a ClientBrief."""
    CLEARED = "cleared"
    NEEDS_REVIEW = "needs_review"
    BLOCKED = "blocked"


class SuitabilityVerdict(str, Enum):
    """
    Suitability assessment verdict for a single recommendation.
    Drives compliance gate logic.
    """
    SUITABLE = "suitable"
    MARGINAL = "marginal"
    UNSUITABLE = "unsuitable"
    REQUIRES_LICENSED_ADVICE = "requires_licensed_advice"


# ── Citation ──────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """
    A reference to a specific knowledge chunk that grounds a recommendation.
    Every Recommendation must carry at least one Citation.
    """
    doc_id: str = Field(..., description="Source document ID from vector store")
    chunk_id: str = Field(..., description="Specific chunk identifier")
    doc_type: str = Field(..., description="'product' | 'policy' | 'research'")
    source: str = Field(..., description="Original filename or document title")
    chunk_text: str = Field(
        ..., max_length=600,
        description="The relevant excerpt that grounds the recommendation"
    )
    page_or_section: Optional[str] = Field(
        None, description="Page number or section heading within source document"
    )


# ── Recommendation ────────────────────────────────────────────────────────────

class Recommendation(BaseModel):
    """
    A single investment or portfolio action recommendation.
    Must include a suitability verdict and at least one citation.
    """
    idea: str = Field(..., description="The recommendation or action item")
    rationale: str = Field(..., description="Why this recommendation is appropriate for the client")
    suitability: SuitabilityVerdict
    suitability_reasoning: str = Field(
        ..., description="Explanation of the suitability verdict with policy references"
    )
    citations: list[Citation] = Field(
        ..., min_length=1,
        description="Must contain ≥1 grounding citation from the knowledge base"
    )
    priority: Optional[str] = Field(
        None, description="Urgency: 'high' | 'medium' | 'low'"
    )

    @model_validator(mode="after")
    def unsuitable_requires_reasoning(self) -> "Recommendation":
        """Unsuitable / blocked recommendations must explain why."""
        if self.suitability in (
            SuitabilityVerdict.UNSUITABLE,
            SuitabilityVerdict.REQUIRES_LICENSED_ADVICE,
        ) and len(self.suitability_reasoning) < 20:
            raise ValueError(
                "Unsuitable or licensed-advice recommendations require detailed reasoning (≥20 chars)"
            )
        return self


# ── ClientBrief ───────────────────────────────────────────────────────────────

class ClientBrief(BaseModel):
    """
    The final structured output of the RM Copilot agent.
    Contains a grounded, compliant meeting preparation brief for a specific client.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    client_id: str = Field(..., description="Client identifier (e.g. 'C-204')")
    client_name: str = Field(..., description="Client full name")
    risk_profile: RiskProfile
    rm_id: Optional[str] = Field(None, description="Generating RM's ID")

    # ── Portfolio Summary ─────────────────────────────────────────────────────
    portfolio_summary: str = Field(
        ..., description="Narrative summary of current holdings and allocation"
    )
    portfolio_risk_assessment: str = Field(
        ..., description="Assessment of overall portfolio risk versus stated risk profile"
    )
    total_aum: float = Field(..., ge=0.0, description="Total AUM in USD")
    allocation_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Asset class → allocation % mapping"
    )

    # ── Recommendations ───────────────────────────────────────────────────────
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="List of grounded, suitability-checked recommendations"
    )

    # ── Compliance ────────────────────────────────────────────────────────────
    compliance_status: ComplianceStatus
    compliance_notes: str = Field(
        default="",
        description="Compliance gate reasoning (e.g., why needs_review or blocked)"
    )

    # ── Escalation ────────────────────────────────────────────────────────────
    escalation_required: bool = Field(default=False)
    escalation_reason: Optional[str] = Field(
        None, description="Why escalation to a licensed advisor is needed"
    )

    # ── Talking Points ────────────────────────────────────────────────────────
    talking_points: list[str] = Field(
        ..., min_length=1,
        description="RM-ready bullet points for the client meeting"
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    disclaimers: list[str] = Field(
        default_factory=lambda: [
            "This brief is decision-support material for Relationship Managers only.",
            "It does not constitute personalised investment advice.",
            "All recommendations must be reviewed by a licensed advisor before acting.",
        ]
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of brief generation"
    )
    trace_id: Optional[str] = Field(
        None, description="LangSmith run trace ID for audit purposes"
    )
    model_used: str = Field(default="gpt-4o")

    @model_validator(mode="after")
    def escalation_consistency(self) -> "ClientBrief":
        """If blocked, escalation_required must be True."""
        if self.compliance_status == ComplianceStatus.BLOCKED:
            if not self.escalation_required:
                self.escalation_required = True
            if not self.escalation_reason:
                self.escalation_reason = self.compliance_notes or "Blocked by compliance gate"
        return self

    @property
    def has_unsuitable_recommendations(self) -> bool:
        return any(
            r.suitability == SuitabilityVerdict.UNSUITABLE
            for r in self.recommendations
        )

    @property
    def requires_licensed_advice(self) -> bool:
        return any(
            r.suitability == SuitabilityVerdict.REQUIRES_LICENSED_ADVICE
            for r in self.recommendations
        )
