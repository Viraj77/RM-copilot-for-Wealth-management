"""
Pydantic models for Wealth Manager Copilot
"""
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class RiskProfile(str, Enum):
    """Client risk profile classification"""
    CONSERVATIVE = "Conservative"
    BALANCED = "Balanced"
    GROWTH = "Growth"
    AGGRESSIVE = "Aggressive"


class ComplianceStatus(str, Enum):
    """Compliance gate status"""
    CLEARED = "Cleared"
    NEEDS_REVIEW = "Needs Review"
    BLOCKED = "Blocked"


class SuitabilityAssessment(BaseModel):
    """Suitability assessment for a recommendation"""
    suitable_for_profile: bool = Field(..., description="Is this suitable for the risk profile?")
    reasoning: str = Field(..., description="Reasoning for suitability assessment")
    compliance_check_passed: bool = Field(..., description="Did it pass compliance checks?")
    compliance_notes: Optional[str] = Field(None, description="Any compliance concerns")


class Citation(BaseModel):
    """Citation linking to source material"""
    doc_id: str = Field(..., description="Document identifier")
    doc_type: str = Field(..., description="Type of document (product/policy/research)")
    chunk_text: str = Field(..., description="Relevant chunk from the source")
    page_num: Optional[int] = Field(None, description="Page number if applicable")
    source: str = Field(..., description="Document source/name")
    date: Optional[str] = Field(None, description="Document date")


class Recommendation(BaseModel):
    """A single recommendation with rationale and citations"""
    idea: str = Field(..., description="The recommendation/idea")
    product_name: Optional[str] = Field(None, description="Product name if applicable")
    rationale: str = Field(..., description="Why this recommendation makes sense")
    suitability: SuitabilityAssessment = Field(..., description="Suitability assessment")
    citations: List[Citation] = Field(..., description="Supporting citations")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    action_required: bool = Field(default=False, description="Does this need human review?")


class PortfolioSummary(BaseModel):
    """Client portfolio summary"""
    client_id: str = Field(..., description="Client identifier")
    total_value: float = Field(..., description="Total portfolio value")
    allocation: Dict[str, float] = Field(..., description="Asset allocation percentages")
    holdings: List[Dict[str, Any]] = Field(..., description="Detailed holdings")
    risk_score: float = Field(..., ge=0, le=10, description="Current portfolio risk score")


class ClientBrief(BaseModel):
    """Main output: a grounded, compliant brief for the RM"""
    client_id: str = Field(..., description="Client identifier")
    brief_id: str = Field(..., description="Brief unique identifier")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    risk_profile: RiskProfile = Field(..., description="Client risk profile")
    portfolio_summary: PortfolioSummary = Field(..., description="Portfolio summary")
    
    recommendations: List[Recommendation] = Field(
        ..., 
        description="List of recommendations with citations"
    )
    talking_points: List[str] = Field(
        ..., 
        description="RM-ready discussion points"
    )
    
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.CLEARED,
        description="Overall compliance status"
    )
    compliance_notes: Optional[str] = Field(
        None, 
        description="Any compliance notes or concerns"
    )
    
    escalated_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Items requiring licensed advice or human sign-off"
    )
    
    disclaimer: str = Field(
        default="This brief is decision support only. Not automated investment advice.",
        description="Standard disclaimer"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (retrieval stats, tool calls, etc.)"
    )


class AgentState(BaseModel):
    """Shared state for the LangGraph agent"""
    client_id: str = Field(..., description="Client identifier")
    request: str = Field(..., description="RM's request")
    step: int = Field(default=0, description="Current step in agent flow")
    
    # Gathered information
    portfolio: Optional[PortfolioSummary] = Field(None, description="Fetched portfolio")
    research_docs: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved research")
    product_docs: List[Dict[str, Any]] = Field(default_factory=list, description="Product information")
    
    # Processing
    draft_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Draft recommendations before compliance check"
    )
    suitability_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Suitability check results"
    )
    
    # Output
    brief: Optional[ClientBrief] = Field(None, description="Final client brief")
    
    # Metadata
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="All tool calls made")
    trace_logs: List[str] = Field(default_factory=list, description="Execution trace logs")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


class KnowledgeDocument(BaseModel):
    """Metadata for knowledge documents in vector store"""
    doc_id: str = Field(..., description="Unique document ID")
    doc_type: str = Field(..., description="Type: product/policy/research")
    title: str = Field(..., description="Document title")
    source: str = Field(..., description="Document source/filename")
    date: Optional[str] = Field(None, description="Document date")
    sensitivity: str = Field(default="public", description="Sensitivity level: public/restricted")
    chunks: List[str] = Field(..., description="Text chunks from document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
