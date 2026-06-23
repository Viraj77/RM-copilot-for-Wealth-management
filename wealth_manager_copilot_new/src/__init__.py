"""
Wealth Manager Copilot - Package initialization
"""
from src.models import (
    ClientBrief,
    Recommendation,
    PortfolioSummary,
    RiskProfile,
    ComplianceStatus,
)
from src.agent import WealthManagerAgent, create_langgraph_agent
from src.retriever import RAGRetriever, SuitabilityChecker
from src.ingestion import KnowledgeIngestionPipeline
from src.tools import create_tools_dict

__version__ = "1.0.0"
__all__ = [
    "ClientBrief",
    "Recommendation",
    "PortfolioSummary",
    "RiskProfile",
    "ComplianceStatus",
    "WealthManagerAgent",
    "create_langgraph_agent",
    "RAGRetriever",
    "SuitabilityChecker",
    "KnowledgeIngestionPipeline",
    "create_tools_dict",
]
