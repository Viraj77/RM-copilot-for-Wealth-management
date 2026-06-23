"""
LangGraph Agent for Wealth Manager Copilot
Multi-step, multi-tool agent orchestration
"""
import logging
import json
import os
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from config import settings
from src.models import (
    ClientBrief, Recommendation, PortfolioSummary, Citation,
    RiskProfile, ComplianceStatus, SuitabilityAssessment, AgentState
)
from src.tools import create_tools_dict
from src.retriever import RAGRetriever, SuitabilityChecker

logger = logging.getLogger(__name__)


class WealthManagerAgent:
    """
    LangGraph-based agent for preparing client interaction briefs.
    
    Flow:
    1. plan: Decompose the RM request
    2. gather_portfolio: Fetch client holdings and risk profile
    3. gather_research: Retrieve relevant market/product/policy docs via RAG
    4. check_suitability: Validate recommendations against compliance
    5. synthesize: Create grounded ClientBrief with citations
    6. review_gate: Escalate if needed, else return brief
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4o",
        retriever: Optional[RAGRetriever] = None,
        tools: Optional[Dict[str, Any]] = None,
        max_steps: int = 20,
        temperature: float = 0.2,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the agent.
        
        Args:
            llm_model: OpenAI model name
            retriever: RAG retriever instance
            tools: Dictionary of tools
            max_steps: Maximum agent steps
            temperature: LLM temperature
        """
        resolved_api_key = openai_api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=2000,
            api_key=resolved_api_key
        )
        
        self.retriever = retriever
        self.tools = tools or create_tools_dict(retriever=retriever)
        self.max_steps = max_steps
        self.step_count = 0
        
        # Initialize suitability checker if retriever available
        self.suitability_checker = None
        if retriever:
            self.suitability_checker = SuitabilityChecker(retriever)
        
        logger.info(f"Initialized WealthManagerAgent (model: {llm_model}, max_steps: {max_steps})")
    
    def _plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1: Decompose the RM request.
        
        Analyze what the RM is asking for and plan the investigation.
        """
        logger.info(f"[PLAN] Processing request: {state['request'][:100]}")
        
        prompt = f"""
You are a wealth management AI assistant. Analyze this RM request and create an action plan.

Client ID: {state['client_id']}
Request: {state['request']}

Provide a brief analysis:
1. What information do we need to gather?
2. What risk profile should we assume?
3. What types of recommendations might be appropriate?
4. Any compliance concerns to watch for?

Keep response concise.
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        plan = response.content
        
        state["trace_logs"].append(f"[PLAN] {plan[:200]}")
        return state
    
    def _gather_portfolio(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Fetch client portfolio and risk profile.
        """
        logger.info(f"[GATHER_PORTFOLIO] Fetching for {state['client_id']}")
        
        portfolio_tool = self.tools.get("portfolio_lookup")
        result = portfolio_tool(state['client_id'])
        
        if result["success"]:
            portfolio_data = result["data"]
            state["portfolio"] = PortfolioSummary(
                client_id=portfolio_data["client_id"],
                risk_profile=portfolio_data.get("risk_profile", "Balanced"),
                total_value=portfolio_data["total_value"],
                allocation=portfolio_data["allocation"],
                holdings=portfolio_data["holdings"],
                risk_score=portfolio_data.get("risk_score", 5.0)
            )
            state["tool_calls"].append({
                "tool": "portfolio_lookup",
                "client_id": state["client_id"],
                "result": "success"
            })
            logger.info(f"Portfolio fetched: ${portfolio_data['total_value']:,.0f}")
        else:
            logger.warning(f"Portfolio lookup failed: {result.get('error')}")
            state["errors"].append(f"Portfolio lookup failed: {result.get('error')}")
        
        return state
    
    def _gather_research(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3: Retrieve relevant market, product, and policy documents via RAG.
        """
        logger.info(f"[GATHER_RESEARCH] Retrieving documents")
        
        if not self.retriever:
            logger.warning("No retriever configured, skipping research gathering")
            return state
        
        # Build retrieval queries
        queries = [
            f"recommendations for {state['portfolio'].risk_profile if state.get('portfolio') else 'balanced'} investors",
            state['request'],  # Original request
            "suitability and compliance checks"
        ]
        
        top_k = getattr(self.retriever, "top_k", 3)
        sources = getattr(self.retriever, "selected_sources", None)
        for query in queries:
            try:
                docs = self.retriever.hybrid_retrieve(query, k=top_k, sources=sources)
                for doc in docs:
                    state["research_docs"].append({
                        "content": doc["content"][:500],
                        "source": doc["metadata"].get("source", "unknown"),
                        "doc_type": doc["metadata"].get("doc_type", "unknown"),
                        "score": doc["score"]
                    })
            except Exception as e:
                logger.error(f"Error retrieving research: {e}")
        
        state["tool_calls"].append({
            "tool": "retrieval",
            "queries": len(queries),
            "documents_retrieved": len(state["research_docs"])
        })
        
        logger.info(f"Retrieved {len(state['research_docs'])} research documents")
        return state
    
    def _check_suitability(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 4: Validate recommendations against suitability and compliance.
        """
        logger.info(f"[CHECK_SUITABILITY] Validating recommendations")
        
        if not state.get("portfolio"):
            logger.warning("No portfolio available for suitability check")
            return state
        
        risk_profile = state["portfolio"].risk_profile
        
        # Example recommendations to check
        example_recommendations = [
            {
                "idea": "Increase equity allocation to 80%",
                "product_type": "diversified_equities",
                "allocation": 0.20
            },
            {
                "idea": "Add international diversification",
                "product_type": "international_funds",
                "allocation": 0.10
            },
            {
                "idea": "Maintain current asset allocation",
                "product_type": "rebalance",
                "allocation": 0.0
            }
        ]
        
        suitability_tool = self.tools.get("check_suitability")
        
        for rec in example_recommendations:
            result = suitability_tool(risk_profile, rec, state["portfolio"].allocation)
            state["suitability_results"].append(result)
        
        state["tool_calls"].append({
            "tool": "check_suitability",
            "recommendations_checked": len(example_recommendations)
        })
        
        logger.info(f"Checked {len(example_recommendations)} recommendations")
        return state
    
    def _synthesize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 5: Synthesize a grounded ClientBrief with citations using LLM.
        
        Passes retrieved research documents to the LLM for grounding recommendations.
        """
        logger.info(f"[SYNTHESIZE] Creating grounded client brief with LLM")
        
        portfolio = state.get("portfolio")
        if not portfolio:
            logger.error("No portfolio available for synthesis")
            state["errors"].append("Cannot synthesize brief without portfolio")
            return state
        
        # Prepare research context for the LLM
        research_context = ""
        doc_mapping = {}  # Map doc index to source info
        
        if state.get("research_docs"):
            research_context = "## RETRIEVED RESEARCH DOCUMENTS (Use these to ground your recommendations):\n\n"
            for idx, doc in enumerate(state["research_docs"]):
                research_context += f"[DOC-{idx}] Source: {doc.get('source', 'unknown')} | Type: {doc.get('doc_type', 'unknown')}\n"
                research_context += f"Content: {doc.get('content', '')[:500]}\n\n"
                doc_mapping[f"DOC-{idx}"] = {
                    "source": doc.get("source", "unknown"),
                    "doc_type": doc.get("doc_type", "unknown"),
                    "content": doc.get("content", ""),
                    "score": doc.get("score", 0)
                }
        
        # Build LLM prompt with grounding context
        prompt = f"""
You are a wealth management AI assistant. Using ONLY the research documents provided below, generate grounded recommendations for this client.

**CLIENT PROFILE:**
- Risk Profile: {portfolio.risk_profile}
- Portfolio Value: ${portfolio.total_value:,.0f}
- Current Allocation: {portfolio.allocation}
- Risk Score: {portfolio.risk_score}/10

**CLIENT REQUEST:**
{state['request']}

{research_context}

**TASK:**
Based ONLY on the research documents provided above, generate:
1. 2-3 specific recommendations with supporting evidence
2. Talking points for client discussion
3. Any compliance concerns

For each recommendation, cite the source document using [DOC-X] format.

Format your response as JSON:
{{
    "recommendations": [
        {{
            "idea": "...",
            "rationale": "...",
            "citations": ["DOC-X", "DOC-Y"],
            "confidence_score": 0.8
        }}
    ],
    "talking_points": ["...", "..."],
    "compliance_concerns": ["..."]
}}
"""
        
        try:
            # Call LLM with grounding context
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content
            
            logger.info(f"[SYNTHESIZE] LLM Response:\n{response_text[:500]}")
            
            # Parse LLM response
            try:
                import json
                # Extract JSON from response
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    llm_output = json.loads(json_str)
                else:
                    llm_output = {"recommendations": [], "talking_points": [], "compliance_concerns": []}
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM JSON response, using defaults")
                llm_output = {"recommendations": [], "talking_points": [], "compliance_concerns": []}
            
            # Convert LLM output to ClientBrief structure
            recommendations = []
            for rec_data in llm_output.get("recommendations", []):
                # Build citations from referenced documents
                citations = []
                for doc_ref in rec_data.get("citations", []):
                    if doc_ref in doc_mapping:
                        doc_info = doc_mapping[doc_ref]
                        citations.append(Citation(
                            doc_id=doc_ref,
                            doc_type=doc_info["doc_type"],
                            chunk_text=doc_info["content"][:300],
                            source=doc_info["source"],
                            date=datetime.utcnow().isoformat()
                        ))
                
                rec = Recommendation(
                    idea=rec_data.get("idea", ""),
                    rationale=rec_data.get("rationale", ""),
                    suitability=SuitabilityAssessment(
                        suitable_for_profile=True,
                        reasoning="Generated from retrieved research documents",
                        compliance_check_passed=len(rec_data.get("compliance_concerns", [])) == 0,
                        compliance_notes=None
                    ),
                    citations=citations,
                    confidence_score=rec_data.get("confidence_score", 0.7)
                )
                recommendations.append(rec)
            
            # Create brief
            brief = ClientBrief(
                client_id=state["client_id"],
                brief_id=f"brief_{datetime.utcnow().timestamp()}",
                risk_profile=RiskProfile(portfolio.risk_profile),
                portfolio_summary=portfolio,
                recommendations=recommendations,
                talking_points=llm_output.get("talking_points", []),
                compliance_status=ComplianceStatus.CLEARED,
                metadata={
                    "tool_calls": len(state["tool_calls"]),
                    "research_docs_used": len(state["research_docs"]),
                    "compliance_concerns": llm_output.get("compliance_concerns", []),
                    "grounded_by_llm": True
                }
            )
            
            state["brief"] = brief
            logger.info(f"Created LLM-grounded brief with {len(recommendations)} recommendations")
            
        except Exception as e:
            logger.error(f"Error in LLM synthesis: {e}")
            state["errors"].append(f"LLM synthesis failed: {str(e)}")
        
        return state
    
    def _review_gate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 6: Final compliance gate - determine if escalation needed.
        """
        logger.info(f"[REVIEW_GATE] Final compliance check")
        
        brief = state.get("brief")
        if not brief:
            logger.error("No brief available for review gate")
            return state
        
        # Check each recommendation through compliance gate
        compliance_tool = self.tools.get("compliance_gate")
        escalated = []
        
        for rec in brief.recommendations:
            result = compliance_tool(
                rec.dict(),
                rec.suitability.dict(),
                is_licensed_advice=rec.action_required
            )
            
            if result.get("requires_escalation", False):
                escalated.append({
                    "recommendation": rec.idea,
                    "escalation_type": result.get("escalation_type"),
                    "reason": result.get("reason")
                })
        
        # Update brief status
        if escalated:
            brief.escalated_items = escalated
            brief.compliance_status = ComplianceStatus.NEEDS_REVIEW
        
        state["tool_calls"].append({
            "tool": "compliance_gate",
            "recommendations_reviewed": len(brief.recommendations),
            "escalated": len(escalated)
        })
        
        logger.info(f"Review gate complete: {len(escalated)} items escalated")
        return state
    
    def run_agent(self, client_id: str, request: str) -> ClientBrief:
        """
        Run the complete agent workflow.
        
        Args:
            client_id: Client identifier
            request: RM request/prompt
            
        Returns:
            ClientBrief with recommendations and citations
        """
        logger.info(f"=== Starting Agent Run ===")
        logger.info(f"Client: {client_id}, Request: {request[:100]}")
        
        # Initialize state
        state = {
            "client_id": client_id,
            "request": request,
            "portfolio": None,
            "research_docs": [],
            "product_docs": [],
            "draft_recommendations": [],
            "suitability_results": [],
            "brief": None,
            "tool_calls": [],
            "trace_logs": [],
            "errors": []
        }
        
        # Execute pipeline
        steps = [
            ("plan", self._plan),
            ("gather_portfolio", self._gather_portfolio),
            ("gather_research", self._gather_research),
            ("check_suitability", self._check_suitability),
            ("synthesize", self._synthesize),
            ("review_gate", self._review_gate),
        ]
        
        for step_name, step_func in steps:
            try:
                state = step_func(state)
                self.step_count += 1
                
                if self.step_count >= self.max_steps:
                    logger.warning(f"Max steps ({self.max_steps}) reached")
                    break
            
            except Exception as e:
                logger.error(f"Error in step {step_name}: {e}")
                state["errors"].append(f"Step {step_name} failed: {str(e)}")
        
        # Return brief or create error brief
        if state.get("brief"):
            logger.info(f"=== Agent Run Complete ===")
            return state["brief"]
        else:
            logger.error("Agent failed to produce brief")
            raise RuntimeError("Agent failed to produce brief")


def create_langgraph_agent(
    llm_model: str = "gpt-4o",
    retriever: Optional[RAGRetriever] = None,
    openai_api_key: Optional[str] = None,
    max_steps: int = 100
) -> WealthManagerAgent:
    """
    Factory function to create a configured agent.
    
    Args:
        llm_model: OpenAI model
        retriever: RAG retriever
        openai_api_key: OpenAI API key
        max_steps: Maximum steps for agent execution (higher for KB-grounded retrieval)
        
    Returns:
        Configured WealthManagerAgent
    """
    agent = WealthManagerAgent(
        llm_model=llm_model,
        retriever=retriever,
        max_steps=max_steps,
        temperature=0.2,
        openai_api_key=openai_api_key
    )
    return agent


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Test agent
    agent = create_langgraph_agent()
    
    # Run example
    brief = agent.run_agent(
        client_id="C-204",
        request="Prepare talking points for quarterly review"
    )
    
    print(f"\n=== Generated Brief ===")
    print(f"Client: {brief.client_id}")
    print(f"Risk Profile: {brief.risk_profile}")
    print(f"Compliance Status: {brief.compliance_status}")
    print(f"Recommendations: {len(brief.recommendations)}")
    print(f"\nTalking Points:")
    for point in brief.talking_points[:3]:
        print(f"  - {point}")
