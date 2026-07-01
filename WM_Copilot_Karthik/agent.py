import os
from typing import TypedDict, List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from client_db import get_client_profile
from tools import portfolio_lookup, market_data_tool, suitability_checker, rag_retriever

load_dotenv()

# Pydantic Schemas for Structured Output
class Reco(BaseModel):
    idea: str = Field(description="The recommended investment idea or product.")
    rationale: str = Field(description="The underlying investment rationale for this recommendation.")
    suitability: str = Field(description="The suitability assessment, outlining why it fits the client's risk profile.")
    citations: List[str] = Field(default_factory=list, description="Citations to specific source documents/sections supporting this recommendation.")

class ClientBrief(BaseModel):
    client_id: str = Field(description="The client identifier.")
    risk_profile: str = Field(description="The client's documented risk profile.")
    portfolio_summary: str = Field(description="Summary of the client's current holdings and allocation.")
    recommendations: List[Reco] = Field(default_factory=list, description="List of investment recommendations.")
    compliance_status: str = Field(description="The compliance clearance status. Must be one of: Cleared, Needs Review, Blocked.")
    talking_points: List[str] = Field(default_factory=list, description="Pre-approved RM-ready discussion points.")
    disclaimer: str = Field(description="A clear, standard disclaimer: decision support for RMs, not automated advice.")

# Define Graph State
class AgentState(TypedDict):
    # Inputs
    query: str
    client_id: Optional[str]
    product_code: Optional[str]
    allocation_amount: float
    
    # Routing / mode
    response_mode: str          # "structured" | "freeform" — set by classify_intent_node
    force_structured: bool      # True when simulate_trade sidebar is active
    is_out_of_context: Optional[bool] # True if query is unrelated to wealth management / client advisory
    
    # Internal variables
    client_profile: Optional[Dict[str, Any]]
    plan: Optional[str]
    retrieved_evidence: List[Dict[str, Any]]
    suitability_report: Optional[Dict[str, Any]]
    draft_brief: Optional[Dict[str, Any]]
    
    # Output/Compliance variables
    compliance_status: str
    escalated: bool
    review_notes: Optional[str]
    final_brief: Optional[Dict[str, Any]]
    free_form_response: Optional[str]  # populated by free_form_answer_node

# Initialize LLM
def get_llm(temperature=0.0):
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

# Graph Nodes
def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """
    First node in the graph. Combines entity extraction (plan_node responsibility)
    with intent classification — deciding between structured (ClientBrief) and
    freeform (markdown prose) response modes, and detecting out-of-context queries.
    """
    llm = get_llm()

    force_structured = state.get("force_structured", False)

    prompt = f"""
    You are an assistant for a Wealth Management Relationship Manager (RM).
    Analyze this request: "{state['query']}"

    Step 1 — Identify if the query is out-of-context:
    Is this query unrelated to wealth management, client advisory, investment portfolios, fund comparisons, market research, or financial regulations/policies?
    Set `is_out_of_context` to True if it is a general knowledge question (e.g., "What is the capital of France?"), a greeting, trivia, or any other topic completely unrelated to wealth management or client advisory.
    Otherwise, set `is_out_of_context` to False.

    Step 2 — Extract entities (only if is_out_of_context is False):
    1. client_id: one of C-101, C-204, C-302 if mentioned. If multiple clients mentioned, use the first.
    2. product_code: e.g. HBGF, HCIF, HAEF, SCN-US-24, PG-001, PG-002, PG-003.
    3. allocation_amount: numerical USD value if mentioned, else 0.0.

    Step 3 — Classify response_mode (only if is_out_of_context is False):
    Choose "structured" if the request is:
    - A client-specific recommendation or suitability check ("Is X suitable for client Y?")
    - Asking to prepare a formal talking points brief for a specific client
    - Involves a specific allocation amount (trade simulation)
    - Needs a compliance gate check for a specific client+product pair
    - Asks to review, update, or summarise a specific client's past activity or account history ("What happened in C-204's last review?")
    - Requests a rebalancing plan or portfolio action for a named client ("Rebalance C-101's portfolio")

    Choose "freeform" if the request is:
    - A comparison of two or more funds ("Compare HBGF and HCIF")
    - A comparison of two or more client portfolios ("Compare C-204 and C-302 portfolios")
    - A general market/sector research question ("house view on fixed income", "tech outlook")
    - An educational or methodology question ("how is risk tier calculated?")
    - A broad summary with no specific client action intent
    - A hybrid query containing both a general research/macro outlook part and a client suitability/exposure question (e.g. "What is the house view on fixed income duration? Should Eleanor Vance add any fixed income exposure?")
    - A product shelf or product listing question, even if filtered by risk profile ("List all products suitable for a Growth profile", "What fixed income options are available?")
    - A stress test, scenario analysis, or hypothetical market impact question ("What happens to HAEF in a rate-hike scenario?")
    - A benchmark or relative-performance comparison ("How does C-204's portfolio compare to the benchmark?")
    - A multi-client advisory question without a specific action ("Should C-101 or C-302 hold more equities?")

    Tie-breaker rule: If the request is genuinely ambiguous, default to "freeform" UNLESS it explicitly asks
    to recommend, allocate, buy, or prepare a client brief for a named client — in which case use "structured".

    Note: if force_structured={force_structured}, ALWAYS return response_mode="structured".

    Step 4 — Write a brief plan explaining what data you will need to answer this query. If out-of-context, state that no financial data is needed.
    """

    class IntentClassification(BaseModel):
        client_id: Optional[str] = None
        product_code: Optional[str] = None
        allocation_amount: float = 0.0
        response_mode: Literal["structured", "freeform"] = "structured"
        is_out_of_context: bool = Field(description="Set to True if the query is completely unrelated to wealth management or client advisory queries.")
        plan: str
        reasoning: str

    structured_llm = llm.with_structured_output(IntentClassification)
    res = structured_llm.invoke(prompt)

    # Force structured when sidebar trade simulation is active
    mode = "structured" if force_structured else res.response_mode

    return {
        "client_id": res.client_id or state.get("client_id"),
        "product_code": res.product_code or state.get("product_code"),
        "allocation_amount": res.allocation_amount or state.get("allocation_amount", 0.0),
        "response_mode": mode,
        "is_out_of_context": res.is_out_of_context,
        "plan": f"[Mode: {mode.upper()}] {res.plan} | Reasoning: {res.reasoning}",
    }

def gather_portfolio_node(state: AgentState) -> Dict[str, Any]:
    """
    Fetches the client profile and current holdings.
    """
    client_id = state.get("client_id")
    if not client_id:
        return {"client_profile": None}
        
    profile = get_client_profile(client_id)
    return {"client_profile": profile}

def gather_research_node(state: AgentState) -> Dict[str, Any]:
    """
    Performs RAG retrieval over the vector store.
    Uses metadata filtering based on RM's entitlement level.
    """
    client_profile = state.get("client_profile")
    # Default to False (Public only) if no client/RM is loaded
    rm_access = False
    if client_profile:
        rm_access = client_profile.get("rm_access_to_private", False)
        
    query = state["query"]
    
    # Expand query with client risk profile and product details if available
    search_query = query
    if client_profile:
        search_query += f" client risk: {client_profile.get('risk_profile')}"
    if state.get("product_code"):
        search_query += f" product: {state.get('product_code')}"
        
    evidence = rag_retriever(search_query, rm_access_to_private=rm_access)
    
    # Add product details directly if product code is queried
    if state.get("product_code"):
        p_details = market_data_tool(state["product_code"])
        if "error" not in p_details:
            evidence.append({
                "doc_id": p_details["product_code"],
                "type": "product",
                "date": "N/A",
                "source": "System Database",
                "sensitivity": "Public",
                "content": str(p_details)
            })
            
    return {"retrieved_evidence": evidence}

def check_suitability_node(state: AgentState) -> Dict[str, Any]:
    """
    Runs programmatic and LLM-assisted compliance check.
    """
    client_id = state.get("client_id")
    product_code = state.get("product_code")
    allocation_amount = state.get("allocation_amount", 0.0)
    
    # Standard programmatic check
    report = {"status": "Cleared", "violations": [], "reasons": [], "citations": []}
    if client_id and product_code:
        report = suitability_checker(client_id, product_code, allocation_amount)
        
    # LLM check to see if there are any qualitative violations in compliance documents
    # (e.g. licensed advice request like providing legal, tax, or discretionary advice)
    llm = get_llm()
    evidence_text = "\n\n".join([f"Source: {e['doc_id']} ({e['type']})\nContent: {e['content']}" for e in state["retrieved_evidence"]])
    
    client_profile = state.get("client_profile") or {}
    client_risk = client_profile.get("risk_profile", "Unknown")
    client_holdings = client_profile.get("holdings", [])
    
    prompt = f"""
    Review the RM request: "{state['query']}"
    
    Client Risk Profile: {client_risk}
    Holdings: {client_holdings}
    
    Retrieved Compliance & Research Evidence:
    {evidence_text}
    
    Does the request require licensing advice or represent a compliance issue that standard rules might miss?
    Specifically check:
    1. Does the RM request tax, legal, estate planning, or discretionary portfolio management advice (which requires separate licensing and should trigger 'Needs Review' or 'Blocked')?
    2. Is there restricted research accessed by an unauthorized RM (e.g. Tier 2 research for Tier 1 RM)? Note that our retriever filters this, but verify.
    3. Is this recommendation suitable and grounded?
    
    Important: Informational requests (comparing funds, listing products, or explaining rules) are NOT investment recommendations and do NOT trigger suitability blocks, even if a client profile is in context. Only flag suitability mismatches as 'Blocked' or 'Needs Review' if the RM is explicitly asking to recommend, buy, or allocate a specific product to the client.
    
    If there is a license restriction or general compliance issue, specify it.
    """
    
    class ComplianceLLMReport(BaseModel):
        compliance_status: str = Field(description="Must be: Cleared, Needs Review, or Blocked")
        reasons: List[str] = Field(description="Reasons for this status.")
        citations: List[str] = Field(description="Rules cited.")
        requires_escalation: bool = Field(description="True if escalation or human review is required.")
        
    structured_llm = llm.with_structured_output(ComplianceLLMReport)
    llm_report = structured_llm.invoke(prompt)
    
    # Merge reports: take the most restrictive status
    final_status = "Cleared"
    if "Blocked" in [report["status"], llm_report.compliance_status]:
        final_status = "Blocked"
    elif "Needs Review" in [report["status"], llm_report.compliance_status] or llm_report.requires_escalation:
        final_status = "Needs Review"
        
    merged_violations = report.get("violations", []) + [c for c in llm_report.citations if c not in report.get("citations", [])]
    merged_reasons = report.get("reasons", []) + llm_report.reasons
    merged_citations = report.get("citations", []) + llm_report.citations
    
    merged_report = {
        "status": final_status,
        "violations": merged_violations,
        "reasons": list(set(merged_reasons)),
        "citations": list(set(merged_citations))
    }
    
    return {
        "suitability_report": merged_report,
        "compliance_status": final_status
    }
 
def synthesize_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates the ClientBrief structure with citations and pre-approved talking points.
    Ensure all claims are grounded in the retrieved evidence.
    Used by the STRUCTURED response path only.
    """
    llm = get_llm()
    evidence_list = state.get("retrieved_evidence", []) or []
    evidence_text = "\n\n".join([f"Source: {e['doc_id']} ({e['type']})\nContent: {e['content']}" for e in evidence_list])
    client_profile = state.get("client_profile") or {}
    suitability_report = state.get("suitability_report") or {}
    
    # If the compliance status was overridden to Cleared by human review
    current_status = state.get("compliance_status", "Cleared")
    is_override = (current_status == "Cleared" and suitability_report.get("status", "Cleared") != "Cleared")
    
    if is_override:
        suitability_report = suitability_report.copy()
        suitability_report["status"] = "Cleared"
        
    prompt = f"""
    You are a Wealth Management Assistant. Generate a grounded ClientBrief for the Relationship Manager.
    
    Request: "{state['query']}"
    Client Profile: {client_profile}
    Compliance Report: {suitability_report}
    
    Retrieved Knowledge / Evidence:
    {evidence_text}
    
    Requirements:
    1. Summarize the portfolio holdings and risk profile clearly.
    2. Under 'recommendations', provide recommendations with rationale, suitability assessment, and precise citations to the document IDs.
    3. Set 'compliance_status' based on the Compliance Report status.
    4. Provide pre-approved talking points for the RM. All talking points must be pre-approved or directly derived from product guides (e.g. PG-001/PG-002/PG-003 Section 15 or similar). Do not make up facts. Limit the list strictly to a maximum of 3 highly concise bullet points focusing only on client-facing talking points. Omit relationship manager meta-text or administrative compliance summaries.
    5. State a clear disclaimer: "Decision support for RMs, not automated advice."
    6. Grounding Constraints: All rationales, suitability remarks, and talking points must rely strictly and solely on the facts, numbers, rates, and parameters directly present in the Retrieved Knowledge / Evidence. Do not extrapolate, infer, or assume details not present. If the evidence does not state a metric or detail, explicitly note that it is not documented, rather than fabricating a figure.
    
    Generate the structured brief.
    """
    
    structured_llm = llm.with_structured_output(ClientBrief)
    brief = structured_llm.invoke(prompt)
    brief_dict = brief.model_dump()
    
    # Programmatically enforce override values to ensure compliance consistency
    if is_override or state.get("escalated"):
        brief_dict["compliance_status"] = "Cleared"
        notes = state.get("review_notes") or "Approved by Relationship Manager."
        override_note = f"RM Note: Approved with review notes: {notes}"
        if override_note not in brief_dict["talking_points"]:
            brief_dict["talking_points"].append(override_note)
            
    return {
        "draft_brief": brief_dict,
        "final_brief": brief_dict
    }


def free_form_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Terminal node for the FREEFORM response path.
    Uses all data already in state — retrieved_evidence (RAG) and client_profile
    (portfolio data) — to produce a rich, cited markdown response.

    If the query is out of context, returns a standardized reply bypass.
    """
    if state.get("is_out_of_context"):
        return {
            "free_form_response": "I can only provide support related to wealth management and client advisory queries.",
            "final_brief": None
        }

    llm = get_llm()

    evidence_text = "\n\n".join(
        [f"[{e['doc_id']}] ({e['type'].upper()}, {e['sensitivity']}):\n{e['content']}"
         for e in state.get("retrieved_evidence", [])]
    )

    client_profile = state.get("client_profile")
    portfolio_section = ""
    # Only include client holdings details if the query explicitly refers to the client, portfolio, or client review.
    query_lower = state["query"].lower()
    has_client_ref = "client" in query_lower or "portfolio" in query_lower or "holdings" in query_lower or (client_profile and client_profile.get("client_id", "").lower() in query_lower)
    
    if client_profile and has_client_ref:
        holdings_lines = "\n".join(
            [f"  - {h['product_name']} ({h['product_code']}): USD {h['allocation_amount']:,.2f} [{h['asset_class']}]"
             for h in client_profile.get("holdings", [])]
        )
        portfolio_section = f"""
## Client Context
- **Client**: {client_profile.get('name')} ({client_profile.get('client_id')})
- **Risk Profile**: {client_profile.get('risk_profile')}
- **Current Holdings**:
{holdings_lines}
"""

    prompt = f"""
    You are an expert Wealth Management Research Analyst assisting a Relationship Manager.
    Answer the following RM request directly and concisely. Do not include boilerplate greetings, introductory summaries, or off-topic meta-commentary.

    Request: "{state['query']}"
    {portfolio_section}

    Retrieved Knowledge Base Evidence:
    {evidence_text}

    Response Guidelines:
    1. Be extremely concise and focus entirely on answering the request. Limit the response only to relevant facts directly stated in the retrieved evidence. Keep it short (2-4 sentences max) and highly specific.
       - **Important constraint**: If the request asks to list all products, structured products, or fixed income options, do NOT dump a long table or list of all rows. Instead, summarize the categories (e.g., Structured Capital Notes, Fixed Income) and list only the top 5 most prominent products as examples. State that the full shelf of 160+ items is available in the system registry.
    2. Use markdown formatting: headers (##, ###), bullet lists, and **bold** for key terms.
    3. Use comparison tables (| Column | ... |) where relevant, e.g. for fund or portfolio comparisons.
    4. Cite source documents inline using brackets, e.g. [PG-001], [RN-003], [CMP-002].
    5. If comparing funds or portfolios, structure the response with a clear side-by-side section. Keep it highly concise — limit the description of each compared fund to 2 lines max, using a bulleted format.
    6. If analysing market/sector views, clearly state the source note and date.
    7. Do NOT fabricate, extrapolate, or assume performance numbers or facts not directly stated in the evidence. If the requested details are not present, explicitly state that they are not documented in the sources.
    8. End with a brief **Disclaimer**: "This is an analytical summary for RM reference. Not client-specific investment advice."

    Produce the response now:
    """

    # Invoke LLM without structured output — free prose/markdown
    response = llm.invoke(prompt)

    return {
        "free_form_response": response.content,
        "final_brief": None,  # Explicitly no ClientBrief for freeform
    }

def human_review_node(state: AgentState) -> Dict[str, Any]:
    """
    Node representing human RM override.
    LangGraph will pause BEFORE this node executes.
    When resumed, it updates the state based on human inputs.
    """
    notes = state.get("review_notes", "No notes provided.")
    brief = (state.get("draft_brief") or {}).copy()
    
    # If the human RM approves/overrides, we cleared the status
    # In a real app, the RM inputs are passed into state updates.
    # We will mark it cleared if the RM chooses to approve.
    brief["compliance_status"] = "Cleared"
    if "talking_points" not in brief:
        brief["talking_points"] = []
    brief["talking_points"].append(f"RM Note: Approved with review notes: {notes}")
    
    return {
        "final_brief": brief,
        "compliance_status": "Cleared",
        "escalated": True
    }

# Conditional Edges
def route_compliance_or_freeform(state: AgentState) -> str:
    """
    The ONLY routing fork in the graph — runs after check_suitability_node.

    Priority:
    1. If compliance is not Cleared (Blocked or Needs Review) → human_review
       (applies to BOTH structured and freeform queries — compliance always runs)
    2. If Cleared and response_mode is freeform → free_form_answer
    3. If Cleared and response_mode is structured → synthesize
    """
    status = state.get("compliance_status", "Cleared")
    if status != "Cleared":
        return "human_review"
    if state.get("response_mode") == "freeform":
        return "free_form_answer"
    return "synthesize"

def route_post_review(state: AgentState) -> str:
    """
    After human review override, always re-synthesize to generate the final brief.
    Note: post-override always produces a structured ClientBrief regardless of
    original response_mode — the RM's override decision is a formal compliance action.
    """
    return "synthesize"

def route_classify_intent(state: AgentState) -> str:
    """
    Routes from classify_intent node.
    If the query is out of context, goes directly to free_form_answer node.
    Otherwise, proceeds with the standard pipeline (gather_portfolio).
    """
    if state.get("is_out_of_context"):
        return "free_form_answer"
    return "gather_portfolio"

# Assemble Graph
def create_agent(checkpointer=None, interrupt_before_nodes=["human_review"]):
    workflow = StateGraph(AgentState)

    # Add Nodes
    # Entry point: classify_intent_node replaces plan_node and also classifies
    # response_mode (structured vs freeform)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("gather_portfolio", gather_portfolio_node)
    workflow.add_node("gather_research", gather_research_node)
    workflow.add_node("check_suitability", check_suitability_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("free_form_answer", free_form_answer_node)
    workflow.add_node("human_review", human_review_node)

    # Entry point and routing
    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_classify_intent,
        {
            "gather_portfolio": "gather_portfolio",
            "free_form_answer": "free_form_answer",
        }
    )
    
    workflow.add_edge("gather_portfolio", "gather_research")
    workflow.add_edge("gather_research", "check_suitability")

    # The ONLY fork — after compliance check:
    # Cleared + freeform   → free_form_answer
    # Cleared + structured → synthesize
    # Blocked/Needs Review → human_review (both modes)
    workflow.add_conditional_edges(
        "check_suitability",
        route_compliance_or_freeform,
        {
            "synthesize": "synthesize",
            "free_form_answer": "free_form_answer",
            "human_review": "human_review",
        }
    )

    # After human review override, always produce a structured ClientBrief
    workflow.add_edge("human_review", "synthesize")
    workflow.add_edge("synthesize", END)
    workflow.add_edge("free_form_answer", END)

    # Enable memory for checkpointing (required for HITL interrupts)
    if checkpointer is None:
        checkpointer = MemorySaver()

    # Compile graph with custom interrupt list (can be empty to bypass pauses during eval)
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before_nodes
    )

    return app
