"""LangGraph agent node implementations."""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from src.agent.state import AgentState
from src.config import get_openai_api_key, settings
from src.utils.http_client import get_async_http_client, get_sync_http_client
from src.guardrails.compliance import apply_review_gate, detect_licensed_advice_request
from src.models import ClientBrief, ComplianceStatus, Reco, RiskProfile, Sensitivity
from src.tools.market_data import market_data
from src.tools.portfolio import get_portfolio_summary, load_client_portfolio, portfolio_lookup
from src.tools.retriever import HybridRAGRetriever, chunks_to_citations
from src.tools.suitability import check_suitability_logic, suitability_checker
from src.utils.logging import log_audit_event

retriever = HybridRAGRetriever()
AGENT_TOOLS = [portfolio_lookup, market_data, suitability_checker]
TOOL_MAP = {t.name: t for t in AGENT_TOOLS}


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.llm_temperature,
        api_key=get_openai_api_key() or None,
        http_client=get_sync_http_client(),
        http_async_client=get_async_http_client(),
    )


def _steps_exceeded(state: AgentState) -> bool:
    return state.get("step_count", 0) >= settings.max_agent_steps


def _extract_client_id(query: str) -> str | None:
    match = re.search(r"C-\d+", query, re.IGNORECASE)
    return match.group(0).upper() if match else None


def _extract_risk_profile(query: str) -> RiskProfile | None:
    ql = query.lower()
    for profile in RiskProfile:
        if profile.value.lower() in ql:
            return profile
    return None


def _extract_fund(query: str) -> str | None:
    """Extract Horizon product code or alias from query text."""
    patterns = [r"PG-\d{3}", r"\bHAEF\b", r"\bHCIF\b", r"\bHBGF\b", r"\bPG\d{3}\b"]
    query_upper = query.upper()
    for pattern in patterns:
        match = re.search(pattern, query_upper)
        if match:
            token = match.group(0)
            if token.startswith("PG") and "-" not in token:
                return token[:2] + "-" + token[2:]
            return token
    match = re.search(r"(?:fund|product)\s+(\w[\w-]*)", query, re.IGNORECASE)
    return match.group(1).upper() if match else None


def plan_node(state: AgentState) -> dict:
    """Decompose the RM request into an execution plan."""
    query = state["query"]
    client_id = state.get("client_id") or _extract_client_id(query)
    requires_escalation = detect_licensed_advice_request(query)

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You are a wealth management copilot planner. Decompose the RM request "
            "into steps: portfolio lookup, research retrieval, suitability check, "
            "synthesis. Be concise."
        )),
        HumanMessage(content=f"RM Request: {query}\nClient ID detected: {client_id}"),
    ])

    plan = response.content
    log_audit_event("plan", {"query": query, "client_id": client_id, "plan": plan})

    return {
        "plan": plan,
        "client_id": client_id,
        "requires_escalation": requires_escalation,
        "step_count": state.get("step_count", 0) + 1,
        "messages": [AIMessage(content=f"Plan: {plan}")],
    }


def react_tools_node(state: AgentState) -> dict:
    """ReAct-style bounded tool calling (max 1 tool invocation per step)."""
    if _steps_exceeded(state):
        log_audit_event("react_tools", {"skipped": "max_steps_reached"})
        return {"step_count": state.get("step_count", 0)}

    llm = get_llm().bind_tools(AGENT_TOOLS)
    response = llm.invoke([
        SystemMessage(content=(
            "You are an RM copilot. Call at most ONE tool if it helps answer the request. "
            "Tools: portfolio_lookup(client_id) — only for IDs like C-204, C-301; "
            "market_data(symbols); suitability_checker(fund_or_product, client_risk_profile). "
            "For suitability questions (e.g. PG-003 for conservative), use suitability_checker, "
            "NOT portfolio_lookup."
        )),
        HumanMessage(content=state["query"]),
    ])

    trace = list(state.get("tools_trace", []))
    messages: list = [response]

    if response.tool_calls:
        for tc in response.tool_calls[:1]:
            tool = TOOL_MAP.get(tc["name"])
            if tool:
                try:
                    result = tool.invoke(tc["args"])
                except (ValueError, KeyError) as exc:
                    result = f"Tool error: {exc}"
                trace.append(tc["name"])
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )
                log_audit_event("tool_call", {"tool": tc["name"], "args": tc["args"]})

    return {
        "messages": messages,
        "tools_trace": trace,
        "step_count": state.get("step_count", 0) + 1,
    }


def gather_portfolio_node(state: AgentState) -> dict:
    """Fetch client portfolio and risk profile."""
    if _steps_exceeded(state):
        return {"step_count": state.get("step_count", 0)}

    client_id = state.get("client_id")
    if not client_id:
        return {
            "portfolio_summary": "No client ID specified in request.",
            "step_count": state.get("step_count", 0) + 1,
        }

    try:
        portfolio = load_client_portfolio(client_id)
        summary = get_portfolio_summary(portfolio)
        log_audit_event("gather_portfolio", {
            "client_id": client_id,
            "risk_profile": portfolio.risk_profile.value,
        })
        return {
            "portfolio": portfolio,
            "portfolio_summary": summary,
            "rm_entitlements": [e.value for e in portfolio.rm_entitlements],
            "step_count": state.get("step_count", 0) + 1,
            "messages": [AIMessage(content=f"Portfolio loaded for {client_id}")],
        }
    except ValueError as e:
        return {
            "portfolio_summary": str(e),
            "error": str(e),
            "step_count": state.get("step_count", 0) + 1,
        }


def gather_research_node(state: AgentState) -> dict:
    """Hybrid RAG retrieval over product/policy/research knowledge."""
    if _steps_exceeded(state):
        return {"step_count": state.get("step_count", 0)}

    query = state["query"]
    entitlements = [
        Sensitivity(e) for e in state.get("rm_entitlements", ["public", "internal"])
    ]

    use_multi_hop = any(
        kw in query.lower()
        for kw in ["complex", "compare", "review", "quarterly", "prepare"]
    )

    if use_multi_hop:
        chunks = retriever.multi_hop_retrieve(query, k=5, rm_entitlements=entitlements)
    else:
        chunks = retriever.retrieve(query, k=5, rm_entitlements=entitlements)

    context = retriever.format_chunks_for_context(chunks)
    log_audit_event("gather_research", {
        "query": query,
        "chunks_retrieved": len(chunks),
        "doc_ids": [c.doc_id for c in chunks],
        "entitlements": [e.value for e in entitlements],
    })

    return {
        "retrieved_chunks": chunks,
        "research_context": context,
        "step_count": state.get("step_count", 0) + 1,
        "messages": [AIMessage(content=f"Retrieved {len(chunks)} relevant documents")],
    }


def gather_market_node(state: AgentState) -> dict:
    """Fetch market data context."""
    if _steps_exceeded(state):
        return {"step_count": state.get("step_count", 0)}

    portfolio = state.get("portfolio")
    symbols = "SPY,AGG,VTI,PG-002,PG-003"
    if portfolio:
        symbols = ",".join(h.symbol for h in portfolio.holdings[:5])

    market_ctx = market_data.invoke({"symbols": symbols})
    log_audit_event("gather_market", {"symbols": symbols})
    return {
        "market_context": market_ctx,
        "step_count": state.get("step_count", 0) + 1,
    }


def check_suitability_node(state: AgentState) -> dict:
    """Validate ideas against suitability and compliance policy."""
    if _steps_exceeded(state):
        return {"step_count": state.get("step_count", 0)}

    query = state["query"]
    fund = _extract_fund(query)
    results: list[str] = []

    portfolio = state.get("portfolio")
    risk_profile = (
        portfolio.risk_profile
        if portfolio
        else (_extract_risk_profile(query) or RiskProfile.BALANCED)
    )

    if fund:
        result = check_suitability_logic(fund, risk_profile, retriever)
        results.append(
            f"{result.fund_or_product}: suitable={result.suitable}, "
            f"{result.assessment}"
        )
        log_audit_event("check_suitability", {
            "fund": fund,
            "risk_profile": risk_profile.value,
            "suitable": result.suitable,
        })
    elif "suitable" in query.lower():
        results.append(
            "No specific fund identified. Specify product code (e.g. PG-003) for suitability check."
        )

    return {
        "suitability_results": results,
        "step_count": state.get("step_count", 0) + 1,
        "messages": [AIMessage(content="Suitability check complete")],
    }


def synthesize_node(state: AgentState) -> dict:
    """Produce a grounded ClientBrief with citations using structured output."""
    if _steps_exceeded(state):
        return {"step_count": state.get("step_count", 0)}

    query = state["query"]
    portfolio = state.get("portfolio")
    chunks = state.get("retrieved_chunks", [])
    citations = chunks_to_citations(chunks)

    client_id = state.get("client_id") or "UNKNOWN"
    risk_profile = (
        portfolio.risk_profile
        if portfolio
        else (_extract_risk_profile(query) or RiskProfile.BALANCED)
    )
    portfolio_summary = state.get("portfolio_summary", "No portfolio data available.")
    research_context = state.get("research_context", "")
    market_context = state.get("market_context", "")
    suitability_results = "\n".join(state.get("suitability_results", []))

    llm = get_llm()
    structured_llm = llm.with_structured_output(ClientBrief)

    try:
        brief = structured_llm.invoke([
            SystemMessage(content=(
                "You are a wealth management RM copilot generating a ClientBrief.\n"
                "RULES:\n"
                "- Every recommendation MUST include citations from provided documents\n"
                "- Do NOT give personalized investment advice (specific trades, withdrawal rates)\n"
                "- Use only information from the provided context\n"
                f"- Available citations: {json.dumps(citations)}"
            )),
            HumanMessage(content=(
                f"Request: {query}\n"
                f"client_id: {client_id}\n"
                f"risk_profile: {risk_profile.value}\n\n"
                f"Portfolio:\n{portfolio_summary}\n\n"
                f"Research:\n{research_context}\n\n"
                f"Market:\n{market_context}\n\n"
                f"Suitability:\n{suitability_results}"
            )),
        ])
        if isinstance(brief, ClientBrief):
            for reco in brief.recommendations:
                if not reco.citations and citations:
                    reco.citations = citations[:2]
        else:
            raise ValueError("Unexpected structured output type")
    except Exception:
        brief = ClientBrief(
            client_id=client_id,
            risk_profile=risk_profile,
            portfolio_summary=portfolio_summary,
            recommendations=[
                Reco(
                    idea="Review portfolio alignment with stated risk profile",
                    rationale="Based on retrieved Horizon product and policy documents.",
                    suitability="Pending RM review",
                    citations=citations[:3],
                )
            ] if citations else [],
            compliance_status=ComplianceStatus.NEEDS_REVIEW if not citations else ComplianceStatus.CLEARED,
            talking_points=[
                "Review current asset allocation vs CMP-003 risk profile targets",
                "Discuss RN-001 market outlook for fixed income positioning",
                "Confirm PG-002 (HCIF) remains appropriate for conservative income goals",
            ] if citations else ["Insufficient retrieved context — escalate for review"],
        )

    log_audit_event("synthesize", {
        "client_id": brief.client_id,
        "num_recommendations": len(brief.recommendations),
        "num_talking_points": len(brief.talking_points),
    })

    return {
        "draft_brief": brief,
        "step_count": state.get("step_count", 0) + 1,
        "messages": [AIMessage(content="ClientBrief synthesized")],
    }


def review_gate_node(state: AgentState) -> dict:
    """Compliance gate with human-in-the-loop interrupt for escalation."""
    brief = state.get("draft_brief")
    if not brief:
        return {"final_brief": None, "step_count": state.get("step_count", 0) + 1}

    preliminary = apply_review_gate(
        brief,
        state["query"],
        requires_escalation=state.get("requires_escalation", False),
    )

    needs_human = (
        state.get("requires_escalation")
        or preliminary.compliance_status != ComplianceStatus.CLEARED
    )

    if needs_human:
        decision = interrupt({
            "type": "human_review",
            "compliance_status": preliminary.compliance_status.value,
            "escalation_reason": preliminary.escalation_reason,
            "brief": preliminary.model_dump(),
            "message": (
                "Human RM review required before client communication. "
                "Approve to release brief, or reject to escalate only."
            ),
        })
        if not decision.get("approved", False):
            preliminary.recommendations = []
            preliminary.talking_points = [
                "Human reviewer declined automated suggestions.",
                "Escalate to licensed advisor (Series 7/66) before client communication.",
            ]
            preliminary.compliance_status = ComplianceStatus.NEEDS_REVIEW

    log_audit_event("review_gate", {
        "compliance_status": preliminary.compliance_status.value,
        "escalation_reason": preliminary.escalation_reason,
        "human_approved": needs_human,
    })

    return {
        "final_brief": preliminary,
        "step_count": state.get("step_count", 0) + 1,
        "messages": [
            AIMessage(content=f"Review gate: {preliminary.compliance_status.value}")
        ],
    }
