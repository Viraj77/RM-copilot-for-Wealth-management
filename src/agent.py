from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from src.tools import portfolio_lookup, rag_search, suitability_checker
from src.schemas import ClientBrief
from src.system_prompt import SYSTEM_PROMPT
from src.guardrails import validate_output
import re

from dotenv import load_dotenv
import os

import json

from langchain_core.output_parsers import PydanticOutputParser
from src.schemas import ClientBrief
from src.hitl import (   create_review_ticket,     log_escalation)

from langgraph.graph import (    StateGraph,     END)


load_dotenv()

parser = PydanticOutputParser(pydantic_object=ClientBrief)
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
valid_context: bool

class AgentState(TypedDict):
    query:str
    client_id:str
    portfolio:dict
    research:list
    suitability:str
    response:str

def is_wealth_management_query(query: str):

    q = query.lower()

    wealth_keywords = [

        "portfolio",
        "investment",
        "fund",
        "bond",
        "equity",
        "wealth",
        "market",
        "risk",
        "asset",
        "allocation",
        "performance",
        "returns",
        "research",
        "mutual fund",
        "fixed income",
        "client",
        "advisor",
        "rm",
        "financial",
        "compliance",
        "suitability",
        "stock",
        "etf"
    ]

    score = sum(
        1 for k in wealth_keywords
        if k in q
    )

    return score >= 1

def detect_intent(query: str):

    q = query.lower()

    # =====================================
    # CLIENT REVIEW
    # =====================================

    if (
        "client" in q
        or "portfolio" in q
        or "talking points" in q
        or "quarterly review" in q
    ):

        return "client_review"

    # =====================================
    # PRODUCT / FUND
    # =====================================

    elif (
        "fund" in q
        or "bond" in q
        or "product" in q
        or "best performing" in q
    ):

        return "product"

    # =====================================
    # MARKET RESEARCH
    # =====================================

    elif (
        "market" in q
        or "outlook" in q
        or "sector" in q
        or "equity vs" in q
    ):

        return "research"

    # =====================================
    # COMPLIANCE
    # =====================================

    elif (
        "compliance" in q
        or "suitable" in q
        or "policy" in q
    ):

        return "compliance"

    return "generic"

def extract_client_id(query: str):

    match = re.search(r"C-\d+", query)

    if match:
        return match.group(0)

    return None

def gather_portfolio(state):

    client_id = extract_client_id(state["query"])

    state["client_id"] = client_id

    # Only lookup if client exists
    if client_id:
        state["portfolio"] = portfolio_lookup(client_id)
    else:
        state["portfolio"] = None
    return state
        
def gather_portfolio(state):
    
    client_id = extract_client_id(state["query"])

    state["client_id"] = client_id

    state["portfolio"] = portfolio_lookup(client_id)

    return state

def gather_research(state):

    state["research"] = rag_search(
        query=state["query"],
        client_id=state.get("client_id")
    )

    return state

def suitability(state):

    # Generic query
    if not state.get("client_id"):

        state["suitability"] = "Not Applicable"
        return state

    risk = state["portfolio"].get(
        "risk_profile",
        "Balanced"
    )

    state["suitability"] = suitability_checker(
        risk,
        state["query"]
    )

    return state


def synthesize(state):

    # ====================================================
    # HUMAN ESCALATION CHECK
    # ====================================================

    if requires_human_review(state["query"]):

        state["response"] = """
            Escalate to Human RM.

            This request may require licensed financial advice,
            manual suitability review, or additional compliance validation.

            Please route this request to an authorized
            relationship manager or investment advisor.
        """

        return state

    # ====================================================
    # DETECT USER INTENT
    # ====================================================

    intent = detect_intent(
        state["query"]
    )

    # ====================================================
    # CLIENT REVIEW WORKFLOW
    # ONLY HERE USE PYDANTIC PARSING
    # ====================================================

    if intent == "client_review":

        format_instructions = (
            parser.get_format_instructions()
        )

        prompt = f"""
    {SYSTEM_PROMPT}

    ====================================================
    CLIENT REVIEW RESPONSE REQUIREMENTS
    ====================================================

    1. Return ONLY valid JSON
    2. Do NOT generate markdown
    3. Do NOT add explanations outside JSON
    4. Generate detailed RM-ready insights
    5. Include:
    - portfolio analysis
    - investment observations
    - suitability commentary
    - recommendations
    - talking points
    - citations
    6. Responses should resemble:
    - private banking reviews
    - RM advisory summaries
    - investment committee notes
    7. Portfolio summary should contain
    at least 5-10 lines of meaningful analysis.

    {format_instructions}

    ====================================================
    USER QUERY
    ====================================================

    {state['query']}

    ====================================================
    PORTFOLIO
    ====================================================

    {state.get('portfolio', {})}

    ====================================================
    RESEARCH
    ====================================================

    {state.get('research', [])}

    ====================================================
    SUITABILITY
    ====================================================

    {state.get('suitability', '')}
    """         

        try:

            response = llm.invoke(prompt)

            raw_output = response.content

            # ============================================
            # PYDANTIC VALIDATION
            # ============================================

            validated_output = parser.parse(
                raw_output
            )

            # ============================================
            # FORMAT CLIENT BRIEF
            # ============================================

            formatted_text = format_client_brief(
                validated_output
            )

            state["response"] = formatted_text

        except Exception as e:

            state["response"] = f"""
    Client Brief Generation Failed.

    Validation Error:
    {str(e)}

    The LLM response could not be parsed
    into the required ClientBrief schema.

    Raw LLM Output:

    {raw_output if 'raw_output' in locals() else 'No Output Returned'}
    """                                 

    # ====================================================
    # PRODUCT / FUND / RESEARCH / GENERIC WORKFLOWS
    # NO PYDANTIC PARSING
    # ====================================================

    else:

        prompt = f"""
            {SYSTEM_PROMPT}

        ====================================================
        RESPONSE QUALITY REQUIREMENTS
        ====================================================

        Generate a professional enterprise-grade
        wealth-management response.

        IMPORTANT:

        1. Response MUST contain 5-10 meaningful lines minimum.
        2. Avoid extremely short responses.
        3. Include:
        - market context
        - investment rationale
        - opportunities
        - risks
        - suitability observations
        - practical RM considerations
        4. Use professional advisory language.
        5. Response should resemble:
        - wealth advisory commentary
        - institutional research summaries
        - RM investment guidance
        6. Keep response conversational but professional.
        7. Do NOT generate JSON.
        8. Do NOT generate markdown.
        9. Use uploaded research and policy documents whenever possible.
        10. Include citations/references where relevant.

        ====================================================
        USER QUERY
        ====================================================

        {state['query']}

        ====================================================
        RESEARCH
        ====================================================

        {state.get('research', [])}

        ====================================================
        SUITABILITY
        ====================================================

        {state.get('suitability', '')}
        """

        try:

            response = llm.invoke(prompt)

            raw_output = response.content

            # ============================================
            # FORMAT NON-CLIENT RESPONSES
            # ============================================

            if intent == "product":

                formatted_text = format_product_response(
                    raw_output
                )

            elif intent == "research":

                formatted_text = format_research_response(
                    raw_output
                )

            elif intent == "compliance":

                formatted_text = format_compliance_response(
                    raw_output
                )

            else:

                formatted_text = format_generic_response(
                    raw_output
                )

            state["response"] = formatted_text

        except Exception as e:

            state["response"] = f"""
        Response Generation Failed.

        Error:
        {str(e)}
"""

    return state

def format_client_brief(brief):

    output = f"""
══════════════════════════════════════════════════════
        WEALTH MANAGEMENT CLIENT REVIEW
══════════════════════════════════════════════════════

CLIENT OVERVIEW
──────────────────────────────────────────────────────

Client ID:
{brief.client_id}

Risk Profile:
{brief.risk_profile}


PORTFOLIO ANALYSIS
──────────────────────────────────────────────────────

{brief.portfolio_summary}


INVESTMENT OBSERVATIONS
──────────────────────────────────────────────────────

The current portfolio allocation reflects the
client's stated risk appetite and investment objectives.

Market conditions continue to emphasize diversification,
income stability, and disciplined risk management
across asset classes.


RECOMMENDATIONS
──────────────────────────────────────────────────────
"""

    for idx, reco in enumerate(
        brief.recommendations,
        start=1
    ):

        output += f"""

Recommendation {idx}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Investment Idea:
{reco.idea}

Business Rationale:
{reco.rationale}

Suitability Assessment:
{reco.suitability}

Supporting References:
{', '.join(reco.citations)}

"""

    output += """
RM TALKING POINTS
──────────────────────────────────────────────────────
"""

    for point in brief.talking_points:

        output += f"""
• {point}
"""

    output += f"""


COMPLIANCE REVIEW
──────────────────────────────────────────────────────

{brief.compliance_status}


IMPORTANT DISCLAIMER
──────────────────────────────────────────────────────

This output is generated for relationship-manager
decision-support purposes only and should not be
considered automated investment advice.

Licensed advisor review may be required depending
on suitability and compliance considerations.
"""

    return output


# ====================================================
# RESEARCH FORMATTER
# ====================================================

def format_research_response(text):

    return f"""
══════════════════════════════════════════════════════
              MARKET RESEARCH SUMMARY
══════════════════════════════════════════════════════

MARKET OVERVIEW
──────────────────────────────────────────────────────

{text}


KEY MARKET THEMES
──────────────────────────────────────────────────────

• Interest-rate expectations continue influencing
  portfolio positioning and fixed-income allocations

• Investors remain focused on diversification,
  quality exposure, and defensive positioning

• Current macroeconomic conditions favor disciplined
  risk management and balanced asset allocation


PORTFOLIO IMPLICATIONS
──────────────────────────────────────────────────────

• Fixed-income instruments may provide stability
  during volatile market environments

• Diversification across sectors and asset classes
  remains important for long-term resilience

• Duration and credit-quality exposure should align
  with client risk tolerance


RM TALKING POINTS
──────────────────────────────────────────────────────

• Discuss current market volatility and macro trends

• Review portfolio resilience under changing
  interest-rate environments

• Evaluate opportunities across fixed income,
  equities, and diversified strategies


DISCLAIMER
──────────────────────────────────────────────────────

Research insights are generated using uploaded
documents, retrieved investment commentary,
and available market research inputs.
"""


# ====================================================
# PRODUCT FORMATTER
# ====================================================

def format_product_response(text):

    return f"""
══════════════════════════════════════════════════════
               PRODUCT & FUND INSIGHTS
══════════════════════════════════════════════════════

EXECUTIVE SUMMARY
──────────────────────────────────────────────────────

{text}


KEY RM CONSIDERATIONS
──────────────────────────────────────────────────────

• Review alignment with client risk appetite

• Evaluate duration exposure and interest-rate sensitivity

• Assess credit quality and diversification benefits

• Validate liquidity profile and income objectives

• Ensure suitability before recommendation


RISK OBSERVATIONS
──────────────────────────────────────────────────────

• Bond funds remain sensitive to interest-rate changes

• Credit-spread widening may impact lower-quality issuers

• Liquidity conditions should be monitored during
  stressed market environments


RM NEXT STEPS
──────────────────────────────────────────────────────

• Compare performance against benchmark indices

• Review historical volatility and drawdown trends

• Validate suitability and compliance considerations

• Escalate complex recommendations where appropriate


DISCLAIMER
──────────────────────────────────────────────────────

This content is intended for wealth-management
decision-support purposes only and should not
be treated as automated financial advice.
"""


# ====================================================
# COMPLIANCE FORMATTER
# ====================================================

def format_compliance_response(text):

    return f"""
══════════════════════════════════════════════════════
          COMPLIANCE & SUITABILITY REVIEW
══════════════════════════════════════════════════════

COMPLIANCE SUMMARY
──────────────────────────────────────────────────────

{text}


SUITABILITY OBSERVATIONS
──────────────────────────────────────────────────────

• Recommendations must align with client
  investment objectives and risk tolerance

• High-risk allocations may require additional review

• Regulatory and internal-policy requirements
  should be validated before implementation


RM ACTION ITEMS
──────────────────────────────────────────────────────

• Confirm suitability documentation requirements

• Validate compliance-policy alignment

• Escalate restricted or high-risk products
  to authorized advisors where necessary


DISCLAIMER
──────────────────────────────────────────────────────

This output is intended for internal RM support
and does not constitute formal investment advice.
"""


# ====================================================
# GENERIC FORMATTER
# ====================================================

def format_generic_response(text):

    return f"""
══════════════════════════════════════════════════════
             WEALTH MANAGEMENT INSIGHTS
══════════════════════════════════════════════════════

{text}


ADDITIONAL OBSERVATIONS
──────────────────────────────────────────────────────

• Insights are generated using uploaded documents
  and retrieved financial knowledge sources

• Recommendations should be reviewed against
  suitability and compliance requirements

• Relationship managers should validate alignment
  with client investment objectives


DISCLAIMER
──────────────────────────────────────────────────────

Generated content is intended for RM decision-support
and operational-efficiency purposes only.
"""


def review_gate(state):

    if state["suitability"] == "Needs Review":

        state["response"] += """
            ------------------------------------
            HUMAN REVIEW REQUIRED
            ------------------------------------

            This recommendation requires review
            by a licensed advisor.
        """

    return state

def requires_human_review(query):

    q = query.lower()

    risky_patterns = [

        "all savings",
        "entire retirement",
        "guaranteed returns",
        "100 percent allocation",
        "high leverage",
        "tax avoidance",
        "move everything",
        "aggressive allocation",
        "life savings"
    ]

    return any(
        pattern in q
        for pattern in risky_patterns
    )

def human_review_node(state):

    # ============================================
    # CREATE REVIEW TICKET
    # ============================================

    ticket = create_review_ticket(
        state["query"]
    )

    # ============================================
    # LOG ESCALATION
    # ============================================

    log_escalation(ticket)

    # ============================================
    # STORE IN STATE
    # ============================================

    state["review_ticket"] = ticket

    # ============================================
    # RESPONSE
    # ============================================

    state["response"] = f"""
══════════════════════════════════════════════════════
                HUMAN REVIEW REQUIRED
══════════════════════════════════════════════════════

This request requires additional review
by a licensed relationship manager or
investment advisor before recommendations
can be provided.

REASON FOR ESCALATION
──────────────────────────────────────────────────────

• Request may involve regulated investment advice

• Additional suitability assessment may be required

• Compliance validation is needed before proceeding

• Portfolio allocation changes may carry elevated risk


REVIEW TICKET DETAILS
──────────────────────────────────────────────────────

Ticket ID:
{ticket.get('ticket_id', 'N/A')}

Priority:
{ticket['priority']}

Status:
{ticket['status']}


NEXT STEPS
──────────────────────────────────────────────────────

Please route this request to an
authorized wealth-management professional
for further review and suitability checks.


DISCLAIMER
──────────────────────────────────────────────────────

This AI assistant provides decision-support
guidance only and cannot independently issue
licensed investment recommendations.
"""

    return state



def route_human_review(state):

    if requires_human_review(
        state["query"]
    ):

        return "human_review"

    return "synthesize"



def context_guardrail(state):

    valid = is_wealth_management_query(
        state["query"]
    )

    state["valid_context"] = valid

    return state



def outside_context_response(state):

    prompt = f"""
    The user asked:

    {state['query']}

    This query is outside the supported
    wealth-management domain.

    Generate a professional response that:

    1. Politely explains the assistant specializes in:
       - wealth management
       - portfolio analysis
       - investments
       - market research
       - compliance
       - RM workflows

    2. Explain supported capabilities clearly.

    3. Suggest example supported queries.

    4. Keep response professional and user-friendly.

    5. Response should be at least 6-8 lines.

    6. Do NOT generate JSON.
    """

    response = llm.invoke(prompt)

    state["response"] = response.content

    return state



def build_graph():

    graph = StateGraph(AgentState)

    # ====================================================
    # ADD NODES
    # ====================================================

    graph.add_node(
        "context_guardrail",
        context_guardrail
    )

    graph.add_node(
        "outside_context",
        outside_context_response
    )

    graph.add_node(
        "portfolio",
        gather_portfolio
    )

    graph.add_node(
        "research",
        gather_research
    )

    # ✅ IMPORTANT
    graph.add_node(
        "suitability",
        suitability
    )

    graph.add_node(
        "human_review",
        human_review_node
    )

    graph.add_node(
        "synthesize",
        synthesize
    )

    graph.add_node(
        "review",
        review_gate
    )

    # ====================================================
    # ENTRY POINT
    # ====================================================

    graph.set_entry_point(
        "context_guardrail"
    )

    # ====================================================
    # CONTEXT ROUTING
    # ====================================================

    graph.add_conditional_edges(
        "context_guardrail",
        route_context
    )

    # ====================================================
    # OUTSIDE CONTEXT FLOW
    # ====================================================

    graph.add_edge(
        "outside_context",
        END
    )

    # ====================================================
    # MAIN FLOW
    # ====================================================

    graph.add_edge(
        "portfolio",
        "research"
    )

    graph.add_edge(
        "research",
        "suitability"
    )

    # ====================================================
    # HUMAN REVIEW ROUTING
    # ====================================================

    graph.add_conditional_edges(
        "suitability",
        route_human_review
    )

    # ====================================================
    # HUMAN REVIEW END
    # ====================================================

    graph.add_edge(
        "human_review",
        END
    )

    # ====================================================
    # NORMAL FLOW
    # ====================================================

    graph.add_edge(
        "synthesize",
        "review"
    )

    graph.add_edge(
        "review",
        END
    )

    # ====================================================
    # COMPILE GRAPH
    # ====================================================

    return graph.compile()





def route_context(state):

    if state["valid_context"]:
        return "portfolio"

    return "outside_context"


agent = build_graph()
