"""
Suitability Checker Tool — Phase 3: Tools.

Validates investment recommendations against:
  1. Hard-coded rule-based checks (fast, deterministic)
  2. LLM-guided suitability assessment grounded in policy chunks

Returns a SuitabilityVerdict with detailed reasoning and policy citations.
This is the compliance gateway before any recommendation reaches the ClientBrief.
"""

import json
import re
from typing import Any, Optional
import warnings

from config.settings import settings
from src.models.brief import SuitabilityVerdict

# ── Hard-coded suitability rules ──────────────────────────────────────────────
# These run BEFORE the LLM and can short-circuit with a definitive verdict.

# Keywords that always flag licensed-advice escalation
LICENSED_ADVICE_TRIGGERS = [
    "personalized", "personalised", "specific advice", "you should buy",
    "you should sell", "i recommend you", "guaranteed return", "guarantee",
    "certain return", "100% safe", "no risk",
]

# Asset class allocation limits by risk profile
ALLOCATION_LIMITS: dict[str, dict[str, float]] = {
    "conservative": {
        "equity": 25.0,
        "alternatives": 0.0,
        "high_yield": 0.0,
        "emerging_markets": 0.0,
    },
    "balanced": {
        "equity": 65.0,
        "alternatives": 15.0,
        "high_yield": 10.0,
        "emerging_markets": 0.0,
    },
    "growth": {
        "equity": 80.0,
        "alternatives": 25.0,
        "high_yield": 20.0,
        "emerging_markets": 20.0,
    },
    "aggressive": {
        "equity": 100.0,
        "alternatives": 40.0,
        "high_yield": 40.0,
        "emerging_markets": 35.0,
    },
}

# Products that are ALWAYS unsuitable for certain profiles
PROFILE_PROHIBITIONS: dict[str, list[str]] = {
    "conservative": [
        "high yield", "junk bond", "emerging market", "cryptocurrency", "crypto",
        "bitcoin", "digital asset", "alternative", "private credit", "hedge fund",
        "leveraged", "concentrated", "speculative", "arkk", "technology growth fund",
    ],
    "balanced": [
        "cryptocurrency", "crypto", "bitcoin", "digital asset", "private credit",
        "leveraged", "100% equity", "emerging market",
    ],
}


def _check_licensed_advice_triggers(recommendation: str) -> bool:
    """
    Return True if the recommendation text triggers licensed-advice escalation.

    Uses a negative lookbehind ``(?<!-)`` combined with a word-boundary ``\b``
    to avoid false positives on hyphenated prefixes like 'non-personalized'.
    The lookbehind ensures the trigger word is not immediately preceded by a
    hyphen (as in 'non-personalized'), while still catching standalone
    occurrences like 'personalized advice'.
    """
    lower = recommendation.lower()
    for trigger in LICENSED_ADVICE_TRIGGERS:
        # (?<!-) ensures we don't match when the trigger follows a hyphen
        # (e.g. "non-personalized" would not match "personalized")
        pattern = r"(?<!-)\b" + re.escape(trigger) + r"\b"
        if re.search(pattern, lower):
            return True
    return False


def _check_profile_prohibition(recommendation: str, risk_profile: str) -> Optional[str]:
    """
    Check if the recommendation violates a hard-coded profile prohibition.
    Returns the matched prohibited term, or None if compliant.
    """
    prohibitions = PROFILE_PROHIBITIONS.get(risk_profile.lower(), [])
    lower = recommendation.lower()
    for term in prohibitions:
        if term in lower:
            return term
    return None


# ── LLM-based suitability assessment ─────────────────────────────────────────

SUITABILITY_SYSTEM_PROMPT = """You are a compliance officer assessing whether an investment 
recommendation is suitable for a specific client based on their risk profile and applicable 
suitability policy.

Your assessment MUST:
1. Reference specific policy excerpts from the provided context
2. Produce a structured JSON verdict
3. Be conservative — when in doubt, flag for review rather than clearing
4. Never approve recommendations that violate hard policy rules

Respond ONLY with valid JSON in this exact format:
{
  "verdict": "<suitable|marginal|unsuitable|requires_licensed_advice>",
  "reasoning": "<detailed explanation referencing policy>",
  "policy_references": ["<excerpt 1>", "<excerpt 2>"],
  "requires_escalation": <true|false>
}"""

SUITABILITY_USER_TEMPLATE = """
CLIENT RISK PROFILE: {risk_profile}

RECOMMENDATION TO ASSESS:
{recommendation}

RELEVANT POLICY CONTEXT:
{policy_context}

Assess whether this recommendation is suitable for a client with a {risk_profile} risk profile.
Apply the policy context strictly. Return your assessment as JSON.
"""


def _llm_suitability_check(
    recommendation: str,
    risk_profile: str,
    policy_context: list[str],
) -> dict[str, Any]:
    """
    Use the LLM to perform a nuanced suitability assessment grounded in policy chunks.
    """
    try:
        from openai import OpenAI  # type: ignore

        api_key = settings.openai_api_key
        if not api_key:
            return _fallback_verdict(
                "LLM check skipped — OPENAI_API_KEY not configured.",
                SuitabilityVerdict.MARGINAL,
            )

        client = OpenAI(api_key=api_key)
        policy_text = "\n\n---\n\n".join(policy_context[:5])  # Limit to 5 chunks

        user_msg = SUITABILITY_USER_TEMPLATE.format(
            risk_profile=risk_profile,
            recommendation=recommendation,
            policy_context=policy_text,
        )

        response = client.chat.completions.create(
            model=settings.openai_model,
            temperature=settings.suitability_temperature,
            messages=[
                {"role": "system", "content": SUITABILITY_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            max_tokens=800,
        )

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)

        # Validate and normalise the verdict
        verdict_str = parsed.get("verdict", "marginal").lower()
        try:
            verdict = SuitabilityVerdict(verdict_str)
        except ValueError:
            verdict = SuitabilityVerdict.MARGINAL

        return {
            "verdict": verdict.value,
            "reasoning": parsed.get("reasoning", "No reasoning provided."),
            "policy_references": parsed.get("policy_references", []),
            "requires_escalation": parsed.get("requires_escalation", False),
            "source": "llm",
        }

    except Exception as exc:
        return _fallback_verdict(
            f"LLM suitability check failed: {str(exc)}. Defaulting to NEEDS_REVIEW.",
            SuitabilityVerdict.MARGINAL,
        )


def _fallback_verdict(reason: str, verdict: SuitabilityVerdict) -> dict[str, Any]:
    return {
        "verdict": verdict.value,
        "reasoning": reason,
        "policy_references": [],
        "requires_escalation": verdict in (
            SuitabilityVerdict.UNSUITABLE,
            SuitabilityVerdict.REQUIRES_LICENSED_ADVICE,
        ),
        "source": "fallback",
    }


# ── Core suitability checker ──────────────────────────────────────────────────

def check_suitability(
    recommendation: str,
    client_risk_profile: str,
    policy_context: Optional[list[str]] = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Validate a recommendation against suitability and compliance policy.

    Runs a two-stage check:
    1. Fast rule-based checks (instant, deterministic)
    2. LLM-guided assessment grounded in policy chunks (if enabled)

    Args:
        recommendation: The investment recommendation text to assess.
        client_risk_profile: Client's risk profile ("conservative", "balanced",
                             "growth", "aggressive").
        policy_context: List of policy document excerpts from RAG retrieval.
                        Used to ground the LLM assessment.
        use_llm: If True, run the LLM assessment (requires OpenAI API key).

    Returns:
        Dict with: verdict, reasoning, policy_references, requires_escalation,
        rule_violations, citations.
    """
    policy_context = policy_context or []
    risk_profile = client_risk_profile.lower().strip()

    rule_violations: list[str] = []

    # ── Stage 1: Licensed advice trigger check ────────────────────────────────
    if _check_licensed_advice_triggers(recommendation):
        return {
            "success": True,
            "verdict": SuitabilityVerdict.REQUIRES_LICENSED_ADVICE.value,
            "reasoning": (
                "This recommendation contains language that constitutes personalised "
                "investment advice requiring a financial advice licence. "
                "The RM is not authorised to provide this recommendation. "
                "Please escalate to a licensed financial advisor."
            ),
            "policy_references": [
                "General Compliance Framework §5.1: RMs are NOT authorised to provide "
                "personalised investment advice that requires a regulated financial advice licence."
            ],
            "requires_escalation": True,
            "rule_violations": ["licensed_advice_language_detected"],
            "stage": "rule_based",
        }

    # ── Stage 2: Profile prohibition check ────────────────────────────────────
    prohibited_term = _check_profile_prohibition(recommendation, risk_profile)
    if prohibited_term:
        violation_msg = (
            f"Recommendation mentions '{prohibited_term}' which is prohibited "
            f"for {risk_profile} clients per suitability policy."
        )
        rule_violations.append(violation_msg)

        # Get relevant policy excerpt for citation
        policy_cite = (
            f"Conservative Portfolio Policy §3: The following are NEVER suitable "
            f"for {risk_profile.title()} clients: {prohibited_term}."
        ) if risk_profile == "conservative" else (
            f"Aggressive Risk Suitability Rules §4: {prohibited_term} requires "
            f"licensed advisor approval for {risk_profile} clients."
        )

        return {
            "success": True,
            "verdict": SuitabilityVerdict.UNSUITABLE.value,
            "reasoning": violation_msg,
            "policy_references": [policy_cite],
            "requires_escalation": False,
            "rule_violations": rule_violations,
            "stage": "rule_based",
        }

    # ── Stage 3: LLM-guided assessment ────────────────────────────────────────
    if use_llm and policy_context:
        llm_result = _llm_suitability_check(
            recommendation=recommendation,
            risk_profile=risk_profile,
            policy_context=policy_context,
        )
        llm_result["success"] = True
        llm_result["rule_violations"] = rule_violations
        llm_result["stage"] = "llm"
        return llm_result

    # ── Stage 4: Fallback — no policy context or LLM disabled ─────────────────
    if not policy_context:
        return {
            "success": True,
            "verdict": SuitabilityVerdict.MARGINAL.value,
            "reasoning": (
                "No policy context provided for assessment. "
                "Please retrieve relevant policy chunks via rag_retriever first. "
                "Marking as MARGINAL pending full policy review."
            ),
            "policy_references": [],
            "requires_escalation": False,
            "rule_violations": [],
            "stage": "fallback_no_context",
        }

    # Passed all checks without LLM
    return {
        "success": True,
        "verdict": SuitabilityVerdict.SUITABLE.value,
        "reasoning": (
            f"Recommendation passed all rule-based suitability checks for "
            f"a {risk_profile} client. No prohibited terms detected."
        ),
        "policy_references": [],
        "requires_escalation": False,
        "rule_violations": [],
        "stage": "rule_based_pass",
    }


# ── LangChain Tool wrapper ────────────────────────────────────────────────────

def get_suitability_checker_tool():
    """Return a LangChain-compatible suitability checker tool."""
    try:
        from langchain_core.tools import tool  # type: ignore

        @tool
        def suitability_checker_tool(
            recommendation: str,
            client_risk_profile: str,
            policy_context: str = "",
        ) -> str:
            """
            Validate an investment recommendation against suitability and compliance policy.

            ALWAYS use this tool before including any recommendation in the ClientBrief.
            A recommendation must receive a 'suitable' or 'marginal' verdict before being
            surfaced to the RM.

            Args:
                recommendation: The recommendation text to check (e.g. 'Add 10% allocation
                                to High Yield Corporate Bond Fund').
                client_risk_profile: Client's risk profile — one of 'conservative',
                                     'balanced', 'growth', 'aggressive'.
                policy_context: JSON-encoded list of relevant policy excerpt strings
                                retrieved from the knowledge base, e.g.
                                '["Policy excerpt 1", "Policy excerpt 2"]'.
                                Leave empty if not yet retrieved (will return MARGINAL verdict).
                                Pipe-separated format is also accepted for backward compatibility.

            Returns:
                JSON string with verdict, reasoning, policy_references, and
                requires_escalation flag.
            """
            policy_chunks: list[str] = []
            if policy_context and policy_context.strip():
                stripped = policy_context.strip()
                # Primary: JSON array format
                if stripped.startswith("["):
                    try:
                        parsed = json.loads(stripped)
                        policy_chunks = [str(c).strip() for c in parsed if str(c).strip()]
                    except json.JSONDecodeError:
                        # Malformed JSON — fall through to pipe fallback
                        policy_chunks = []
                # Fallback: legacy pipe-separated format
                if not policy_chunks:
                    if "|" in stripped:
                        warnings.warn(
                            "policy_context pipe-separated format is deprecated. "
                            "Use a JSON array string instead.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                    policy_chunks = [c.strip() for c in stripped.split("|") if c.strip()]

            result = check_suitability(
                recommendation=recommendation,
                client_risk_profile=client_risk_profile,
                policy_context=policy_chunks,
                use_llm=bool(settings.openai_api_key),
            )
            return json.dumps(result, indent=2, default=str)

        return suitability_checker_tool

    except ImportError:
        raise ImportError(
            "langchain-core is required. Install with: pip install langchain-core"
        )
