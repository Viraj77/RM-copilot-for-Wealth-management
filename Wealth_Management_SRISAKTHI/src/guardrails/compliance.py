"""Compliance guardrails and review gate."""

import re

from src.models import ClientBrief, ComplianceStatus, Reco, RiskProfile

LICENSED_ADVICE_PATTERNS = [
    r"\b\d+\.?\d*%\s*(withdrawal|withdraw)\b",
    r"\bbuy\s+\d+\s+shares\b",
    r"\bsell\s+(all|your)\b",
    r"\brebalance\s+to\s+\d+%\b",
    r"\bpersonalized\s+(investment\s+)?advice\b",
    r"\bspecific\s+(trade|order|allocation)\b",
    r"\bclaim\s+social\s+security\b",
    r"\bannuity\s+recommendation\b",
    r"\btax[\-\s]?loss\s+harvest",
]

ESCALATION_KEYWORDS = [
    "withdrawal rate",
    "personalized investment advice",
    "specific trade",
    "rebalance to",
    "social security claiming",
    "annuity recommendation",
    "tax-loss harvest",
    "licensed advisor",
    "series 7",
]


def detect_licensed_advice_request(query: str) -> bool:
    """Detect if the user query requires licensed investment advice."""
    query_lower = query.lower()
    for keyword in ESCALATION_KEYWORDS:
        if keyword in query_lower:
            return True
    return False


def detect_licensed_advice_in_content(text: str) -> bool:
    """Detect licensed advice patterns in generated content."""
    for pattern in LICENSED_ADVICE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def validate_recommendations(
    recommendations: list[Reco],
    client_risk_profile: RiskProfile,
) -> tuple[ComplianceStatus, str | None]:
    """Validate recommendations against compliance rules."""
    from src.tools.suitability import check_suitability_logic

    for reco in recommendations:
        if detect_licensed_advice_in_content(reco.idea + " " + reco.rationale):
            return ComplianceStatus.NEEDS_REVIEW, (
                "Recommendation contains language requiring licensed advisor sign-off"
            )

        if not reco.citations:
            return ComplianceStatus.NEEDS_REVIEW, (
                f"Recommendation '{reco.idea[:50]}...' lacks required citations"
            )

        for fund in ["PG-002", "PG-003", "PG-004", "HAEF", "HCIF"]:
            if fund in reco.idea.upper():
                result = check_suitability_logic(fund, client_risk_profile)
                if not result.suitable and result.blocked_reason:
                    return ComplianceStatus.BLOCKED, result.blocked_reason
                if result.requires_licensed_advice:
                    return ComplianceStatus.NEEDS_REVIEW, (
                        f"{fund} recommendation requires licensed advisor review"
                    )

    return ComplianceStatus.CLEARED, None


def apply_review_gate(
    brief: ClientBrief,
    query: str,
    requires_escalation: bool = False,
) -> ClientBrief:
    """Apply compliance review gate to a ClientBrief."""
    if requires_escalation or detect_licensed_advice_request(query):
        brief.compliance_status = ComplianceStatus.NEEDS_REVIEW
        brief.escalation_reason = (
            "Request involves personalized investment advice requiring "
            "licensed advisor (Series 7/66) sign-off. Escalated to human review."
        )
        brief.recommendations = []
        brief.talking_points = [
            "This request requires licensed financial planning advice.",
            "Escalate to the Wealth Planning team for personalized analysis.",
            "RM may discuss general educational concepts only.",
            "Do not provide specific withdrawal rates or trade instructions.",
        ]
        return brief

    status, reason = validate_recommendations(
        brief.recommendations, brief.risk_profile
    )
    brief.compliance_status = status
    if reason:
        if status == ComplianceStatus.BLOCKED:
            brief.escalation_reason = reason
            brief.recommendations = [
                r for r in brief.recommendations
                if "blocked" not in r.suitability.lower()
            ]
        elif status == ComplianceStatus.NEEDS_REVIEW:
            brief.escalation_reason = reason

    return brief
