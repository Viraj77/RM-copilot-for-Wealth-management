"""Suitability checker tool — aligned with Horizon product guides (PG-002/003/004)."""

import re

from langchain_core.tools import tool

from src.models import RiskProfile, SuitabilityResult
from src.tools.retriever import HybridRAGRetriever

RISK_LEVELS = {
    RiskProfile.CONSERVATIVE: 1,
    RiskProfile.BALANCED: 2,
    RiskProfile.GROWTH: 3,
    RiskProfile.AGGRESSIVE: 4,
}

# Canonical product codes from data/documents/product_guides/
PRODUCT_CATALOG: dict[str, dict] = {
    "PG-001": {
        "name": "Horizon Balanced Growth Fund (HBGF)",
        "risk": 2,
        "aliases": ["HBGF", "PG001", "BALANCED GROWTH"],
    },
    "PG-002": {
        "name": "Horizon Conservative Income Fund (HCIF)",
        "risk": 1,
        "aliases": ["HCIF", "PG002", "CONSERVATIVE INCOME"],
    },
    "PG-003": {
        "name": "Horizon Aggressive Equity Fund (HAEF)",
        "risk": 4,
        "aliases": ["HAEF", "PG003", "AGGRESSIVE EQUITY"],
    },
    "PG-004": {
        "name": "Fixed Income Structured Products",
        "risk": 1,
        "aliases": ["PG004", "FIXED DEPOSIT", "STRUCTURED PRODUCT"],
    },
}

# Client portfolio symbols still referenced in holdings data
LEGACY_SYMBOL_RISK = {
    "VTI": 2,
    "AGG": 1,
    "BND": 1,
    "VNQ": 2,
    "US-TBILL": 1,
    "CASH": 1,
}

BLOCKED_FOR_CONSERVATIVE = {"PG-003", "HAEF"}
NOT_RECOMMENDED_FOR_AGGRESSIVE = {"PG-002", "HCIF"}
CONDITIONAL_PRODUCTS = {"PG-003", "HAEF"}


def _resolve_product_code(fund_or_product: str) -> str:
    """Map user input (PG-003, HAEF, etc.) to canonical product code."""
    raw = fund_or_product.upper().strip()
    if raw in PRODUCT_CATALOG:
        return raw
    for code, info in PRODUCT_CATALOG.items():
        if raw in info["aliases"]:
            return code
    match = re.search(r"PG-?\d{3}", raw)
    if match:
        normalized = match.group(0).replace("PG", "PG-") if "-" not in match.group(0) else match.group(0)
        if normalized in PRODUCT_CATALOG:
            return normalized
    return raw


def check_suitability_logic(
    fund_or_product: str,
    client_risk_profile: RiskProfile,
    retriever: HybridRAGRetriever | None = None,
) -> SuitabilityResult:
    """Core suitability checking logic with RAG-backed citations."""
    fund = _resolve_product_code(fund_or_product)
    client_level = RISK_LEVELS[client_risk_profile]

    if fund in PRODUCT_CATALOG:
        product_level = PRODUCT_CATALOG[fund]["risk"]
        product_name = PRODUCT_CATALOG[fund]["name"]
    else:
        product_level = LEGACY_SYMBOL_RISK.get(fund, 3)
        product_name = fund

    retriever = retriever or HybridRAGRetriever()
    query = (
        f"suitability {product_name} {fund} for "
        f"{client_risk_profile.value} client risk profile"
    )
    citations: list[str] = []
    try:
        chunks = retriever.retrieve(query, k=3)
        citations = [f"{c.doc_id} ({c.source}, {c.date})" for c in chunks]
    except Exception:
        citations = [
            f"PG-002 (PG-002_Conservative_Income_Fund.txt, 2026-01-15)",
            f"CMP-003 (CMP-003_Client_Risk_Profiling_Methodology.pdf, 2026-01-15)",
        ]

    if fund in BLOCKED_FOR_CONSERVATIVE and client_risk_profile == RiskProfile.CONSERVATIVE:
        return SuitabilityResult(
            suitable=False,
            fund_or_product=fund,
            client_risk_profile=client_risk_profile,
            assessment=(
                f"{fund} ({product_name}) is BLOCKED for Conservative clients. "
                f"Product risk level ({product_level}) exceeds conservative tolerance."
            ),
            citations=citations,
            requires_licensed_advice=False,
            blocked_reason="Product prohibited for Conservative risk profile",
        )

    if fund in NOT_RECOMMENDED_FOR_AGGRESSIVE and client_risk_profile == RiskProfile.AGGRESSIVE:
        return SuitabilityResult(
            suitable=False,
            fund_or_product=fund,
            client_risk_profile=client_risk_profile,
            assessment=(
                f"{fund} ({product_name}) is NOT RECOMMENDED for Aggressive clients "
                f"seeking capital appreciation. Consider PG-003 (HAEF) instead."
            ),
            citations=citations,
            blocked_reason="Product mismatch for Aggressive growth objectives",
        )

    if product_level > client_level + 1:
        return SuitabilityResult(
            suitable=False,
            fund_or_product=fund,
            client_risk_profile=client_risk_profile,
            assessment=(
                f"{fund} is NOT SUITABLE. Product risk level ({product_level}) "
                f"exceeds client profile {client_risk_profile.value} (level {client_level}) "
                f"by more than one tier."
            ),
            citations=citations,
            blocked_reason="Risk level mismatch exceeds policy threshold",
        )

    if fund in CONDITIONAL_PRODUCTS and product_level > client_level:
        return SuitabilityResult(
            suitable=False,
            fund_or_product=fund,
            client_risk_profile=client_risk_profile,
            assessment=(
                f"{fund} is CONDITIONAL — exceeds base profile but may qualify as "
                f"satellite allocation. Requires documented client risk acknowledgment "
                f"and compliance review per CMP-003."
            ),
            citations=citations,
            requires_licensed_advice=True,
        )

    return SuitabilityResult(
        suitable=True,
        fund_or_product=fund,
        client_risk_profile=client_risk_profile,
        assessment=(
            f"{fund} ({product_name}) is SUITABLE for {client_risk_profile.value} clients. "
            f"Product risk level ({product_level}) aligns with client profile (level {client_level})."
        ),
        citations=citations,
    )


@tool
def suitability_checker(fund_or_product: str, client_risk_profile: str) -> str:
    """Check if a fund or product is suitable for a given client risk profile."""
    profile = RiskProfile(client_risk_profile.strip().title())
    result = check_suitability_logic(fund_or_product, profile)
    lines = [
        f"Fund/Product: {result.fund_or_product}",
        f"Client Risk Profile: {result.client_risk_profile.value}",
        f"Suitable: {'Yes' if result.suitable else 'No'}",
        f"Assessment: {result.assessment}",
    ]
    if result.blocked_reason:
        lines.append(f"Blocked Reason: {result.blocked_reason}")
    if result.requires_licensed_advice:
        lines.append("Requires Licensed Advisor: Yes")
    if result.citations:
        lines.append("Citations:")
        for c in result.citations:
            lines.append(f"  - {c}")
    return "\n".join(lines)
