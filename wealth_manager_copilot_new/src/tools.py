"""
Tools for the Wealth Manager Agent
"""
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import random

logger = logging.getLogger(__name__)


class PortfolioLookupTool:
    """
    Tool for fetching client portfolio information from database.
    In production, this would query a real portfolio management system.
    """
    
    # Mock database of client portfolios — aligned with Horizon product shelf
    # (PG-001 HBGF, PG-002 HCIF, PG-003 HAEF, PG-004 FDs and Bond Ladders)
    MOCK_PORTFOLIOS = {
        "C-201": {
            "client_id": "C-201",
            "client_name": "John Smith",
            "risk_profile": "Balanced",
            "total_value": 500000,
            "last_updated": "2026-06-01",
            "holdings": [
                {"ticker": "HBGF",       "name": "Horizon Balanced Growth Fund",          "value": 200000, "allocation": 0.40},
                {"ticker": "FD-USD-012", "name": "Fixed Deposit 12-Month USD (4.31%)",    "value": 100000, "allocation": 0.20},
                {"ticker": "BL-SHORT-AA","name": "Bond Ladder Short 2Y AA Agency",        "value": 100000, "allocation": 0.20},
                {"ticker": "HCIF",       "name": "Horizon Conservative Income Fund",      "value":  75000, "allocation": 0.15},
                {"ticker": "BL-ULTRAS-A","name": "Bond Ladder Ultra-Short 1Y A Corp",    "value":  25000, "allocation": 0.05},
            ],
            "allocation": {
                "equities": 0.40,
                "fixed_income": 0.55,
                "cash_equivalents": 0.05
            },
            "risk_score": 5.5
        },
        "C-202": {
            "client_id": "C-202",
            "client_name": "Jane Doe",
            "risk_profile": "Conservative",
            "total_value": 750000,
            "last_updated": "2026-06-01",
            "holdings": [
                {"ticker": "HCIF",        "name": "Horizon Conservative Income Fund",      "value": 375000, "allocation": 0.50},
                {"ticker": "FD-USD-024",  "name": "Fixed Deposit 24-Month USD (4.39%)",   "value": 150000, "allocation": 0.20},
                {"ticker": "FD-GBP-012",  "name": "Fixed Deposit 12-Month GBP (4.57%)",  "value": 112500, "allocation": 0.15},
                {"ticker": "BL-ULTRAS-AAA","name": "Bond Ladder Ultra-Short 1Y AAA Govt","value":  75000, "allocation": 0.10},
                {"ticker": "BL-SHORT-AAA","name": "Bond Ladder Short 2Y AAA Govt",        "value":  37500, "allocation": 0.05},
            ],
            "allocation": {
                "equities": 0.10,
                "fixed_income": 0.82,
                "cash_equivalents": 0.08
            },
            "risk_score": 2.8
        },
        "C-203": {
            "client_id": "C-203",
            "client_name": "Robert Johnson",
            "risk_profile": "Aggressive",
            "total_value": 1000000,
            "last_updated": "2026-06-01",
            "holdings": [
                {"ticker": "HAEF",        "name": "Horizon Aggressive Equity Fund",        "value": 500000, "allocation": 0.50},
                {"ticker": "HBGF",        "name": "Horizon Balanced Growth Fund",          "value": 200000, "allocation": 0.20},
                {"ticker": "BL-ULTRAS-BBB","name": "Bond Ladder Ultra-Short 1Y BBB Corp", "value": 150000, "allocation": 0.15},
                {"ticker": "FD-AUD-036",  "name": "Fixed Deposit 36-Month AUD (4.70%)",  "value": 100000, "allocation": 0.10},
                {"ticker": "BL-SHORT-BBB","name": "Bond Ladder Short 2Y BBB Corp",        "value":  50000, "allocation": 0.05},
            ],
            "allocation": {
                "equities": 0.70,
                "fixed_income": 0.25,
                "alternatives": 0.05
            },
            "risk_score": 8.2
        },
        "C-204": {
            "client_id": "C-204",
            "client_name": "Sarah Wilson",
            "risk_profile": "Growth",
            "total_value": 600000,
            "last_updated": "2026-06-01",
            "holdings": [
                {"ticker": "HBGF",        "name": "Horizon Balanced Growth Fund",          "value": 240000, "allocation": 0.40},
                {"ticker": "HAEF",        "name": "Horizon Aggressive Equity Fund",        "value": 150000, "allocation": 0.25},
                {"ticker": "BL-SHORTM-A","name": "Bond Ladder Short-Medium 3Y A Corp",    "value": 120000, "allocation": 0.20},
                {"ticker": "FD-SGD-018",  "name": "Fixed Deposit 18-Month SGD (3.33%)",  "value":  60000, "allocation": 0.10},
                {"ticker": "BL-SHORT-BBB","name": "Bond Ladder Short 2Y BBB Corp",        "value":  30000, "allocation": 0.05},
            ],
            "allocation": {
                "equities": 0.65,
                "fixed_income": 0.30,
                "cash_equivalents": 0.05
            },
            "risk_score": 6.8
        }
    }
    
    def __call__(self, client_id: str) -> Dict[str, Any]:
        """
        Fetch portfolio for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Portfolio data
        """
        portfolio = self.MOCK_PORTFOLIOS.get(client_id)
        
        if portfolio:
            logger.info(f"Portfolio lookup for {client_id}: {portfolio['total_value']}")
            return {
                "success": True,
                "data": portfolio
            }
        else:
            logger.warning(f"Portfolio not found for {client_id}")
            return {
                "success": False,
                "error": f"Portfolio not found for client {client_id}"
            }


class MarketDataTool:
    """
    Tool for fetching current market data and economic indicators.
    In production, would connect to market data APIs.
    """
    
    # Horizon fund NAVs + key fixed income products + Q2-2026 macro
    # Sources: PG-001 HBGF, PG-002 HCIF, PG-003 HAEF, PG-004 FDs/Bond Ladders,
    #          RN-001 Q2 2026 Market Outlook, RN-002 Tech/AI Sector Deep Dive
    MOCK_MARKET_DATA = {
        # Horizon Funds (illustrative NAVs)
        "HBGF":  {"price": 142.85, "change": 0.6,  "nav_date": "2026-06-20", "fund": "Horizon Balanced Growth Fund",        "risk_rating": 3},
        "HCIF":  {"price": 108.42, "change": 0.1,  "nav_date": "2026-06-20", "fund": "Horizon Conservative Income Fund",     "risk_rating": 1},
        "HAEF":  {"price": 231.10, "change": 1.4,  "nav_date": "2026-06-20", "fund": "Horizon Aggressive Equity Fund",       "risk_rating": 5},
        # Fixed Deposits — indicative rates from PG-004
        "FD-USD-012": {"rate_pct": 4.31, "tenor_months": 12, "currency": "USD", "min_investment": 1000, "capital_protection": "Full"},
        "FD-USD-024": {"rate_pct": 4.39, "tenor_months": 24, "currency": "USD", "min_investment": 1000, "capital_protection": "Full"},
        "FD-GBP-012": {"rate_pct": 4.57, "tenor_months": 12, "currency": "GBP", "min_investment": 1000, "capital_protection": "Full"},
        "FD-AUD-036": {"rate_pct": 4.70, "tenor_months": 36, "currency": "AUD", "min_investment": 1000, "capital_protection": "Full"},
        "FD-SGD-018": {"rate_pct": 3.33, "tenor_months": 18, "currency": "SGD", "min_investment": 1000, "capital_protection": "Full"},
        # Bond Ladder Rungs — indicative yields from PG-004
        "BL-ULTRAS-AAA": {"yield_pct": 4.55, "tenor_months": 12, "credit_rating": "AAA", "risk_rating": 2, "liquidity": "T+2 secondary"},
        "BL-ULTRAS-AA":  {"yield_pct": 4.84, "tenor_months": 12, "credit_rating": "AA",  "risk_rating": 2, "liquidity": "T+2 secondary"},
        "BL-ULTRAS-A":   {"yield_pct": 5.20, "tenor_months": 12, "credit_rating": "A",   "risk_rating": 3, "liquidity": "T+2 secondary"},
        "BL-ULTRAS-BBB": {"yield_pct": 5.80, "tenor_months": 12, "credit_rating": "BBB", "risk_rating": 4, "liquidity": "T+2 secondary"},
        "BL-SHORT-AAA":  {"yield_pct": 4.65, "tenor_months": 24, "credit_rating": "AAA", "risk_rating": 2, "liquidity": "T+2 secondary"},
        "BL-SHORT-AA":   {"yield_pct": 4.92, "tenor_months": 24, "credit_rating": "AA",  "risk_rating": 2, "liquidity": "T+2 secondary"},
        "BL-SHORT-BBB":  {"yield_pct": 5.81, "tenor_months": 24, "credit_rating": "BBB", "risk_rating": 4, "liquidity": "T+2 secondary"},
        "BL-SHORTM-A":   {"yield_pct": 5.36, "tenor_months": 36, "credit_rating": "A",   "risk_rating": 3, "liquidity": "T+2 secondary"},
    }

    # Q2 2026 macro indicators — sourced from RN-001 Q2 2026 Market Outlook
    # and RN-002 Tech/AI Sector Deep Dive context
    ECONOMIC_INDICATORS = {
        "gdp_growth": "2.1% (Q1 2026 annualised, moderating)",
        "inflation_rate": "2.8% (CPI YoY, trending toward target)",
        "unemployment_rate": "4.1%",
        "federal_funds_rate": "4.75%-5.00% (post Q1 2026 rate-volatility episode)",
        "10yr_yield": "4.35%",
        "vix_index": 16.2,
        "market_sentiment": "cautiously constructive — balanced growth environment post Q1 volatility",
        "tech_ai_view": "constructive but selective — cohort (a) compute/infra and (b) enterprise AI software preferred; cohort (c) early-stage AI-native unsuitable for Conservative/Balanced",
        "fixed_income_view": "duration discipline favoured; 3.5-5.5 yr band optimal for Conservative; BBB floor for credit quality",
        "em_view": "selective EM exposure viable for Growth/Aggressive; single-country EM cap 15% per CMP-002 RULE-006",
    }
    
    def __call__(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch market data and economic indicators.
        
        Args:
            ticker: Optional ticker symbol for specific security
            
        Returns:
            Market data
        """
        if ticker and ticker in self.MOCK_MARKET_DATA:
            logger.info(f"Market data for {ticker}")
            return {
                "success": True,
                "ticker": ticker,
                "data": self.MOCK_MARKET_DATA[ticker]
            }
        else:
            logger.info("Economic indicators")
            return {
                "success": True,
                "economic_indicators": self.ECONOMIC_INDICATORS,
                "timestamp": datetime.utcnow().isoformat()
            }


class SuitabilityCheckerTool:
    """
    Tool that validates recommendations against compliance rules.
    When a retriever is supplied the tool also queries the knowledge base
    for policy excerpts relevant to the risk profile and product type,
    surfaces them as citations, and uses them to detect additional
    violations beyond the hard-coded baseline rules.
    """

    def __init__(self, retriever=None):
        self.retriever = retriever

    # Suitability rules derived from CMP-002 Restricted List & Entitlement Rules
    # and PG-001/002/003/004 product suitability guidance.
    # Keys match RiskProfile enum values in models.py.
    SUITABILITY_RULES = {
        "Conservative": {
            # Asset allocation caps — based on HCIF mandate (75-90% debt, 10-25% equity)
            "max_equity_allocation": 0.25,
            "min_bond_allocation": 0.70,
            "max_volatility": 0.08,
            "max_single_equity_pct": 0.10,       # RULE-001
            "max_single_sector_equity_pct": 0.25, # RULE-002
            "max_alternative_pct": 0.00,           # RULE-008: not permitted
            "max_illiquid_pct": 0.00,              # RULE-005: Growth/Aggressive only
            "max_single_country_em_pct": 0.00,    # RULE-021: requires Balanced+
            "min_cash_pct": 0.02,                  # RULE-015
            # Completely restricted product types (CMP-002 Product Gate rules)
            "restricted_products": [
                "leveraged_etf", "leveraged_etp", "inverse_etp",   # RULE-017
                "private_equity", "private_placement",             # RULE-018, RULE-009
                "hedge_fund",                                       # RULE-019
                "cryptocurrency", "crypto_linked",                  # RULE-020
                "structured_capital_note", "scn",                   # RULE-016
                "volatility_linked_note",                           # RULE-024
                "perpetual_bond", "coco", "contingent_convertible", # RULE-025
                "margin_lending",                                   # RULE-026
                "commodity_futures",                                # RULE-027
                "long_dated_bond_20y",                              # RULE-028
                "high_yield_concentrated",                          # RULE-022
                "em_local_currency_debt",                           # RULE-021
                "single_stock_mandate",                             # RULE-023
                "derivatives",
            ],
            # Approved Horizon products
            "approved_products": ["HCIF", "HBGF", "FD-USD", "FD-EUR", "FD-GBP", "FD-SGD",
                                   "FD-AUD", "FD-CAD", "BL-ULTRAS-AAA", "BL-ULTRAS-AA",
                                   "BL-SHORT-AAA", "BL-SHORT-AA", "BL-SHORTM-AAA", "BL-SHORTM-AA"],
        },
        "Balanced": {
            "max_equity_allocation": 0.60,
            "min_bond_allocation": 0.30,
            "max_volatility": 0.12,
            "max_single_equity_pct": 0.10,        # RULE-001
            "max_single_sector_equity_pct": 0.25, # RULE-002
            "max_alternative_pct": 0.15,           # RULE-008
            "max_illiquid_pct": 0.00,              # RULE-005: Growth/Aggressive only
            "max_single_country_em_pct": 0.15,    # RULE-006
            "min_cash_pct": 0.02,                  # RULE-015
            "restricted_products": [
                "leveraged_etf", "leveraged_etp", "inverse_etp",   # RULE-017
                "private_equity", "private_placement",             # RULE-018
                "cryptocurrency", "crypto_linked",                  # RULE-020
                "structured_capital_note", "scn",                   # RULE-016
                "volatility_linked_note",                           # RULE-024
                "perpetual_bond", "coco", "contingent_convertible", # RULE-025
                "margin_lending",                                   # RULE-026
                "commodity_futures",                                # RULE-027
                "single_stock_mandate",                             # RULE-023
            ],
            "approved_products": ["HCIF", "HBGF", "FD-USD", "FD-EUR", "FD-GBP", "FD-SGD",
                                   "FD-AUD", "FD-CAD", "BL-ULTRAS-AAA", "BL-ULTRAS-AA",
                                   "BL-ULTRAS-A", "BL-SHORT-AAA", "BL-SHORT-AA", "BL-SHORT-A",
                                   "BL-SHORTM-AAA", "BL-SHORTM-AA", "BL-SHORTM-A"],
        },
        "Growth": {
            "max_equity_allocation": 0.80,
            "min_bond_allocation": 0.10,
            "max_volatility": 0.18,
            "max_single_equity_pct": 0.10,        # RULE-001
            "max_single_sector_equity_pct": 0.25, # RULE-002
            "max_alternative_pct": 0.15,           # RULE-008
            "max_illiquid_pct": 0.20,              # RULE-005
            "max_single_country_em_pct": 0.15,    # RULE-006
            "min_cash_pct": 0.02,                  # RULE-015
            "restricted_products": [
                "leveraged_etf", "leveraged_etp", "inverse_etp",   # RULE-017
                "private_equity",                                   # RULE-018: Aggressive only
                "cryptocurrency", "crypto_linked",                  # RULE-020
                "volatility_linked_note",                           # RULE-024
                "margin_lending",                                   # RULE-026
            ],
            "approved_products": ["HBGF", "HAEF", "HCIF", "FD-USD", "FD-EUR", "FD-GBP",
                                   "BL-ULTRAS-A", "BL-ULTRAS-BBB", "BL-SHORT-A", "BL-SHORT-BBB",
                                   "BL-SHORTM-A"],
        },
        "Aggressive": {
            "max_equity_allocation": 1.0,
            "min_bond_allocation": 0.00,
            "max_volatility": None,
            "max_single_equity_pct": 0.10,        # RULE-001 (override requires DH2)
            "max_single_sector_equity_pct": 0.25, # RULE-002
            "max_alternative_pct": 0.25,           # RULE-008
            "max_illiquid_pct": 0.20,              # RULE-005
            "max_single_country_em_pct": 0.15,    # RULE-006
            "min_cash_pct": 0.00,
            # Leveraged/complex products require attestations (not outright blocked)
            "restricted_products": [
                "cryptocurrency",  # RULE-020: pending product committee review even for Aggressive
            ],
            "requires_attestation": [
                "leveraged_etf", "leveraged_etp", "inverse_etp",   # RULE-017: derivatives attestation
                "private_equity", "private_placement",             # RULE-018/009: accredited investor
                "margin_lending",                                   # RULE-026: margin risk disclosure
            ],
            "approved_products": ["HAEF", "HBGF", "HCIF", "BL-ULTRAS-BBB", "BL-SHORT-BBB",
                                   "FD-USD", "FD-EUR", "FD-AUD"],
        },
    }
    
    def __call__(
        self,
        risk_profile: str,
        recommendation: Dict[str, Any],
        current_portfolio: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check suitability of a recommendation.
        
        Args:
            risk_profile: Client risk profile
            recommendation: Recommendation to check
            current_portfolio: Current portfolio state
            
        Returns:
            Suitability assessment
        """
        rules = self.SUITABILITY_RULES.get(risk_profile, {})

        assessment = {
            "suitable": True,
            "risk_profile": risk_profile,
            "recommendation": recommendation.get("idea", ""),
            "checks": [],
            "violations": [],
            "cmp002_rules_applied": []
        }

        product_type = recommendation.get("product_type", "").lower()
        ticker = recommendation.get("ticker", "").upper()

        # --- CMP-002 Product Gate check ---
        restricted = rules.get("restricted_products", [])
        for restricted_type in restricted:
            if restricted_type in product_type or restricted_type.replace("_", " ") in product_type:
                assessment["suitable"] = False
                assessment["violations"].append(
                    f"CMP-002: '{product_type}' is restricted for {risk_profile} profile"
                )
                assessment["cmp002_rules_applied"].append(f"Product Gate — {restricted_type}")

        # --- CMP-002 Attestation flag (Aggressive only) ---
        requires_attestation = rules.get("requires_attestation", [])
        for att_type in requires_attestation:
            if att_type in product_type:
                assessment["checks"].append(
                    f"CMP-002: '{product_type}' requires attestation on file before proceeding"
                )

        # --- CMP-002 RULE-001: Single equity concentration ---
        if current_portfolio:
            allocation = recommendation.get("allocation", 0)
            max_single = rules.get("max_single_equity_pct", 0.10)
            if allocation > max_single:
                assessment["suitable"] = False
                assessment["violations"].append(
                    f"CMP-002 RULE-001: Single position {allocation*100:.1f}% exceeds {max_single*100:.0f}% limit"
                )
                assessment["cmp002_rules_applied"].append("RULE-001 Concentration")

        # --- CMP-002 RULE-005: Illiquid product cap ---
        max_illiquid = rules.get("max_illiquid_pct", 0.0)
        if "illiquid" in product_type or "lock" in product_type or "private" in product_type:
            if max_illiquid == 0.0:
                assessment["suitable"] = False
                assessment["violations"].append(
                    f"CMP-002 RULE-005: Illiquid/lock-up products not permitted for {risk_profile}"
                )
                assessment["cmp002_rules_applied"].append("RULE-005 Illiquid Cap")

        assessment["checks"].append(
            "CMP-002 baseline screening: PASSED" if assessment["suitable"] else "CMP-002 baseline screening: FAILED"
        )

        # --- Knowledge-base grounding ---
        if self.retriever is not None:
            try:
                kb_queries = [
                    f"suitability rules for {risk_profile} investors",
                    f"restricted products {product_type} compliance policy",
                    f"client risk profiling methodology {risk_profile}",
                ]
                policy_docs = []
                for q in kb_queries:
                    hits = self.retriever.hybrid_retrieve(q, k=2)
                    policy_docs.extend(hits)

                assessment["policy_citations"] = [
                    {
                        "source": d["metadata"].get("source", "unknown"),
                        "doc_type": d["metadata"].get("doc_type", "unknown"),
                        "excerpt": d["content"][:300],
                        "relevance_score": round(d["score"], 4),
                    }
                    for d in policy_docs
                ]

                # Scan retrieved text for explicit restrictions
                combined_text = " ".join(d["content"].lower() for d in policy_docs)
                if product_type and product_type in combined_text:
                    # Check for restriction keywords near the product type
                    import re
                    pattern = rf"(restrict|not.{{0,20}}allow|prohibit|forbidden|not.{{0,10}}permit).{{0,80}}{re.escape(product_type)}"
                    if re.search(pattern, combined_text, re.IGNORECASE):
                        assessment["suitable"] = False
                        assessment["violations"].append(
                            f"Policy document flags '{product_type}' as restricted for {risk_profile}"
                        )

                logger.info(
                    f"Suitability KB grounding: {len(policy_docs)} policy docs retrieved"
                )
            except Exception as e:
                logger.warning(f"KB grounding for suitability failed (using baseline): {e}")

        logger.info(f"Suitability check - {risk_profile}: {'PASSED' if assessment['suitable'] else 'FAILED'}")
        return assessment


class ComplianceGateTool:
    """
    Tool that determines if recommendations need escalation.
    When a retriever is supplied the tool queries the restricted-list and
    compliance-gate documents from the knowledge base and surfaces them
    as citations alongside the gate decision.
    """

    def __init__(self, retriever=None):
        self.retriever = retriever

    def __call__(
        self,
        recommendation: Dict[str, Any],
        suitability_assessment: Dict[str, Any],
        is_licensed_advice: bool = False
    ) -> Dict[str, Any]:
        """
        Determine if recommendation needs escalation.
        
        Args:
            recommendation: The recommendation
            suitability_assessment: Result from suitability checker
            is_licensed_advice: Whether this requires licensed advice
            
        Returns:
            Gate decision (cleared, review, blocked)
        """
        decision = {
            "gate_status": "cleared",
            "requires_escalation": False,
            "reason": "",
            "escalation_type": None,
            "cmp002_rules_triggered": []
        }

        # --- CMP-002: Suitability failure → block ---
        if not suitability_assessment.get("suitable", True):
            decision["gate_status"] = "blocked"
            decision["requires_escalation"] = True
            decision["escalation_type"] = "compliance"
            decision["reason"] = "Fails CMP-002 suitability checks: " + "; ".join(
                suitability_assessment.get("violations", ["see suitability assessment"])
            )
            decision["cmp002_rules_triggered"] = suitability_assessment.get("cmp002_rules_applied", [])

        # --- CMP-002 RULE-036: Discretionary / licensed advice requirement ---
        elif is_licensed_advice or recommendation.get("requires_licensed_advice", False):
            decision["gate_status"] = "review"
            decision["requires_escalation"] = True
            decision["escalation_type"] = "licensed_advisor"
            decision["reason"] = "CMP-002 RULE-036/041: Requires licensed advisor review before proceeding"
            decision["cmp002_rules_triggered"].append("RULE-036 Licensing Gate")

        # --- CMP-002 RULE-030/031/032: Liquidity lock-up disclosure ---
        elif any(k in recommendation.get("product_type", "").lower()
                 for k in ["private", "hedge_fund", "scn", "structured_capital", "lock"]):
            decision["gate_status"] = "review"
            decision["requires_escalation"] = True
            decision["escalation_type"] = "liquidity_disclosure"
            decision["reason"] = "CMP-002 RULE-030/031/032: Lock-up terms must be disclosed in writing before subscription"
            decision["cmp002_rules_triggered"].append("RULE-030/031/032 Liquidity Gate")

        # --- CMP-002 RULE-051: Standard disclaimer check ---
        # (always passes — disclaimer is enforced in ClientBrief schema)

        # --- CMP-002 RULE-059/064/069: RPQ staleness check ---
        rpq_months_old = recommendation.get("rpq_months_old", 0)
        if rpq_months_old > 20:
            if decision["gate_status"] == "cleared":
                decision["gate_status"] = "review"
                decision["requires_escalation"] = True
                decision["escalation_type"] = "rpq_refresh"
                decision["reason"] = f"CMP-002 RULE-059: Client RPQ is {rpq_months_old} months old (>20); refresh required before proceeding"
                decision["cmp002_rules_triggered"].append("RULE-059 RPQ Staleness")

        # --- Low confidence fallback ---
        elif recommendation.get("confidence_score", 1.0) < 0.6 and decision["gate_status"] == "cleared":
            decision["gate_status"] = "review"
            decision["requires_escalation"] = True
            decision["escalation_type"] = "low_confidence"
            decision["reason"] = f"Low confidence score: {recommendation.get('confidence_score', 0):.2f}"
        
        # --- Knowledge-base grounding ---
        if self.retriever is not None:
            try:
                idea = recommendation.get("idea", "")
                product_type = recommendation.get("product_type", "")
                kb_queries = [
                    f"restricted list entitlement rules {product_type}",
                    f"compliance gate escalation requirements {idea[:60]}",
                ]
                gate_docs = []
                for q in kb_queries:
                    hits = self.retriever.hybrid_retrieve(q, k=2)
                    gate_docs.extend(hits)

                decision["policy_citations"] = [
                    {
                        "source": d["metadata"].get("source", "unknown"),
                        "doc_type": d["metadata"].get("doc_type", "unknown"),
                        "excerpt": d["content"][:300],
                        "relevance_score": round(d["score"], 4),
                    }
                    for d in gate_docs
                ]

                # If retrieved docs signal restriction, escalate
                combined_text = " ".join(d["content"].lower() for d in gate_docs)
                if product_type:
                    import re
                    pattern = rf"(restrict|not.{{0,20}}allow|prohibit|forbidden).{{0,80}}{re.escape(product_type.lower())}"
                    if re.search(pattern, combined_text, re.IGNORECASE):
                        if decision["gate_status"] == "cleared":
                            decision["gate_status"] = "review"
                            decision["requires_escalation"] = True
                            decision["escalation_type"] = "policy_document"
                            decision["reason"] = (
                                f"Restricted list document flags '{product_type}'"
                            )

                logger.info(
                    f"Compliance gate KB grounding: {len(gate_docs)} docs retrieved"
                )
            except Exception as e:
                logger.warning(f"KB grounding for compliance gate failed (using baseline): {e}")

        logger.info(f"Compliance gate: {decision['gate_status']}")
        return decision


def create_tools_dict(retriever=None) -> Dict[str, callable]:
    """
    Create a dictionary of all available tools.

    Args:
        retriever: Optional RAGRetriever — when supplied, SuitabilityCheckerTool
                   and ComplianceGateTool will ground their decisions in the
                   knowledge base instead of relying solely on hard-coded rules.

    Returns:
        Dictionary of tool name -> tool callable
    """
    portfolio_tool = PortfolioLookupTool()
    market_tool = MarketDataTool()
    suitability_tool = SuitabilityCheckerTool(retriever=retriever)
    compliance_tool = ComplianceGateTool(retriever=retriever)
    
    return {
        "portfolio_lookup": portfolio_tool,
        "market_data": market_tool,
        "check_suitability": suitability_tool,
        "compliance_gate": compliance_tool
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test tools
    tools = create_tools_dict()
    
    # Test portfolio lookup
    portfolio = tools["portfolio_lookup"]("C-204")
    print(f"\nPortfolio: {portfolio}")
    
    # Test market data
    market = tools["market_data"]("SPY")
    print(f"\nMarket data: {market}")
    
    # Test suitability
    suitability = tools["check_suitability"](
        "Conservative",
        {"idea": "Buy emerging markets fund", "product_type": "emerging_markets", "allocation": 0.15}
    )
    print(f"\nSuitability: {suitability}")
    
    # Test compliance gate
    compliance = tools["compliance_gate"](
        {"idea": "Buy complex structured product", "requires_licensed_advice": True},
        {"suitable": True}
    )
    print(f"\nCompliance: {compliance}")
