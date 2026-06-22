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
    
    # Mock database of client portfolios
    MOCK_PORTFOLIOS = {
        "C-201": {
            "client_id": "C-201",
            "client_name": "John Smith",
            "risk_profile": "Balanced",
            "total_value": 500000,
            "last_updated": "2024-06-20",
            "holdings": [
                {"ticker": "SPY", "name": "S&P 500 ETF", "value": 200000, "allocation": 0.40},
                {"ticker": "BND", "name": "Total Bond Market ETF", "value": 150000, "allocation": 0.30},
                {"ticker": "VEA", "name": "Int'l Developed Markets ETF", "value": 100000, "allocation": 0.20},
                {"ticker": "VWO", "name": "Emerging Markets ETF", "value": 50000, "allocation": 0.10},
            ],
            "allocation": {
                "equities": 0.70,
                "fixed_income": 0.30,
                "international": 0.30
            },
            "risk_score": 6.5
        },
        "C-202": {
            "client_id": "C-202",
            "client_name": "Jane Doe",
            "risk_profile": "Conservative",
            "total_value": 750000,
            "last_updated": "2024-06-20",
            "holdings": [
                {"ticker": "AGG", "name": "Total Bond Market ETF", "value": 450000, "allocation": 0.60},
                {"ticker": "IVV", "name": "Large-Cap Equity ETF", "value": 200000, "allocation": 0.27},
                {"ticker": "VGT", "name": "Tech ETF", "value": 50000, "allocation": 0.07},
                {"ticker": "VGIT", "name": "Intermediate Treasury ETF", "value": 50000, "allocation": 0.06},
            ],
            "allocation": {
                "equities": 0.35,
                "fixed_income": 0.65,
                "international": 0.05
            },
            "risk_score": 3.2
        },
        "C-203": {
            "client_id": "C-203",
            "client_name": "Robert Johnson",
            "risk_profile": "Aggressive",
            "total_value": 1000000,
            "last_updated": "2024-06-20",
            "holdings": [
                {"ticker": "QQQ", "name": "Nasdaq-100 ETF", "value": 400000, "allocation": 0.40},
                {"ticker": "VTSAX", "name": "Total Stock Market", "value": 350000, "allocation": 0.35},
                {"ticker": "VTIAX", "name": "International Stock", "value": 200000, "allocation": 0.20},
                {"ticker": "BND", "name": "Bond ETF", "value": 50000, "allocation": 0.05},
            ],
            "allocation": {
                "equities": 0.95,
                "fixed_income": 0.05,
                "international": 0.20
            },
            "risk_score": 8.5
        },
        "C-204": {
            "client_id": "C-204",
            "client_name": "Sarah Wilson",
            "risk_profile": "Growth",
            "total_value": 600000,
            "last_updated": "2024-06-20",
            "holdings": [
                {"ticker": "VTI", "name": "Total US Stock Market", "value": 320000, "allocation": 0.53},
                {"ticker": "VTIAX", "name": "International Stocks", "value": 150000, "allocation": 0.25},
                {"ticker": "BND", "name": "Bonds", "value": 100000, "allocation": 0.17},
                {"ticker": "VNQ", "name": "Real Estate ETF", "value": 30000, "allocation": 0.05},
            ],
            "allocation": {
                "equities": 0.83,
                "fixed_income": 0.17,
                "international": 0.25
            },
            "risk_score": 7.0
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
    
    MOCK_MARKET_DATA = {
        "SPY": {"price": 549.23, "change": 1.2, "pe_ratio": 28.5, "dividend_yield": 1.42},
        "BND": {"price": 77.85, "change": -0.3, "duration": 6.2, "yield": 4.8},
        "VEA": {"price": 52.40, "change": 0.8, "pe_ratio": 14.2, "dividend_yield": 3.1},
        "VWO": {"price": 42.15, "change": 1.5, "pe_ratio": 10.5, "dividend_yield": 2.8},
        "QQQ": {"price": 498.50, "change": 2.1, "pe_ratio": 45.0, "dividend_yield": 0.5},
        "GEF": {"price": 125.40, "change": -1.5, "pe_ratio": 22.0, "dividend_yield": 2.5},
        "IYW": {"price": 245.80, "change": 2.8, "pe_ratio": 35.0, "dividend_yield": 0.6},
    }
    
    ECONOMIC_INDICATORS = {
        "gdp_growth": "2.4%",
        "inflation_rate": "3.1%",
        "unemployment_rate": "3.9%",
        "federal_funds_rate": "5.25%-5.50%",
        "10yr_yield": "4.15%",
        "vix_index": 14.5,
        "market_sentiment": "cautiously optimistic"
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
    """
    
    # Suitability rules
    SUITABILITY_RULES = {
        "Conservative": {
            "max_equity_allocation": 0.40,
            "min_bond_allocation": 0.50,
            "max_volatility": 0.08,
            "restricted_products": ["emerging_markets", "derivatives", "leveraged_funds"]
        },
        "Balanced": {
            "equity_min": 0.50,
            "equity_max": 0.60,
            "bond_min": 0.30,
            "bond_max": 0.40,
            "max_volatility": 0.12,
            "restricted_products": ["leveraged_funds"]
        },
        "Growth": {
            "equity_min": 0.70,
            "equity_max": 0.80,
            "bond_max": 0.20,
            "max_volatility": 0.15,
            "restricted_products": []
        },
        "Aggressive": {
            "equity_min": 0.85,
            "equity_allocation": 1.0,
            "max_volatility": None,
            "restricted_products": []
        }
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
            "violations": []
        }
        
        # Check if product is restricted
        product_type = recommendation.get("product_type", "").lower()
        restricted = rules.get("restricted_products", [])
        
        for restricted_type in restricted:
            if restricted_type in product_type:
                assessment["suitable"] = False
                assessment["violations"].append(
                    f"Product type '{product_type}' is restricted for {risk_profile} profile"
                )
        
        # Check concentration
        if current_portfolio:
            allocation = recommendation.get("allocation", 0)
            if allocation > 0.10:
                assessment["suitable"] = False
                assessment["violations"].append(
                    f"Concentration exceeds 10% limit ({allocation*100:.1f}%)"
                )
        
        assessment["checks"].append("Product type screening: OK" if assessment["suitable"] else "FAILED")
        
        logger.info(f"Suitability check - {risk_profile}: {'PASSED' if assessment['suitable'] else 'FAILED'}")
        return assessment


class ComplianceGateTool:
    """
    Tool that determines if recommendations need escalation.
    """
    
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
            "escalation_type": None
        }
        
        # Check suitability
        if not suitability_assessment.get("suitable", True):
            decision["gate_status"] = "blocked"
            decision["requires_escalation"] = True
            decision["escalation_type"] = "compliance"
            decision["reason"] = "Fails suitability checks"
        
        # Check for licensed advice requirement
        elif is_licensed_advice or recommendation.get("requires_licensed_advice", False):
            decision["gate_status"] = "review"
            decision["requires_escalation"] = True
            decision["escalation_type"] = "licensed_advice"
            decision["reason"] = "Requires licensed advisor review"
        
        # Check confidence
        elif recommendation.get("confidence_score", 1.0) < 0.6:
            decision["gate_status"] = "review"
            decision["requires_escalation"] = True
            decision["escalation_type"] = "low_confidence"
            decision["reason"] = f"Low confidence score: {recommendation.get('confidence_score', 0)}"
        
        logger.info(f"Compliance gate: {decision['gate_status']}")
        return decision


def create_tools_dict() -> Dict[str, callable]:
    """
    Create a dictionary of all available tools.
    
    Returns:
        Dictionary of tool name -> tool callable
    """
    portfolio_tool = PortfolioLookupTool()
    market_tool = MarketDataTool()
    suitability_tool = SuitabilityCheckerTool()
    compliance_tool = ComplianceGateTool()
    
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
