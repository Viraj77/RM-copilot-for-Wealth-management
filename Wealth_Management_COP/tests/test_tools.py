"""
Unit tests for Agent Tools — Phase 3.
Tests portfolio lookup, market data, and suitability checker
without requiring OpenAI API access (all pure-logic tests).

Run: pytest tests/test_tools.py -v
"""

import json
import pytest

from src.tools.portfolio_tool import (
    _check_entitlement,
    _load_client_db,
    get_all_client_ids,
    get_client_display_names,
    portfolio_lookup,
)
from src.tools.market_data_tool import (
    _load_market_db,
    get_available_tickers,
    market_data,
    market_data_bulk,
)
from src.tools.suitability_checker_tool import (
    PROFILE_PROHIBITIONS,
    _check_licensed_advice_triggers,
    _check_profile_prohibition,
    check_suitability,
)
from src.models.client import EntitlementTier
from src.models.brief import SuitabilityVerdict


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio Tool Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioTool:
    """Tests for portfolio_lookup and related functions."""

    def test_lookup_known_client(self):
        result = portfolio_lookup("C-204", rm_tier="premium")
        assert result["success"] is True
        assert result["client_id"] == "C-204"
        assert result["client_name"] == "Sarah Chen"
        assert result["risk_profile"] == "balanced"
        assert result["total_aum_usd"] == 2_400_000.0
        assert len(result["holdings"]) > 0

    def test_lookup_all_clients_succeed(self):
        for client_id in ["C-204", "C-301", "C-115", "C-442", "C-510"]:
            result = portfolio_lookup(client_id, rm_tier="institutional")
            assert result["success"] is True, f"Expected success for {client_id}"

    def test_lookup_unknown_client_returns_error(self):
        result = portfolio_lookup("C-999")
        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert "available_clients" in result

    def test_client_id_case_insensitive(self):
        result_upper = portfolio_lookup("C-204", rm_tier="premium")
        result_mixed = portfolio_lookup("c-204", rm_tier="premium")
        assert result_upper["success"] == result_mixed["success"]

    def test_allocation_breakdown_present(self):
        result = portfolio_lookup("C-204", rm_tier="premium")
        assert "allocation_breakdown" in result
        allocation = result["allocation_breakdown"]
        total = sum(allocation.values())
        assert abs(total - 100.0) <= 2.0, f"Allocation total {total} should be ~100%"

    def test_holdings_structure(self):
        result = portfolio_lookup("C-204", rm_tier="premium")
        for holding in result["holdings"]:
            assert "ticker" in holding
            assert "name" in holding
            assert "asset_class" in holding
            assert "allocation_pct" in holding
            assert "current_value_usd" in holding
            assert "gain_loss_pct" in holding

    def test_entitlement_check_standard_rm_standard_client(self):
        from src.models.client import ClientProfile, RiskProfile, EntitlementTier
        client = ClientProfile(
            client_id="C-001",
            name="Test",
            risk_profile=RiskProfile.BALANCED,
            investment_horizon="5 years",
            total_aum=100_000.0,
            entitlement_tier=EntitlementTier.STANDARD,
        )
        assert _check_entitlement(client, "standard") is True

    def test_entitlement_check_standard_rm_premium_client_blocked(self):
        from src.models.client import ClientProfile, RiskProfile, EntitlementTier
        client = ClientProfile(
            client_id="C-204",
            name="Sarah Chen",
            risk_profile=RiskProfile.BALANCED,
            investment_horizon="7 years",
            total_aum=2_400_000.0,
            entitlement_tier=EntitlementTier.PREMIUM,
        )
        # Standard RM cannot access premium client
        assert _check_entitlement(client, "standard") is False

    def test_entitlement_check_premium_rm_accesses_premium_client(self):
        from src.models.client import ClientProfile, RiskProfile, EntitlementTier
        client = ClientProfile(
            client_id="C-204",
            name="Sarah Chen",
            risk_profile=RiskProfile.BALANCED,
            investment_horizon="7 years",
            total_aum=2_400_000.0,
            entitlement_tier=EntitlementTier.PREMIUM,
        )
        assert _check_entitlement(client, "premium") is True

    def test_get_all_client_ids_returns_list(self):
        ids = get_all_client_ids()
        assert isinstance(ids, list)
        assert len(ids) >= 5
        assert "C-204" in ids

    def test_get_client_display_names(self):
        names = get_client_display_names()
        assert "C-204" in names
        assert "Sarah Chen" in names["C-204"]

    def test_conservative_client_lookup(self):
        result = portfolio_lookup("C-115")
        assert result["success"] is True
        assert result["risk_profile"] == "conservative"
        assert result["client_name"] == "Margaret Thompson"

    def test_aggressive_client_lookup(self):
        result = portfolio_lookup("C-301", rm_tier="institutional")
        assert result["success"] is True
        assert result["risk_profile"] == "aggressive"


# ═══════════════════════════════════════════════════════════════════════════════
# Market Data Tool Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketDataTool:
    """Tests for market_data and related functions."""

    def test_lookup_known_ticker(self):
        result = market_data("VTI")
        assert result["success"] is True
        assert result["ticker"] == "VTI"
        assert result["current_price_usd"] is not None
        assert result["asset_class"] == "equity"

    def test_lookup_all_holding_tickers(self):
        tickers = ["VTI", "QQQ", "AGG", "BND", "TIPS", "REET", "VYM", "CASH"]
        for ticker in tickers:
            result = market_data(ticker)
            assert result["success"] is True, f"Expected success for {ticker}"

    def test_period_returns_present(self):
        result = market_data("VTI", period="3M")
        assert result["period"] == "3M"
        assert result["period_return_pct"] is not None
        assert result["all_period_returns"] is not None

    def test_invalid_period_returns_error(self):
        result = market_data("VTI", period="2Y")
        assert result["success"] is False
        assert "Invalid period" in result["error"]

    def test_unknown_ticker_returns_error(self):
        result = market_data("FAKEXYZ")
        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert "available_tickers" in result

    def test_case_insensitive_ticker(self):
        result_upper = market_data("VTI")
        result_lower = market_data("vti")
        assert result_upper["success"] == result_lower["success"]

    def test_fixed_income_has_duration(self):
        result = market_data("AGG")
        assert result["success"] is True
        assert "duration_years" in result

    def test_tips_is_inflation_linked(self):
        result = market_data("TIPS")
        assert result["success"] is True
        assert result.get("inflation_linked") is True

    def test_high_yield_has_note(self):
        result = market_data("HYG")
        assert result["success"] is True
        assert "note" in result  # Warning about suitability

    def test_bulk_market_data(self):
        results = market_data_bulk(["VTI", "AGG", "REET"])
        assert len(results) == 3
        for ticker, result in results.items():
            assert result["success"] is True

    def test_get_available_tickers(self):
        tickers = get_available_tickers()
        assert len(tickers) >= 10
        assert "VTI" in tickers
        assert "AGG" in tickers

    def test_all_periods_available(self):
        result = market_data("VTI")
        returns = result["all_period_returns"]
        for period in ["1W", "1M", "3M", "6M", "1Y"]:
            assert period in returns, f"Missing period {period}"


# ═══════════════════════════════════════════════════════════════════════════════
# Suitability Checker Tool Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSuitabilityChecker:
    """Tests for the suitability checker — rule-based logic only (no LLM)."""

    # ── Licensed advice trigger tests ─────────────────────────────────────────

    def test_personalized_advice_triggers_escalation(self):
        assert _check_licensed_advice_triggers(
            "I recommend you buy Tesla immediately."
        ) is True

    def test_guarantee_triggers_escalation(self):
        assert _check_licensed_advice_triggers(
            "This fund offers a guaranteed return of 10%."
        ) is True

    def test_neutral_statement_no_trigger(self):
        assert _check_licensed_advice_triggers(
            "Consider rebalancing equity allocation from 60% to 55%."
        ) is False

    def test_licensed_advice_verdict(self):
        result = check_suitability(
            recommendation="You should buy 20% crypto for guaranteed returns.",
            client_risk_profile="aggressive",
            use_llm=False,
        )
        assert result["verdict"] == SuitabilityVerdict.REQUIRES_LICENSED_ADVICE.value
        assert result["requires_escalation"] is True

    # ── Profile prohibition tests ─────────────────────────────────────────────

    def test_high_yield_prohibited_for_conservative(self):
        prohibited = _check_profile_prohibition(
            "Add 10% High Yield Corporate Bond Fund.", "conservative"
        )
        assert prohibited is not None
        assert "high yield" in prohibited.lower()

    def test_crypto_prohibited_for_balanced(self):
        prohibited = _check_profile_prohibition(
            "Allocate 5% to Bitcoin ETF.", "balanced"
        )
        assert prohibited is not None

    def test_equity_not_prohibited_for_growth(self):
        prohibited = _check_profile_prohibition(
            "Increase equity allocation to 75%.", "growth"
        )
        assert prohibited is None

    def test_conservative_rejects_high_yield_rec(self):
        result = check_suitability(
            recommendation="Add 10% High Yield Corporate Bond allocation.",
            client_risk_profile="conservative",
            policy_context=["High yield is prohibited for conservative clients."],
            use_llm=False,
        )
        assert result["verdict"] == SuitabilityVerdict.UNSUITABLE.value
        assert result["success"] is True
        assert len(result["rule_violations"]) > 0

    def test_conservative_rejects_crypto(self):
        result = check_suitability(
            recommendation="Invest 2% in cryptocurrency ETF.",
            client_risk_profile="conservative",
            use_llm=False,
        )
        assert result["verdict"] in (
            SuitabilityVerdict.UNSUITABLE.value,
            SuitabilityVerdict.REQUIRES_LICENSED_ADVICE.value,
        )

    def test_conservative_rejects_emerging_markets(self):
        result = check_suitability(
            recommendation="Add 10% emerging market equity fund.",
            client_risk_profile="conservative",
            use_llm=False,
        )
        assert result["verdict"] == SuitabilityVerdict.UNSUITABLE.value

    def test_valid_rec_for_balanced_passes_rules(self):
        result = check_suitability(
            recommendation="Rebalance equity allocation from 65% to 55%, adding 10% to core bonds.",
            client_risk_profile="balanced",
            use_llm=False,
        )
        # Should pass rule-based checks (no LLM)
        assert result["success"] is True
        assert result["verdict"] in (
            SuitabilityVerdict.SUITABLE.value,
            SuitabilityVerdict.MARGINAL.value,
        )

    def test_no_policy_context_returns_marginal(self):
        result = check_suitability(
            recommendation="Consider adding TIPS exposure.",
            client_risk_profile="conservative",
            policy_context=[],
            use_llm=False,
        )
        assert result["success"] is True
        # Without context and LLM, should be marginal
        assert result["verdict"] in (
            SuitabilityVerdict.MARGINAL.value,
            SuitabilityVerdict.SUITABLE.value,
        )

    def test_result_always_has_required_keys(self):
        result = check_suitability(
            recommendation="Maintain current allocation.",
            client_risk_profile="balanced",
            use_llm=False,
        )
        required_keys = ["success", "verdict", "reasoning", "requires_escalation"]
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in suitability result"

    def test_profile_prohibitions_cover_all_risky_products(self):
        # Verify conservative clients are protected from all major risky products
        prohibited_terms = PROFILE_PROHIBITIONS.get("conservative", [])
        assert "cryptocurrency" in prohibited_terms
        assert "high yield" in prohibited_terms
        assert "emerging market" in prohibited_terms
        assert "private credit" in prohibited_terms
