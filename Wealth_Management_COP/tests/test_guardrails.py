"""
Unit tests for Guardrails — Phase 5.

Updated to cover:
- Fix 2: Negation-aware compliance gate (e.g., "I cannot recommend you...")
- Fix 1: Word-boundary trigger matching
- Fix 5: Dynamic entitlement filter (fallback path tested without ChromaDB)
"""

from src.guardrails.compliance_gate import run_compliance_gate, _is_negated
from src.guardrails.disclaimers import apply_disclaimers, STANDARD_DISCLAIMER
from src.guardrails.entitlement_filter import verify_entitlements, _HARDCODED_RESTRICTED_FRAGMENTS
from src.models.brief import ComplianceStatus


# ═══════════════════════════════════════════════════════════════════════════════
# Compliance Gate Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_compliance_gate_cleared():
    text = "We should consider a diversified portfolio approach."
    result = run_compliance_gate(text)
    assert result["status"] == ComplianceStatus.CLEARED.value
    assert len(result["flags"]) == 0
    assert result["safe_output"] == text


def test_compliance_gate_blocked_on_guarantee():
    text = "This bond offers a guaranteed return of 5% with no risk."
    result = run_compliance_gate(text)
    assert result["status"] == ComplianceStatus.BLOCKED.value
    assert "no risk" in str(result["flags"])
    assert "COMPLIANCE BLOCK" in result["safe_output"]


def test_compliance_gate_needs_review_licensed_advice():
    """Direct, unnegated licensed advice language should flag for review."""
    text = "I recommend you buy 100 shares of Apple today."
    result = run_compliance_gate(text)
    assert result["status"] == ComplianceStatus.NEEDS_REVIEW.value
    assert "i recommend you" in str(result["flags"]).lower()
    assert result["safe_output"] == text


# Fix 2: Negation-aware tests ─────────────────────────────────────────────────

def test_compliance_gate_negated_cannot_recommend_is_cleared():
    """'I cannot recommend you buy...' should NOT trigger NEEDS_REVIEW (negated)."""
    text = "I cannot recommend you buy any speculative assets under firm policy."
    result = run_compliance_gate(text)
    # No guarantee words, and the trigger is negated → should be CLEARED
    assert result["status"] == ComplianceStatus.CLEARED.value, (
        f"Expected CLEARED for negated advice text, got {result['status']}. Flags: {result['flags']}"
    )


def test_compliance_gate_negated_should_not_recommend_is_cleared():
    """'You should not buy...' (negated 'you should buy') should clear."""
    text = "Under the suitability rules, you should not buy high-yield bonds for this client."
    result = run_compliance_gate(text)
    assert result["status"] == ComplianceStatus.CLEARED.value, (
        f"Expected CLEARED, got {result['status']}. Flags: {result['flags']}"
    )


def test_compliance_gate_guarantee_not_exempt_from_negation():
    """Guarantee language is always blocked, even in a negation context."""
    text = "I would not say this fund has a guaranteed return of 10%, but some claim it does."
    result = run_compliance_gate(text)
    # "guaranteed return" is in the absolute block list — negation doesn't help here
    assert result["status"] == ComplianceStatus.BLOCKED.value


def test_is_negated_helper_detects_cannot():
    lower_text = "the agent cannot recommend you buy this product at this time."
    # "recommend you" starts at index 20 approximately
    idx = lower_text.index("recommend you")
    assert _is_negated(lower_text, idx) is True


def test_is_negated_helper_no_negation():
    lower_text = "i recommend you buy this product at this time."
    idx = lower_text.index("recommend you")
    assert _is_negated(lower_text, idx) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Disclaimers Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_disclaimers_always_adds_standard():
    text = "Here is the portfolio."
    result = apply_disclaimers(text)
    assert STANDARD_DISCLAIMER in result


def test_disclaimers_conditional_crypto():
    text = "Client wants to buy bitcoin."
    result = apply_disclaimers(text)
    assert "Digital Assets Disclaimer" in result
    assert "Cryptocurrencies" in result


def test_disclaimers_no_crypto_disclaimer_for_bonds():
    text = "Consider adding investment-grade bonds to the portfolio."
    result = apply_disclaimers(text)
    assert "Digital Assets Disclaimer" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# Entitlement Filter Tests (Fix 5)
# ═══════════════════════════════════════════════════════════════════════════════

def test_entitlement_filter_institutional_passes_all():
    text = "Here is the emerging_markets_deep_dive report."
    # Institutional has full access — always True
    assert verify_entitlements(text, "institutional") is True


def test_entitlement_filter_standard_blocks_restricted_leak():
    text = "As seen in the emerging_markets_deep_dive, we should buy."
    assert verify_entitlements(text, "standard") is False


def test_entitlement_filter_premium_blocks_restricted_leak():
    text = "The cryptocurrency_digital_assets restricted note says to hold."
    assert verify_entitlements(text, "premium") is False


def test_entitlement_filter_standard_passes_clean_text():
    text = "The portfolio is well balanced with equities and bonds."
    assert verify_entitlements(text, "standard") is True


def test_entitlement_filter_premium_passes_clean_text():
    text = "The equity allocation is within the permitted range for this profile."
    assert verify_entitlements(text, "premium") is True


def test_hardcoded_fallback_covers_key_fragments():
    """Ensure the fallback list covers the known restricted document names."""
    assert any("emerging_markets" in f for f in _HARDCODED_RESTRICTED_FRAGMENTS)
    assert any("cryptocurrency" in f for f in _HARDCODED_RESTRICTED_FRAGMENTS)


# Fix 1: Word-boundary matching (verify via suitability tool) ─────────────────

def test_word_boundary_no_false_positive_on_non_personalized():
    """'non-personalized approach' should NOT trigger the licensed advice check."""
    from src.tools.suitability_checker_tool import _check_licensed_advice_triggers
    assert _check_licensed_advice_triggers(
        "We use a non-personalized approach to portfolio allocation."
    ) is False


def test_word_boundary_still_catches_personalized():
    """'personalized' alone (as a standalone word) should still trigger."""
    from src.tools.suitability_checker_tool import _check_licensed_advice_triggers
    assert _check_licensed_advice_triggers(
        "This is personalized advice tailored for you."
    ) is True
