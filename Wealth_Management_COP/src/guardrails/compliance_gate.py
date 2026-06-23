"""
Compliance Gate — Phase 5: Guardrails.

A post-generation guardrail that scans the LLM's final output for
compliance violations (e.g., licensed advice language, guarantees)
before it is presented to the RM or client.
"""

import re
from typing import Any

from src.models.brief import ComplianceStatus
from src.tools.suitability_checker_tool import LICENSED_ADVICE_TRIGGERS

# Words that, when appearing immediately before a trigger, indicate the trigger
# is being *refused* rather than *asserted* (negation context).
_NEGATION_WORDS = {
    "cannot", "can't", "cant", "should not", "shouldn't", "shouldnt",
    "will not", "won't", "wont", "do not", "don't", "dont",
    "am not", "is not", "are not", "must not", "mustn't",
    "unable to", "not able to", "not authorised", "not authorized",
    "not permitted", "not allowed", "decline to",
}

# How many whitespace-separated tokens to scan before the trigger position.
_NEGATION_LOOKAHEAD_TOKENS = 6


def _is_negated(text: str, trigger_start: int) -> bool:
    """
    Return True if the trigger at *trigger_start* is preceded by a negation
    word within the previous ``_NEGATION_LOOKAHEAD_TOKENS`` tokens.

    Args:
        text: The full lower-cased output text.
        trigger_start: Character index where the trigger phrase begins.
    """
    # Grab the substring before the trigger and take the last N words
    prefix = text[:trigger_start]
    tokens = prefix.split()
    window = " ".join(tokens[-_NEGATION_LOOKAHEAD_TOKENS:])
    return any(neg in window for neg in _NEGATION_WORDS)


def run_compliance_gate(final_output: str) -> dict[str, Any]:
    """
    Scans the final text for compliance violations.

    Guarantee language (e.g. "no risk", "guaranteed return") is always
    flagged and results in a BLOCKED status.

    Licensed-advice language is only flagged when it is NOT preceded by a
    negation context (e.g. "I cannot recommend you..." is safe).

    Args:
        final_output: The generated text from the agent.

    Returns:
        Dict with status (ComplianceStatus), flags (list), and safe_output (str).
    """
    lower_out = final_output.lower()
    flags = []

    # 1. Check for absolute guarantees (no negation exemption — always blocked)
    guarantee_words = ["guaranteed return", "no risk", "100% safe", "cannot lose"]
    for word in guarantee_words:
        if word in lower_out:
            flags.append(f"Contains prohibited guarantee language: '{word}'")

    # 2. Check for licensed advice triggers — skip if negated
    for trigger in LICENSED_ADVICE_TRIGGERS:
        match = re.search(re.escape(trigger), lower_out)
        if match and not _is_negated(lower_out, match.start()):
            flags.append(f"Potential licensed advice language detected: '{trigger}'")

    # Determine status
    if any("guarantee" in f or "safe" in f for f in flags):
        status = ComplianceStatus.BLOCKED
        safe_output = (
            "⚠️ **COMPLIANCE BLOCK**\n\n"
            "The generated response was blocked because it contained prohibited "
            "language implying guaranteed returns or absence of risk. "
            "RMs must not make such statements under firm policy."
        )
    elif flags:
        status = ComplianceStatus.NEEDS_REVIEW
        safe_output = final_output  # Pass through, but UI will highlight it
    else:
        status = ComplianceStatus.CLEARED
        safe_output = final_output

    return {
        "status": status.value,
        "flags": flags,
        "safe_output": safe_output,
    }
