import streamlit as st

from src.guardrails.compliance_gate import run_compliance_gate
from src.guardrails.disclaimers import apply_disclaimers
from src.guardrails.entitlement_filter import verify_entitlements
from src.models.brief import ComplianceStatus


def render_brief(raw_text: str, rm_tier: str) -> None:
    """
    Passes the raw LLM output through the Phase 5 Guardrails and renders it.
    """
    if not raw_text.strip():
        return

    # 1. Entitlement Filter (Hallucination catch)
    if not verify_entitlements(raw_text, rm_tier):
        st.error(
            "🚨 **SECURITY BLOCK**: The generated response attempted to reference "
            "restricted internal documents that your tier does not have access to. "
            "The output has been blocked."
        )
        return

    # 2. Compliance Gate (Guarantees & Licensed Advice)
    gate_result = run_compliance_gate(raw_text)
    
    # Display any compliance flags immediately
    if gate_result["status"] == ComplianceStatus.BLOCKED.value:
        st.error(gate_result["safe_output"])
        for flag in gate_result["flags"]:
            st.warning(f"🚩 **Flag**: {flag}")
        return
        
    elif gate_result["status"] == ComplianceStatus.NEEDS_REVIEW.value:
        st.warning(
            "⚠️ **COMPLIANCE WARNING**: This response contains language that may require "
            "a licensed financial advisor's review. Please verify before sharing with the client."
        )
        for flag in gate_result["flags"]:
            st.info(f"🚩 **Flag**: {flag}")

    # 3. Apply Disclaimers
    final_output = apply_disclaimers(gate_result["safe_output"])

    # 4. Render the safe brief
    st.markdown("### 📄 Generated Brief")
    
    # Using a neat container for the brief text
    with st.container(border=True):
        st.markdown(final_output)
