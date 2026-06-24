"""Streamlit RM Dashboard for the Relationship Manager Copilot."""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.graph import resume_agent, run_agent
from src.guardrails.compliance import detect_licensed_advice_request
from src.models import ClientBrief, ComplianceStatus, RiskProfile, Sensitivity
from src.tools.retriever import HybridRAGRetriever
from src.tools.suitability import check_suitability_logic

st.set_page_config(
    page_title="RM Copilot — Wealth Management",
    page_icon="📊",
    layout="wide",
)

st.title("Relationship Manager Copilot")
st.caption("Decision support for wealth management RMs. Not automated investment advice.")

with st.sidebar:
    st.header("Configuration")
    rm_id = st.selectbox("RM ID", ["RM-001", "RM-002", "RM-003"])
    entitlement_map = {
        "RM-001": ["public", "internal"],
        "RM-002": ["public", "internal", "restricted"],
        "RM-003": ["public"],
    }
    rm_entitlements = entitlement_map[rm_id]
    st.write(f"Entitlements: {', '.join(rm_entitlements)}")
    st.caption("LangSmith tracing: set LANGCHAIN_TRACING_V2=true in .env")

    st.divider()
    st.header("Quick Scenarios")
    scenarios = {
        "Quarterly Review (C-204)": (
            "Prepare talking points for client C-204's quarterly review"
        ),
        "Suitability (PG-003 / Conservative)": (
            "Is PG-003 suitable for a conservative client?"
        ),
        "Suitability (PG-002 / Conservative)": (
            "Is PG-002 suitable for a conservative client?"
        ),
        "Portfolio Risk Summary": (
            "Summarize the portfolio risk for client C-204"
        ),
        "Licensed Advice (Escalation)": (
            "Recommend a personalized 4% withdrawal rate for client C-204 retirement"
        ),
        "Restricted Compliance Rules": (
            "What are the restricted concentration limit rules for client C-301?"
        ),
        "Tech & AI Research": (
            "What are the latest technology and AI sector insights for client C-301?"
        ),
    }
    selected_scenario = st.selectbox("Load scenario", ["Custom"] + list(scenarios.keys()))

st.divider()

tab_brief, tab_suitability, tab_retrieval = st.tabs(
    ["Generate Brief", "Suitability Check", "RAG Retrieval"]
)

with tab_brief:
    if selected_scenario != "Custom":
        default_query = scenarios[selected_scenario]
    else:
        default_query = ""

    query = st.text_area(
        "Enter your request",
        value=default_query,
        height=100,
        placeholder="e.g., Prepare talking points for client C-204's quarterly review",
    )
    client_id = st.text_input("Client ID (optional)", value="C-204")

    if st.button("Generate Brief", type="primary"):
        if not query.strip():
            st.warning("Please enter a request.")
        else:
            with st.spinner("Running agent pipeline..."):
                try:
                    if detect_licensed_advice_request(query):
                        st.warning(
                            "Licensed advisor sign-off may be required. "
                            "Human-in-the-loop review will be triggered."
                        )

                    thread_id = f"streamlit-{rm_id}-{hash(query) % 10000}"
                    output = run_agent(
                        query=query,
                        rm_id=rm_id,
                        rm_entitlements=rm_entitlements,
                        client_id=client_id or None,
                        thread_id=thread_id,
                    )

                    if isinstance(output, dict) and output.get("interrupted"):
                        st.warning("Human-in-the-loop review required")
                        draft = output.get("draft_brief")
                        if draft:
                            if isinstance(draft, ClientBrief):
                                preview = draft
                            else:
                                preview = ClientBrief(**draft)
                            st.json(json.loads(preview.model_dump_json()))

                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("Approve Brief", key="approve"):
                                    final = resume_agent(thread_id, approved=True)
                                    if isinstance(final, ClientBrief):
                                        st.session_state["last_brief"] = final
                                        st.success("Brief approved by RM")
                                    st.rerun()
                            with col_b:
                                if st.button("Reject / Escalate", key="reject"):
                                    resume_agent(thread_id, approved=False)
                                    st.error("Brief rejected — escalated to licensed advisor")
                        st.stop()

                    brief = output if isinstance(output, ClientBrief) else None
                    if not brief:
                        st.error("No brief generated")
                        st.stop()

                    st.session_state["last_brief"] = brief
                    status_color = {
                        ComplianceStatus.CLEARED: "green",
                        ComplianceStatus.NEEDS_REVIEW: "orange",
                        ComplianceStatus.BLOCKED: "red",
                    }
                    st.markdown(
                        f"**Compliance Status:** "
                        f":{status_color[brief.compliance_status]}[{brief.compliance_status.value}]"
                    )

                    if brief.escalation_reason:
                        st.error(f"Escalation: {brief.escalation_reason}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Client Overview")
                        st.write(f"**Client ID:** {brief.client_id}")
                        st.write(f"**Risk Profile:** {brief.risk_profile.value}")
                        st.text(brief.portfolio_summary)

                    with col2:
                        st.subheader("Talking Points")
                        for i, tp in enumerate(brief.talking_points, 1):
                            st.write(f"{i}. {tp}")

                    if brief.recommendations:
                        st.subheader("Recommendations")
                        for reco in brief.recommendations:
                            with st.expander(reco.idea):
                                st.write(f"**Rationale:** {reco.rationale}")
                                st.write(f"**Suitability:** {reco.suitability}")
                                if reco.citations:
                                    st.write("**Citations:**")
                                    for c in reco.citations:
                                        st.write(f"  - {c}")

                    st.info(brief.disclaimer)
                    with st.expander("Raw JSON"):
                        st.json(json.loads(brief.model_dump_json()))

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info(
                        "Ensure OPENAI_API_KEY is set in .env and "
                        "run `python scripts/ingest.py` to build the vector store."
                    )

with tab_suitability:
    st.subheader("Fund Suitability Checker")
    col1, col2 = st.columns(2)
    with col1:
        fund = st.text_input("Fund / Product", value="PG-003")
    with col2:
        profile = st.selectbox(
            "Client Risk Profile",
            ["Conservative", "Balanced", "Growth", "Aggressive"],
        )

    if st.button("Check Suitability"):
        result = check_suitability_logic(fund, RiskProfile(profile))
        if result.suitable:
            st.success(result.assessment)
        else:
            st.error(result.assessment)
        if result.citations:
            st.write("**Citations:**")
            for c in result.citations:
                st.write(f"- {c}")

with tab_retrieval:
    st.subheader("Hybrid RAG Retrieval (BM25 + Vector)")
    rag_query = st.text_input("Search query", value="conservative fixed income PG-002")
    k = st.slider("Number of results", 1, 10, 5)

    if st.button("Search"):
        retriever = HybridRAGRetriever()
        chunks = retriever.retrieve(
            rag_query,
            k=k,
            rm_entitlements=[Sensitivity(e) for e in rm_entitlements],
        )

        if not chunks:
            st.warning("No results found (check entitlements or run ingestion).")
        for chunk in chunks:
            with st.expander(
                f"{chunk.doc_id} | {chunk.doc_type.value} | "
                f"score={chunk.score:.3f} | {chunk.sensitivity.value}"
            ):
                st.text(chunk.content[:500])
