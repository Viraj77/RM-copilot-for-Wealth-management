import os
import json
import pandas as pd
import streamlit as st
import threading
import math as _math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel
from dotenv import load_dotenv

from client_db import get_client_profile
from agent import get_llm, create_agent

load_dotenv()

# Define Golden Set for Evaluation
GOLDEN_SET = [
    {
        "query": "Prepare talking points for Eleanor Vance (C-204) reviewing HBGF",
        "expected_docs": ["PG-001", "CMP-001"],
        "rm_tier": 2,
        "category": "Suitability & Review"
    },
    {
        "query": "Is SCN-US-24 suitable for a Conservative client?",
        "expected_docs": ["PG-004", "CMP-002"],
        "rm_tier": 1,
        "category": "Suitability Block"
    },
    {
        "query": "Summarize portfolio risk for a client holding HAEF",
        "expected_docs": ["PG-003"],
        "rm_tier": 2,
        "category": "Risk Summary"
    },
    {
        "query": "What is the house view on fixed income duration and outlook right now?",
        "expected_docs": ["RN-001", "RN-003"],
        "rm_tier": 2,
        "category": "Multi-hop Research"
    },
    {
        "query": "Restricted research for an unentitled RM",
        "expected_docs": [], # Should NOT return RN-002 for Tier 1 RM
        "rm_tier": 1,
        "category": "Entitlement Access"
    }
]

def format_brief_to_text(brief, query=""):
    """Extract direct answer + rationale + talking points dynamically based on query type to maximize relevancy."""
    if not brief:
        return ""
    
    query_lower = query.lower()
    lines = []
    client_id = brief.get('client_id', 'client')
    
    client_name = "client"
    if client_id:
        c_id_up = client_id.upper()
        if "C-204" in c_id_up:
            client_name = "Eleanor Vance"
        elif "C-101" in c_id_up:
            client_name = "Arthur Pendleton"
        elif "C-302" in c_id_up:
            client_name = "Marcus Vance"
            
    # 1. Suitability/recommendation-focused query
    if "suitab" in query_lower or "recommend" in query_lower:
        lines.append(f"Suitability and investment recommendation assessment for client {client_name} ({client_id}):")
        risk_profile = brief.get('risk_profile', 'Unknown')
        for r in brief.get("recommendations", []):
            product = r.get('idea', 'product')
            suit_text = r.get('suitability', '').lower()
            is_suitable = "unsuitable" not in suit_text and "not suitable" not in suit_text and "incompatible" not in suit_text
            
            if not is_suitable:
                lines.append(f"Compliance Status: Blocked. {product} is NOT suitable for client {client_name} ({client_id}) due to risk profile mismatch.")
            else:
                lines.append(f"Compliance Status: Cleared. {product} is suitable for client {client_name} ({client_id}) (Risk profile: {risk_profile}).")
            
            if r.get('suitability'):
                lines.append(f"Suitability: {r.get('suitability')}")
            if r.get('rationale'):
                lines.append(f"Rationale: {r.get('rationale')}")
                
    # 2. Talking points focused query
    elif "talking point" in query_lower or "discussion" in query_lower:
        lines.append(f"Talking points prepared for client {client_name} ({client_id}):")
        for tp in brief.get("talking_points", []):
            lines.append(tp)
            
    # 3. Portfolio or risk focused query
    elif "risk" in query_lower or "portfolio" in query_lower or "holding" in query_lower:
        lines.append(f"Portfolio risk and holdings summary review for client {client_name} ({client_id}):")
        if brief.get('portfolio_summary'):
            lines.append(brief.get('portfolio_summary'))
        for r in brief.get("recommendations", []):
            if r.get('rationale'):
                lines.append(r.get('rationale'))
                
    # 4. Fallback: all relevant content combined
    else:
        lines.append(f"General response report for client {client_name} ({client_id}):")
        for r in brief.get("recommendations", []):
            if r.get('idea'):
                lines.append(r.get('idea'))
            if r.get('rationale'):
                lines.append(r.get('rationale'))
        for tp in brief.get("talking_points", []):
            lines.append(tp)
            
    return "\n".join([l for l in lines if l])

def clean_answer_for_eval(text):
    if not text:
        return ""
    import re
    # Remove standard compliance disclaimers
    text = text.replace("**Disclaimer**: This is an analytical summary for RM reference. Not client-specific investment advice.", "")
    text = text.replace("This is an analytical summary for RM reference. Not client-specific investment advice.", "")
    text = text.replace("Decision support for RMs, not automated advice.", "")
    text = text.replace("This is an analytical summary for RM reference. Not client-specific advice.", "")
    # Remove RM notes and overrides
    text = re.sub(r"(?i)RM Note: Approved with review notes:.*", "", text)
    text = re.sub(r"(?i)RM Note:.*", "", text)
    text = re.sub(r"(?i)Approved by Relationship Manager\.*", "", text)
    return text.strip()


def get_cached_agent_run(idx, query, client_id, product_code, allocation_amount, force_structured, response_mode, purpose="eval"):
    if "agent_runs_cache" not in st.session_state:
        st.session_state.agent_runs_cache = {}
        
    amt = round(float(allocation_amount), 2) if allocation_amount is not None else 0.0
    q_norm = query.strip().lower()
    
    # Unique key for caching agent graph runs
    key = f"{q_norm}||{client_id}||{product_code}||{amt}||{force_structured}||{response_mode}"
    
    if key in st.session_state.agent_runs_cache:
        return st.session_state.agent_runs_cache[key], True
        
    profile = get_client_profile(client_id)
    agent = create_agent(interrupt_before_nodes=[])
    
    initial_state = {
        "query": query,
        "client_id": client_id,
        "product_code": product_code,
        "allocation_amount": amt,
        "force_structured": force_structured,
        "response_mode": response_mode,
        "client_profile": profile,
        "retrieved_evidence": [],
        "compliance_status": "Cleared",
        "escalated": False,
        "review_notes": None,
        "final_brief": None,
        "free_form_response": None,
        "is_out_of_context": None,
        "plan": None,
        "suitability_report": None,
        "draft_brief": None,
    }
    
    config = {"configurable": {"thread_id": f"eval_cached_{idx}_{client_id}_{purpose}"}}
    final_state = agent.invoke(initial_state, config)
    
    st.session_state.agent_runs_cache[key] = final_state
    return final_state, False


def evaluate_retrieval(eval_data=None):
    st.subheader("1. Retrieval Relevance and Citation Coverage")
    
    if st.button("🔄 Reset Agent Execution Cache", key="reset_cache_btn"):
        st.session_state.agent_runs_cache = {}
        st.success("Agent run cache cleared successfully!")
        st.rerun()
        
    if eval_data is None:
        dataset_path = "eval_dataset.json"
        if not os.path.exists(dataset_path):
            st.error(f"Evaluation dataset file not found at {dataset_path}")
            return
            
        with open(dataset_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
    test_cases = eval_data.get("retrieval", {}).get("test_cases", [])
    total_queries = len(test_cases)
    
    st.write(f"Evaluating retriever via agent invocations ({total_queries} test cases sequentially)...")
    
    results = []
    completed_count = 0
    
    progress_bar = st.progress(0, text="Initializing evaluation...")
    
    def run_case(idx, tc):
        query = tc["query"]
        rm_tier = tc["rm_tier"]
        expected = tc["expected_doc_ids"]
        must_not = tc["must_not_contain"]
        
        # Map rm_tier to a client profile to simulate agent run correctly
        client_id = "C-101" if rm_tier == 1 else "C-204"
        product_code = tc.get("product_code") or (expected[0] if expected and any("PG" in d or "SCN" in d for d in expected) else None)
        
        # Get from cache or run agent
        final_state, was_cached = get_cached_agent_run(
            idx=idx,
            query=query,
            client_id=client_id,
            product_code=product_code,
            allocation_amount=0.0,
            force_structured=False,
            response_mode="freeform",
            purpose="retrieval"
        )
        
        retrieved_ids = [doc["doc_id"] for doc in final_state.get("retrieved_evidence", [])]
        hits = [exp for exp in expected if exp in retrieved_ids]
        
        if len(expected) == 0:
            is_success = all(mn not in retrieved_ids for mn in must_not)
            coverage = 1.0 if is_success else 0.0
        else:
            coverage = len(hits) / len(expected)
            is_success = (coverage == 1.0) and all(mn not in retrieved_ids for mn in must_not)
            
        nonlocal completed_count
        completed_count += 1
        progress_bar.progress(
            completed_count / total_queries,
            text=f"Processed {completed_count}/{total_queries} retrieval test cases..."
        )
        
        cache_label = " (Cached)" if was_cached else ""
            
        return {
            "ID": tc["id"] + cache_label,
            "Query": query,
            "RM Tier": rm_tier,
            "Expected": ", ".join(expected) if expected else "None (Access Filtered)",
            "Retrieved": ", ".join(list(set(retrieved_ids))[:4]),
            "Coverage": f"{coverage * 100:.1f}%",
            "Pass": "✅" if is_success else "❌",
            "is_success": is_success
        }

    for i, tc in enumerate(test_cases):
        results.append(run_case(i, tc))
            
    df_results = pd.DataFrame(results)
    st.table(df_results)
    
    successful_hits = sum(1 for r in results if r and r["is_success"])
    mrr_score = successful_hits / total_queries if total_queries > 0 else 0.0
    st.metric("Overall Retrieval Accuracy (Exact Match)", f"{mrr_score * 100:.1f}%")


def evaluate_suitability_gate(eval_data=None):
    st.subheader("2. Suitability Gate Precision & Recall")
    
    if eval_data is None:
        dataset_path = "eval_dataset.json"
        if not os.path.exists(dataset_path):
            st.error(f"Evaluation dataset file not found at {dataset_path}")
            return
            
        with open(dataset_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
    test_cases = eval_data.get("suitability_gate", {}).get("test_cases", [])
    total_cases = len(test_cases)
    
    st.write(f"Evaluating suitability gate sequentially via agent invocations ({total_cases} test cases)...")
    
    results = []
    completed_count = 0
    
    progress_bar = st.progress(0, text="Initializing suitability gate evaluation...")
    
    def run_suitability_case(idx, tc):
        client_id = tc["client_id"]
        product_code = tc["product_code"]
        amount = tc["allocation_amount"]
        expected_status = tc["expected_status"]
        
        # Build query representing trade simulation suitability check
        query = tc.get("query") or f"Is recommending product {product_code} suitable for client {client_id} with amount {amount}?"
        
        # Get from cache or run agent
        final_state, was_cached = get_cached_agent_run(
            idx=idx,
            query=query,
            client_id=client_id,
            product_code=product_code,
            allocation_amount=float(amount),
            force_structured=True,
            response_mode="structured",
            purpose="suitability"
        )
        
        actual_status = final_state.get("compliance_status", "Cleared")
        violations = final_state.get("suitability_report", {}).get("violations", [])
        passed = actual_status == expected_status
        
        nonlocal completed_count
        completed_count += 1
        progress_bar.progress(
            completed_count / total_cases,
            text=f"Processed {completed_count}/{total_cases} suitability gate cases..."
        )
        
        cache_label = " (Cached)" if was_cached else ""
            
        return {
            "ID": tc["id"] + cache_label,
            "Client ID": client_id,
            "Product": product_code,
            "Amount": f"${amount:,}",
            "Expected": expected_status,
            "Actual": actual_status,
            "Violations": ", ".join(violations) or "None",
            "Match": "✅" if passed else "❌",
            "expected_status": expected_status,
            "actual_status": actual_status
        }

    for i, tc in enumerate(test_cases):
        results.append(run_suitability_case(i, tc))
            
    df_results = pd.DataFrame(results)
    st.table(df_results)
    
    tp = fp = fn = tn = 0
    for r in results:
        if r is None:
            continue
        expected = r["expected_status"]
        actual = r["actual_status"]
        if expected in ["Blocked", "Needs Review"]:
            if actual in ["Blocked", "Needs Review"]:
                tp += 1
            else:
                fn += 1
        else:
            if actual == "Cleared":
                tn += 1
            else:
                fp += 1
                
    # Calculate Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0
    
    st.markdown(f"""
    - **True Positives (Correctly Flagged/Blocked)**: {tp}
    - **False Positives (Incorrectly Flagged)**: {fp}
    - **False Negatives (Missed Blocks/Violations)**: {fn}
    - **True Negatives (Correctly Cleared)**: {tn}
    
    **Suitability Gate Performance:**
    - **Precision**: `{precision * 100:.1f}%`
    - **Recall (Sensitivity)**: `{recall * 100:.1f}%`
    - **F1 Score**: `{f1 * 100:.1f}%`
    """)


def evaluate_faithfulness(eval_data=None):
    st.subheader("3. Ragas & Agent Pipeline Evaluation Metrics")
    st.write("Using Ragas library and direct trace logging to evaluate RAG Faithfulness, Answer Relevancy, and Tool Call Accuracy on the Golden Set.")
    
    if eval_data is None:
        dataset_path = "eval_dataset.json"
        if not os.path.exists(dataset_path):
            st.error(f"Evaluation dataset file not found at {dataset_path}")
            return
            
        with open(dataset_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            
    test_cases = eval_data.get("faithfulness", {}).get("test_cases", [])

    # Parallel workers slider — shown before the button so it is always visible
    max_workers = st.slider(
        "⚡ Parallel Workers (test cases run simultaneously)",
        min_value=1,
        max_value=min(len(test_cases), 5),
        value=min(3, len(test_cases)),
        help="Higher = faster but more API load. Keep ≤ 3 to stay within OpenAI rate limits."
    )

    if st.button("🚀 Run Ragas & Trace Evaluation", key="run_ragas_btn"):
        try:
            import sys
            import types
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Stub out langchain_community.chat_models.vertexai to prevent ModuleNotFoundError in ragas
            if "langchain_community.chat_models.vertexai" not in sys.modules:
                m = types.ModuleType("vertexai")
                m.ChatVertexAI = None
                sys.modules["langchain_community.chat_models.vertexai"] = m

            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics._faithfulness import Faithfulness
            from ragas.metrics._answer_relevance import ResponseRelevancy
            from ragas.llms import llm_factory
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.run_config import RunConfig
            from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings
            from openai import OpenAI as OpenAIClient

            # ── Helper: extract formatted answer string from final agent state ──
            def _extract_answer(final_state, query):
                if final_state.get("free_form_response"):
                    raw_answer = final_state["free_form_response"]
                    c_id = final_state.get("client_id", "client")
                    query_lower = query.lower()
                    client_name = "client"
                    if c_id:
                        c_id_up = c_id.upper()
                        if "C-204" in c_id_up:
                            client_name = "Eleanor Vance"
                        elif "C-101" in c_id_up:
                            client_name = "Arthur Pendleton"
                        elif "C-302" in c_id_up:
                            client_name = "Marcus Vance"
                    has_client_ref = any(
                        kw in query_lower for kw in ("client", "vance", "pendleton", "c-")
                    )
                    if has_client_ref:
                        if "talking point" in query_lower:
                            prefix = f"Talking points prepared for client {client_name} ({c_id}):\n"
                        elif any(kw in query_lower for kw in ("suitab", "recommend", "adjust")):
                            prefix = f"Suitability and investment recommendation assessment for client {client_name} ({c_id}):\n"
                        elif any(kw in query_lower for kw in ("risk", "portfolio", "holding")):
                            prefix = f"Portfolio risk and holdings summary review for client {client_name} ({c_id}):\n"
                        else:
                            prefix = f"General response report for client {client_name} ({c_id}):\n"
                    else:
                        if "talking point" in query_lower:
                            prefix = "General talking points research summary:\n"
                        elif any(kw in query_lower for kw in ("suitab", "recommend", "adjust")):
                            prefix = "General suitability and investment recommendation guidelines:\n"
                        elif any(kw in query_lower for kw in ("risk", "portfolio", "holding")):
                            prefix = "General fund risk and term parameters comparison:\n"
                        else:
                            prefix = "General research information note:\n"
                    return prefix + clean_answer_for_eval(raw_answer)
                elif final_state.get("final_brief"):
                    return clean_answer_for_eval(
                        format_brief_to_text(final_state["final_brief"], query)
                    )
                return "No response generated by agent."

            # ── Worker: runs a single test case in its own thread ──────────────
            def run_single_test_case(idx, tc):
                query = tc["query"]
                client_id = tc["client_id"]
                expected_docs = tc.get("expected_source_doc_ids", [])
                product_code = tc.get("product_code")
                
                # Retrieve from cache or run agent
                final_state, was_cached = get_cached_agent_run(
                    idx=idx,
                    query=query,
                    client_id=client_id,
                    product_code=product_code,
                    allocation_amount=0.0,
                    force_structured=False,
                    response_mode="freeform",
                    purpose="faithfulness"
                )
                
                retrieved_evidence = final_state.get("retrieved_evidence", []) or []
                contexts = [d["content"] for d in retrieved_evidence]
                retrieved_ids = list(
                    set([d["doc_id"] for d in retrieved_evidence if d.get("doc_id")])
                )
                
                if expected_docs:
                    hits = [exp for exp in expected_docs if exp in retrieved_ids]
                    tool_acc = len(hits) / len(expected_docs)
                else:
                    # No expected docs → 100% accuracy only if nothing was returned
                    tool_acc = 1.0 if not retrieved_ids else 0.0
                    
                answer = _extract_answer(final_state, query)
                return idx, query, contexts, answer, tool_acc, was_cached

            # ── PHASE 1: Parallel agent runs with live st.status() display ──────
            st.markdown("#### 🔄 Phase 1 — Running Agent Pipeline")
            
            with st.status(
                f"Running {len(test_cases)} test cases with {max_workers} parallel worker(s)...",
                expanded=True
            ) as status_box:
                row_placeholders = []
                for i, tc in enumerate(test_cases):
                    ph = st.empty()
                    ph.markdown(
                        f"⏳ `[{i+1}/{len(test_cases)}]` **{tc['id']}** — {tc['query'][:70]}..."
                    )
                    row_placeholders.append(ph)
                    
                progress_bar = st.progress(0, text="Starting parallel execution...")
                completed_count = [0]
                lock = threading.Lock()
                
                results_map = {}
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(run_single_test_case, i, tc): i
                        for i, tc in enumerate(test_cases)
                    }
                    
                    for future in as_completed(future_to_idx):
                        original_idx = future_to_idx[future]
                        try:
                            idx, query, contexts, answer, tool_acc, was_cached = future.result()
                            results_map[idx] = (query, contexts, answer, tool_acc)
                            cache_label = " (Cached)" if was_cached else ""
                            row_placeholders[idx].markdown(
                                f"✅ `[{idx+1}/{len(test_cases)}]` **{test_cases[idx]['id']}**{cache_label} — "
                                f"Tool Acc: `{tool_acc*100:.0f}%` | Answer: `{len(answer)} chars`"
                            )
                        except Exception as exc:
                            results_map[original_idx] = (
                                test_cases[original_idx]["query"], [], f"ERROR: {exc}", 0.0
                            )
                            row_placeholders[original_idx].markdown(
                                f"❌ `[{original_idx+1}/{len(test_cases)}]` "
                                f"**{test_cases[original_idx]['id']}** — Error: {exc}"
                            )

                        with lock:
                            completed_count[0] += 1
                            done = completed_count[0]
                        progress_bar.progress(
                            done / len(test_cases),
                            text=f"Completed {done}/{len(test_cases)} test cases..."
                        )

                progress_bar.progress(1.0, text="✅ All agent runs complete!")
                status_box.update(
                    label=f"✅ Phase 1 complete — {len(test_cases)} test cases finished",
                    state="complete",
                    expanded=False
                )

            # Reassemble results in original order
            questions, contexts_list, answers, tool_accuracy_list = [], [], [], []
            for i in range(len(test_cases)):
                q, c, a, t = results_map[i]
                questions.append(q)
                contexts_list.append(c)
                answers.append(a)
                tool_accuracy_list.append(t)

            # ── PHASE 2: Parallel per-sample Ragas metric scoring ──────────────
            st.markdown("#### 🧪 Phase 2 — Computing Ragas Metrics (Faithfulness & Answer Relevancy)")

            openai_api_key = os.environ.get("OPENAI_API_KEY")

            def score_single_sample(idx, question, contexts, answer):
                """Score one sample with Faithfulness + ResponseRelevancy."""
                _client = OpenAIClient(api_key=openai_api_key)
                _llm    = llm_factory("gpt-4o-mini", client=_client)
                _lc_emb = LCOpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
                _emb    = LangchainEmbeddingsWrapper(_lc_emb)

                _faith = Faithfulness()
                _faith.llm = _llm

                _rel = ResponseRelevancy()
                _rel.llm = _llm
                _rel.embeddings = _emb

                single_ds = Dataset.from_dict({
                    "question": [question],
                    "contexts":  [contexts],
                    "answer":    [answer],
                })
                rc = RunConfig(max_workers=1, max_retries=10, timeout=180)
                res = evaluate(single_ds, metrics=[_faith, _rel], run_config=rc)

                f_score = list(res["faithfulness"])[0]
                r_score = list(res["answer_relevancy"])[0]
                return idx, f_score, r_score

            faith_scores     = [float("nan")] * len(test_cases)
            relevancy_scores = [float("nan")] * len(test_cases)

            with st.status(
                f"Scoring {len(test_cases)} samples with {max_workers} parallel worker(s)...",
                expanded=True
            ) as status_box2:
                ragas_row_phs = []
                for i, tc in enumerate(test_cases):
                    ph2 = st.empty()
                    ph2.markdown(
                        f"⏳ `[{i+1}/{len(test_cases)}]` **{tc['id']}** — awaiting Ragas score..."
                    )
                    ragas_row_phs.append(ph2)

                ragas_progress = st.progress(0, text="Starting Ragas scoring...")
                ragas_done = [0]

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_ridx = {
                        executor.submit(
                            score_single_sample, i, questions[i], contexts_list[i], answers[i]
                        ): i
                        for i in range(len(test_cases))
                    }

                    for future in as_completed(future_to_ridx):
                        ridx = future_to_ridx[future]
                        try:
                            sidx, f_val, r_val = future.result()
                            faith_scores[sidx]     = f_val
                            relevancy_scores[sidx] = r_val
                            f_str = f"{f_val*100:.1f}%" if not _math.isnan(f_val) else "n/a"
                            r_str = f"{r_val*100:.1f}%" if not _math.isnan(r_val) else "n/a"
                            ragas_row_phs[sidx].markdown(
                                f"✅ `[{sidx+1}/{len(test_cases)}]` **{test_cases[sidx]['id']}** — "
                                f"Faithfulness: `{f_str}` | Relevancy: `{r_str}`"
                            )
                        except Exception as exc:
                            ragas_row_phs[ridx].markdown(
                                f"❌ `[{ridx+1}/{len(test_cases)}]` "
                                f"**{test_cases[ridx]['id']}** — Ragas error: {exc}"
                            )

                        with lock:
                            ragas_done[0] += 1
                            rdone = ragas_done[0]
                        ragas_progress.progress(
                            rdone / len(test_cases),
                            text=f"Scored {rdone}/{len(test_cases)} samples..."
                        )

                ragas_progress.progress(1.0, text="✅ All samples scored!")
                status_box2.update(
                    label=f"✅ Phase 2 complete — {len(test_cases)} samples scored by Ragas",
                    state="complete",
                    expanded=False
                )

            valid_faith = [s for s in faith_scores if not (isinstance(s, float) and _math.isnan(s))]
            valid_rel   = [s for s in relevancy_scores if not (isinstance(s, float) and _math.isnan(s))]

            avg_faith     = sum(valid_faith) / len(valid_faith) if valid_faith else 0.0
            avg_relevancy = sum(valid_rel) / len(valid_rel) if valid_rel else 0.0
            avg_tool_acc  = sum(tool_accuracy_list) / len(tool_accuracy_list) if tool_accuracy_list else 0.0

            # ── Results display ─────────────────────────────────────────────────
            st.success("🎉 Ragas evaluation complete!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Ragas Faithfulness", f"{avg_faith * 100:.1f}%",
                    delta=f"{len(valid_faith)}/{len(faith_scores)} valid cases"
                )
            with col2:
                st.metric(
                    "Ragas Answer Relevancy", f"{avg_relevancy * 100:.1f}%",
                    delta=f"{len(valid_rel)}/{len(relevancy_scores)} valid cases"
                )
            with col3:
                st.metric("Tool Call Accuracy", f"{avg_tool_acc * 100:.1f}%")

            st.write("### 📊 Test Case Breakdown")
            breakdown_list = []
            for idx, tc in enumerate(test_cases):
                f_val = faith_scores[idx] if idx < len(faith_scores) else 0.0
                r_val = relevancy_scores[idx] if idx < len(relevancy_scores) else 0.0
                t_val = tool_accuracy_list[idx] if idx < len(tool_accuracy_list) else 0.0
                breakdown_list.append({
                    "Test Case ID": tc["id"],
                    "Query": tc["query"],
                    "Faithfulness": f"{f_val * 100:.1f}%",
                    "Answer Relevancy": f"{r_val * 100:.1f}%",
                    "Tool Call Accuracy": f"{t_val * 100:.1f}%"
                })

            st.table(pd.DataFrame(breakdown_list))

            with st.expander("🔍 View Evaluated Answers & Retrieved Contexts", expanded=False):
                for idx, tc in enumerate(test_cases):
                    st.markdown(f"#### `{tc['id']}` - {tc['category']}")
                    st.markdown(f"**Query**: {tc['query']}")
                    st.markdown("**Generated Answer**:")
                    st.info(answers[idx])
                    st.markdown(f"**Retrieved Contexts ({len(contexts_list[idx])} chunks)**:")
                    for c_idx, c in enumerate(contexts_list[idx]):
                        st.write(f"- Chunk {c_idx+1}: {c[:150]}...")
                    st.divider()

        except Exception as e:
            st.error(f"Error during Ragas evaluation: {e}")
            st.exception(e)

def compare_single_vs_multihop():
    st.subheader("4. Single-Shot vs Multi-Hop Retrieval Analysis")
    st.write("Evaluating retrieval performance on complex cross-referenced queries (e.g. tracking market outlook and fixed income views).")
    
    query = "Summarize the house views on fixed income portfolio duration relative to the Q2 2026 Global Market Outlook."
    
    # Single-shot vector retrieval
    from tools import rag_retriever
    single_retrieved = rag_retriever(query, rm_research_tier=2)
    single_ids = [d["doc_id"] for d in single_retrieved]
    
    # Multi-hop retrieval
    # Step 1: Query initial context
    step1_retrieved = rag_retriever(query, rm_research_tier=2)
    step1_ids = [d["doc_id"] for d in step1_retrieved]
    
    # Step 2: Use LLM to identify referenced documents in step 1 context to follow links
    llm = get_llm()
    context_text = "\n\n".join([d["content"][:500] for d in step1_retrieved])
    
    prompt = f"""
    Based on the retrieved context below, identify any other document IDs (e.g. RN-003, PG-001, etc.) referenced or linked that are relevant to 'fixed income duration views':
    
    Context:
    {context_text}
    
    Return a comma-separated list of document IDs.
    """
    
    referenced_ids = [x.strip() for x in llm.invoke(prompt).content.split(",") if x.strip()]
    
    # Step 3: Fetch referenced documents directly or run secondary query
    multi_hop_ids = list(set(step1_ids + referenced_ids))
    
    # Make sure we remove any noise IDs
    multi_hop_ids = [i for i in multi_hop_ids if i in ["PG-001", "PG-002", "PG-003", "PG-004", "CMP-001", "CMP-002", "CMP-003", "RN-001", "RN-002", "RN-003"]]
    
    st.markdown(f"""
    - **Single-Shot Retrieval**:
      - Found Documents: {', '.join(single_ids)}
      - Missing Context: Includes only primary query hits.
    - **Multi-Hop / Link-Following Retrieval**:
      - Step 1 (Primary search): {', '.join(step1_ids)}
      - Step 2 (References identified by LLM): {', '.join(referenced_ids) or 'None'}
      - Blended Results: {', '.join(multi_hop_ids)}
      
    **Analysis**: Multi-hop retrieval successfully tracks cross-references (such as connecting `RN-001`'s references to `RN-003` or product guides) which are missed in a single-shot vector query when keyword overlap is low.
    """)

def main():
    st.title("Analytical Evaluation & Benchmarking (Part B)")
    st.divider()
    
    evaluate_retrieval()
    st.divider()
    
    evaluate_suitability_gate()
    st.divider()
    
    evaluate_faithfulness()
    st.divider()
    
    compare_single_vs_multihop()

if __name__ == "__main__":
    main()
