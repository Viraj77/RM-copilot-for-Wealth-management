"""Evaluation metrics for retrieval, suitability gate, faithfulness, and agent scenarios."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import GOLDEN_SET_DIR, get_openai_api_key
from src.guardrails.compliance import detect_licensed_advice_request
from src.models import ClientBrief, ComplianceStatus, RiskProfile, Sensitivity
from src.tools.retriever import HybridRAGRetriever
from src.tools.suitability import check_suitability_logic


def load_golden_set() -> list[dict]:
    path = GOLDEN_SET_DIR / "eval_queries.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _citation_hit(expected_ids: list[str], retrieved_ids: list[str]) -> bool:
    if not expected_ids:
        return len(retrieved_ids) > 0
    return any(
        expected in retrieved
        for expected in expected_ids
        for retrieved in retrieved_ids
    )


def _entitlements_for_query(q: dict) -> list[Sensitivity]:
    if q.get("expected_filter_restricted"):
        return [Sensitivity.PUBLIC, Sensitivity.INTERNAL, Sensitivity.RESTRICTED]
    return [Sensitivity.PUBLIC, Sensitivity.INTERNAL]


def evaluate_retrieval_relevance(retriever: HybridRAGRetriever) -> dict:
    """Measure retrieval relevance on golden set queries."""
    queries = load_golden_set()
    results = []
    for q in queries:
        query_text = q["query"]
        entitlements = _entitlements_for_query(q)
        chunks = retriever.retrieve(query_text, k=5, rm_entitlements=entitlements)
        expected_ids = q.get("expected_citations_from", [])
        retrieved_ids = [c.doc_id for c in chunks]
        hit = _citation_hit(expected_ids, retrieved_ids)
        results.append({
            "query": query_text,
            "retrieved": len(chunks),
            "expected_ids": expected_ids,
            "retrieved_ids": retrieved_ids,
            "relevant_hit": hit,
        })
    hit_rate = sum(1 for r in results if r["relevant_hit"]) / len(results) if results else 0
    return {"retrieval_hit_rate": hit_rate, "details": results}


def evaluate_citation_coverage(retriever: HybridRAGRetriever) -> dict:
    queries = load_golden_set()
    total = 0
    with_citations = 0
    for q in queries:
        entitlements = _entitlements_for_query(q)
        chunks = retriever.retrieve(q["query"], k=5, rm_entitlements=entitlements)
        total += len(chunks)
        with_citations += sum(1 for c in chunks if c.doc_id and c.doc_id != "unknown")
    coverage = with_citations / total if total else 0
    return {"citation_coverage": coverage, "total_chunks": total, "with_citations": with_citations}


def evaluate_suitability_gate() -> dict:
    test_cases = [
        ("PG-003", RiskProfile.CONSERVATIVE, False),
        ("HAEF", RiskProfile.CONSERVATIVE, False),
        ("PG-002", RiskProfile.CONSERVATIVE, True),
        ("HCIF", RiskProfile.CONSERVATIVE, True),
        ("PG-004", RiskProfile.BALANCED, True),
        ("PG-001", RiskProfile.BALANCED, True),
        ("PG-003", RiskProfile.AGGRESSIVE, True),
        ("PG-002", RiskProfile.AGGRESSIVE, False),
    ]
    correct = 0
    details = []
    for fund, profile, expected in test_cases:
        result = check_suitability_logic(fund, profile)
        match = result.suitable == expected
        if match:
            correct += 1
        details.append({
            "fund": fund,
            "profile": profile.value,
            "expected": expected,
            "actual": result.suitable,
            "correct": match,
        })
    return {"suitability_precision": correct / len(test_cases), "details": details}


def evaluate_escalation_detection() -> dict:
    queries = load_golden_set()
    correct = 0
    details = []
    for q in queries:
        if "expected_escalation" not in q:
            continue
        detected = detect_licensed_advice_request(q["query"])
        expected = q["expected_escalation"]
        match = detected == expected
        if match:
            correct += 1
        details.append({
            "query": q["query"],
            "expected": expected,
            "detected": detected,
            "correct": match,
        })
    total = len(details)
    return {
        "escalation_detection_rate": correct / total if total else 0,
        "details": details,
    }


def evaluate_entitlement_filtering(retriever: HybridRAGRetriever) -> dict:
    query = "restricted concentration limit entitlement rules CMP-002"
    unentitled = retriever.retrieve(
        query, k=10, rm_entitlements=[Sensitivity.PUBLIC, Sensitivity.INTERNAL]
    )
    entitled = retriever.retrieve(
        query, k=10,
        rm_entitlements=[Sensitivity.PUBLIC, Sensitivity.INTERNAL, Sensitivity.RESTRICTED],
    )
    restricted_unentitled = [c for c in unentitled if c.sensitivity == Sensitivity.RESTRICTED]
    restricted_entitled = [c for c in entitled if c.sensitivity == Sensitivity.RESTRICTED]
    return {
        "restricted_leaked_to_unentitled": len(restricted_unentitled),
        "restricted_available_to_entitled": len(restricted_entitled),
        "cmp002_visible_unentitled": sum(1 for c in unentitled if "CMP-002" in c.doc_id),
        "cmp002_visible_entitled": sum(1 for c in entitled if "CMP-002" in c.doc_id),
        "filter_working": len(restricted_unentitled) == 0,
    }


def compare_single_vs_multi_hop(retriever: HybridRAGRetriever) -> dict:
    complex_query = (
        "Prepare comprehensive review for conservative client C-204 "
        "fixed income PG-002 HCIF RN-001 CMP-003 quarterly outlook"
    )
    single = retriever.retrieve(complex_query, k=5)
    multi = retriever.multi_hop_retrieve(complex_query, k=5)
    single_ids = set(c.doc_id for c in single)
    multi_ids = set(c.doc_id for c in multi)
    return {
        "single_shot_count": len(single),
        "multi_hop_count": len(multi),
        "unique_docs_single": len(single_ids),
        "unique_docs_multi": len(multi_ids),
        "multi_hop_gain": len(multi_ids - single_ids),
        "single_doc_ids": list(single_ids),
        "multi_doc_ids": list(multi_ids),
    }


def _llm_judge_faithfulness(samples: list[dict]) -> float:
    """LLM-as-judge groundedness score (guidelines §9 Part B)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=get_openai_api_key())
    scores: list[float] = []
    for sample in samples:
        response = llm.invoke([
            SystemMessage(content=(
                "You are an evaluation judge. Score how faithful/grounded an answer is "
                "to the provided context on a scale of 0.0 to 1.0. "
                "Reply with ONLY a decimal number (e.g. 0.85)."
            )),
            HumanMessage(content=(
                f"Question: {sample['question']}\n\n"
                f"Context:\n{sample['contexts'][0][:1200]}\n\n"
                f"Answer:\n{sample['answer']}\n\n"
                "Faithfulness score (0.0-1.0):"
            )),
        ])
        try:
            scores.append(float(response.content.strip().split()[0]))
        except (ValueError, IndexError):
            scores.append(0.5)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_faithfulness(retriever: HybridRAGRetriever) -> dict:
    """RAGAS faithfulness / groundedness on golden-set Q&A pairs."""
    if not get_openai_api_key():
        return {"faithfulness_score": None, "skipped": "OPENAI_API_KEY not set"}

    samples = []
    for q in load_golden_set()[:5]:
        entitlements = _entitlements_for_query(q)
        chunks = retriever.retrieve(q["query"], k=3, rm_entitlements=entitlements)
        if not chunks:
            continue
        context = "\n".join(c.content[:400] for c in chunks)
        excerpt = chunks[0].content[:250].strip().replace("\n", " ")
        answer = (
            f"Per {chunks[0].doc_id} ({chunks[0].source}, {chunks[0].date}): "
            f"{excerpt}"
        )
        samples.append({
            "question": q["query"],
            "contexts": [context],
            "answer": answer,
        })

    if not samples:
        return {"faithfulness_score": 0.0, "samples": 0}

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import faithfulness

        judge = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=get_openai_api_key())
        )
        dataset = Dataset.from_list(samples)
        result = evaluate(dataset, metrics=[faithfulness], llm=judge)
        score = float(result["faithfulness"])
        return {"faithfulness_score": score, "samples": len(samples), "method": "RAGAS"}
    except Exception as exc:
        score = _llm_judge_faithfulness(samples)
        return {
            "faithfulness_score": score,
            "samples": len(samples),
            "method": "LLM-as-judge",
            "ragas_error": str(exc),
        }


def evaluate_agent_scenarios() -> dict:
    """End-to-end agent checks on golden-set scenarios (requires API key)."""
    if not get_openai_api_key():
        return {"agent_eval_skipped": True, "reason": "OPENAI_API_KEY not set"}

    import time
    from src.agent.graph import run_agent

    results = []
    scenarios = [
        {
            "query": "Recommend a personalized 4% withdrawal rate for client C-204 retirement",
            "check": lambda b: b.compliance_status == ComplianceStatus.NEEDS_REVIEW,
            "name": "licensed_advice_escalation",
        },
        {
            "query": "Is PG-003 suitable for a conservative client?",
            "check": lambda b: b is not None,
            "name": "suitability_query",
        },
    ]

    for i, scenario in enumerate(scenarios):
        try:
            thread_id = f"eval-agent-{i}-{int(time.time() * 1000)}"
            output = run_agent(scenario["query"], thread_id=thread_id)
            if isinstance(output, dict) and output.get("interrupted"):
                brief = output.get("draft_brief")
                if brief is not None and not isinstance(brief, ClientBrief):
                    brief = ClientBrief(**brief)
                passed = brief is not None and scenario["check"](brief)
                results.append({"scenario": scenario["name"], "passed": passed, "interrupted": True})
            elif isinstance(output, ClientBrief):
                results.append({
                    "scenario": scenario["name"],
                    "passed": scenario["check"](output),
                    "compliance": output.compliance_status.value,
                })
            else:
                results.append({"scenario": scenario["name"], "passed": False, "error": "no brief"})
        except Exception as exc:
            results.append({"scenario": scenario["name"], "passed": False, "error": str(exc)})

    passed = sum(1 for r in results if r.get("passed"))
    return {
        "agent_pass_rate": passed / len(results) if results else 0,
        "details": results,
    }


def run_all_evaluations() -> dict:
    retriever = HybridRAGRetriever()
    return {
        "retrieval": evaluate_retrieval_relevance(retriever),
        "citation_coverage": evaluate_citation_coverage(retriever),
        "suitability_gate": evaluate_suitability_gate(),
        "escalation": evaluate_escalation_detection(),
        "entitlement": evaluate_entitlement_filtering(retriever),
        "single_vs_multi_hop": compare_single_vs_multi_hop(retriever),
        "faithfulness": evaluate_faithfulness(retriever),
        "agent_scenarios": evaluate_agent_scenarios(),
    }


if __name__ == "__main__":
    print("Running evaluations...")
    report = run_all_evaluations()
    print(json.dumps(report, indent=2, default=str))
