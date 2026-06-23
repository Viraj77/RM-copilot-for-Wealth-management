import os
import pytest
from client_db import get_client_profile
from tools import portfolio_lookup, market_data_tool, suitability_checker, rag_retriever
from agent import create_agent

def test_client_db():
    profile = get_client_profile("C-204")
    assert profile is not None
    assert profile["name"] == "Eleanor Vance"
    assert profile["risk_profile"] == "Balanced"

def test_portfolio_lookup():
    res = portfolio_lookup("C-204")
    assert "error" not in res
    assert res["client_id"] == "C-204"

def test_market_data_tool():
    res = market_data_tool("PG-001")
    assert "error" not in res
    assert res["product_code"] == "PG-001"
    assert "HBGF" in res["product_name"]

def test_suitability_checker():
    # Suitable trade
    res1 = suitability_checker("C-204", "PG-001", 10000)
    assert res1["status"] == "Cleared"
    
    # Unsuitable risk profile trade (Conservative client trying to buy high-risk SCN)
    res2 = suitability_checker("C-101", "SCN-US-24", 10000)
    assert res2["status"] == "Blocked"
    assert "RULE-016" in res2["violations"]

def test_rag_retriever():
    res = rag_retriever("Balanced Growth Fund", rm_research_tier=2)
    assert len(res) > 0
    # The first document or some documents should carry PG-001
    doc_ids = [doc["doc_id"] for doc in res]
    assert "PG-001" in doc_ids or "CMP-001" in doc_ids or "CMP-003" in doc_ids

def test_agent_graph():
    # Verify that the graph compiles and can be invoked
    agent = create_agent()
    assert agent is not None
    
    # Test a simple cleared scenario
    initial_state = {
        "query": "Prepare talking points for Eleanor Vance (C-204) reviewing HBGF",
        "client_id": "C-204",
        "product_code": "PG-001",
        "allocation_amount": 10000.0,
        "client_profile": get_client_profile("C-204"),
        "retrieved_evidence": [],
        "compliance_status": "Cleared",
        "escalated": False,
        "review_notes": None,
        "final_brief": None
    }
    
    config = {"configurable": {"thread_id": "test_thread_1"}}
    state = agent.invoke(initial_state, config)
    
    # It should finish and output final brief
    assert state is not None
    assert state.get("final_brief") is not None
    assert state["final_brief"]["client_id"] == "C-204"
    assert state["compliance_status"] == "Cleared"

def test_classify_intent_structured():
    from agent import classify_intent_node
    initial_state = {
        "query": "Prepare talking points for Eleanor Vance (C-204) reviewing HBGF",
        "client_id": "C-204",
        "product_code": "PG-001",
        "allocation_amount": 10000.0,
        "force_structured": False
    }
    res = classify_intent_node(initial_state)
    assert res["response_mode"] == "structured"

def test_classify_intent_freeform():
    from agent import classify_intent_node
    initial_state = {
        "query": "Compare HBGF and HCIF side by side",
        "client_id": None,
        "product_code": None,
        "allocation_amount": 0.0,
        "force_structured": False
    }
    res = classify_intent_node(initial_state)
    assert res["response_mode"] == "freeform"

def test_freeform_answer_node():
    from agent import free_form_answer_node
    initial_state = {
        "query": "Compare HBGF and HCIF side by side",
        "client_id": None,
        "product_code": None,
        "allocation_amount": 0.0,
        "response_mode": "freeform",
        "client_profile": None,
        "retrieved_evidence": [
            {"doc_id": "PG-001", "type": "product", "sensitivity": "Public", "content": "HBGF is a balanced fund with moderate risk and growth target."},
            {"doc_id": "PG-002", "type": "product", "sensitivity": "Public", "content": "HCIF is a conservative income fund focusing on high quality fixed income."}
        ]
    }
    res = free_form_answer_node(initial_state)
    assert res["free_form_response"] is not None
    assert "HBGF" in res["free_form_response"] or "HCIF" in res["free_form_response"]
    assert res["final_brief"] is None

def test_out_of_context_query():
    # Verify that an out of context query does not crash check_suitability_node or the graph,
    # and exits early returning the standardized reply without running retriever/portfolio nodes.
    agent = create_agent()
    initial_state = {
        "query": "What is the capital of France?",
        "client_id": "C-204",  # Even if a client ID is provided, routing should bypass gather_portfolio
        "product_code": None,
        "allocation_amount": 0.0,
        "force_structured": False,
        "client_profile": None,
        "retrieved_evidence": [],
        "compliance_status": "Cleared",
        "escalated": False,
        "review_notes": None,
        "final_brief": None,
        "free_form_response": None
    }
    config = {"configurable": {"thread_id": "test_thread_out_of_context"}}
    state = agent.invoke(initial_state, config)
    assert state is not None
    assert state.get("is_out_of_context") is True
    assert state.get("free_form_response") == "I can only provide support related to wealth management and client advisory queries."
    assert state.get("client_profile") is None  # gather_portfolio bypassed
    assert len(state.get("retrieved_evidence", [])) == 0  # gather_research bypassed

