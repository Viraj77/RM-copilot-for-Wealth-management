import os
import sys
import json
from pathlib import Path

# Setup path and env
sys.path.insert(0, os.path.abspath('.'))
from dotenv import load_dotenv
load_dotenv('.env')

from langchain_core.messages import HumanMessage
from src.agent import copilot_app
from src.guardrails import run_compliance_gate, verify_entitlements

print("Environment loaded.")

with open("data/golden_set/evaluation_queries.json", "r") as f:
    queries = json.load(f)

print(f"Loaded {len(queries)} test queries.")

def evaluate_query(q):
    print(f"\n{'='*80}")
    print(f"TEST: {q['id']} ({q['category']})")
    print(f"QUERY: {q['query']}")
    print(f"EXPECTED: {q['expected_behavior']}")
    print(f"{'='*80}")
    
    initial_state = {
        "messages": [HumanMessage(content=q['query'])],
        "client_id": q['client_id'],
        "rm_tier": q['rm_tier'],
        "current_step": 0,
    }
    
    final_response = ""
    
    # Run Agent
    try:
        for event in copilot_app.stream(initial_state):
            for node_name, state_update in event.items():
                if node_name == "agent":
                    msg = state_update["messages"][-1]
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"  [Tool Used] -> {tc['name']}")
                    elif hasattr(msg, 'content') and msg.content:
                        final_response = msg.content
    except Exception as e:
        print(f"Error during agent execution: {e}")
        import traceback
        traceback.print_exc()
        return
                        
    # Run Guardrails
    print("\n--- Guardrail Results ---")
    entitlement_safe = verify_entitlements(final_response, q['rm_tier'])
    print(f"Entitlements Pass: {entitlement_safe}")
    
    gate = run_compliance_gate(final_response)
    print(f"Compliance Status: {gate['status']}")
    for flag in gate['flags']:
        print(f"  Flag: {flag}")
        
    print("\n--- Final Output snippet ---")
    print(gate['safe_output'][:300] + "...")

# Run all
for q in queries:
    evaluate_query(q)
