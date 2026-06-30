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

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class EvalScore(BaseModel):
    faithfulness_score: float = Field(description="Score between 0.0 and 1.0. 1.0 if the answer is completely supported by the provided context. 0.0 if hallucinated.")
    faithfulness_reason: str = Field(description="Brief reason for the faithfulness score.")
    relevancy_score: float = Field(description="Score between 0.0 and 1.0. 1.0 if the answer directly and fully addresses the question.")
    relevancy_reason: str = Field(description="Brief reason for the relevancy score.")

def get_evaluator():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    structured_llm = llm.with_structured_output(EvalScore)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert evaluator for a Wealth Management AI Copilot.
You are evaluating an agent's response based on two metrics:
1. Faithfulness: Is the Answer entirely supported by the Context? (Score 0.0 to 1.0)
2. Answer Relevancy: Does the Answer directly address the User Question? (Score 0.0 to 1.0)"""),
        ("human", """User Question: {question}
        
Retrieved Context:
{context}

Agent Answer:
{answer}
""")
    ])
    return prompt | structured_llm

def run_ragas_evaluation():
    golden_set_path = "data/golden_set/evaluation_queries.json"
    if not os.path.exists(golden_set_path):
        print(f"Golden set not found at {golden_set_path}")
        return

    with open(golden_set_path, "r") as f:
        queries = json.load(f)
    
    questions = []
    answers = []
    contexts_list = []
    
    print(f"Loaded {len(queries)} test queries for RAGAS evaluation.\n")
    
    for idx, q in enumerate(queries):
        print(f"[{idx+1}/{len(queries)}] Executing Query: {q['query']}")
        initial_state = {
            "messages": [HumanMessage(content=q['query'])],
            "client_id": q['client_id'],
            "rm_tier": q['rm_tier'],
            "current_step": 0,
        }
        
        final_response_raw = ""
        contexts = []
        
        try:
            for event in copilot_app.stream(initial_state):
                for node_name, state_update in event.items():
                    if not isinstance(state_update, dict):
                        continue
                    messages = state_update.get("messages", [])
                    if not messages:
                        continue
                    
                    # Capture tool output for contexts
                    if node_name == "tools":
                        for msg in messages:
                            if hasattr(msg, 'name'):
                                if msg.name == "rag_retriever_tool":
                                    try:
                                        data = json.loads(msg.content)
                                        for res in data.get("results", []):
                                            content = res.get("content", "")
                                            if content:
                                                contexts.append(content)
                                    except Exception as e:
                                        print(f"  Warning: Failed to parse RAG tool output: {e}")
                                else:
                                    # For all other tools, append the raw output to contexts
                                    contexts.append(f"[{msg.name} output]: {msg.content}")

                    # Capture final agent response
                    if node_name == "agent":
                        msg = messages[-1]
                        if msg.type == "ai" and not getattr(msg, 'tool_calls', []):
                            final_response_raw = msg.content
                            
        except Exception as e:
            print(f"  Error executing query {q['id']}: {e}")
            final_response_raw = "Execution Error"
            
        # Pass through guardrails to get the final output seen by the user
        gate = run_compliance_gate(final_response_raw)
        safe_output = gate['safe_output'] if isinstance(gate, dict) and 'safe_output' in gate else final_response_raw
        
        # If no contexts were retrieved, provide a default so RAGAS doesn't crash
        if not contexts:
            contexts = ["No documents were retrieved from the knowledge base for this query."]
            
        questions.append(q['query'])
        answers.append(safe_output)
        contexts_list.append(contexts)
        
    print("\n" + "="*80)
    print("Starting LLM-as-a-Judge evaluation (scoring Faithfulness and Relevancy)...")
    print("="*80 + "\n")
    
    evaluator = get_evaluator()
    
    results_list = []
    total_faithfulness = 0.0
    total_relevancy = 0.0
    
    for q, a, ctx in zip(questions, answers, contexts_list):
        context_str = "\n\n".join(ctx)
        print(f"Evaluating: '{q}'")
        try:
            score = evaluator.invoke({"question": q, "context": context_str, "answer": a})
            total_faithfulness += score.faithfulness_score
            total_relevancy += score.relevancy_score
            
            res_dict = {
                "question": q,
                "faithfulness_score": score.faithfulness_score,
                "faithfulness_reason": score.faithfulness_reason,
                "relevancy_score": score.relevancy_score,
                "relevancy_reason": score.relevancy_reason
            }
            results_list.append(res_dict)
            print(f"  -> Faithfulness: {score.faithfulness_score} | Relevancy: {score.relevancy_score}")
        except Exception as e:
            print(f"  -> Evaluation failed: {e}")
            
    num_queries = len(questions)
    avg_faithfulness = total_faithfulness / num_queries if num_queries else 0.0
    avg_relevancy = total_relevancy / num_queries if num_queries else 0.0
    
    final_metrics = {
        "average_faithfulness": avg_faithfulness,
        "average_relevancy": avg_relevancy,
        "average_faithfulness_percent": f"{avg_faithfulness * 100:.2f}%",
        "average_relevancy_percent": f"{avg_relevancy * 100:.2f}%",
        "detailed_results": results_list
    }
    
    print("\n--- Final Evaluation Metrics ---")
    print(f"Average Faithfulness: {final_metrics['average_faithfulness']:.2f} ({final_metrics['average_faithfulness_percent']})")
    print(f"Average Relevancy: {final_metrics['average_relevancy']:.2f} ({final_metrics['average_relevancy_percent']})")
    
    with open("ragas_evaluation_results.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print("\nResults saved to ragas_evaluation_results.json")

if __name__ == "__main__":
    run_ragas_evaluation()
