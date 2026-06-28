"""
JSON Converter Module
Converts OpenAI Assistant structured message format (with file_citation annotations)
to/from the test case format used in eval_dataset.json.

Run from command line or import as a module.
"""

import json
import os
import re

def convert_assistant_to_eval_case(assistant_json, category="General", case_id="CASE-001", query="Default Query", rm_tier=2):
    """
    Converts an OpenAI Assistant Message JSON into an eval_dataset.json test case.
    Extracts citations from the annotations and maps them to expected_doc_ids.
    """
    if isinstance(assistant_json, str):
        data = json.loads(assistant_json)
    else:
        data = assistant_json

    content = data.get("content", [])
    expected_doc_ids = []
    
    for item in content:
        if item.get("type") == "text":
            text_data = item.get("text", {})
            annotations = text_data.get("annotations", [])
            for ann in annotations:
                if ann.get("type") == "file_citation":
                    file_cit = ann.get("file_citation", {})
                    file_id = file_cit.get("file_id")
                    if file_id and file_id not in expected_doc_ids:
                        expected_doc_ids.append(file_id)

    # Reconstruct category from category parameter or annotations
    test_case = {
        "id": case_id,
        "category": category,
        "query": query,
        "rm_tier": rm_tier,
        "expected_doc_ids": expected_doc_ids,
        "must_not_contain": [],
        "rationale": f"Auto-generated case from assistant message citing {', '.join(expected_doc_ids)}.",
        "reference_docs": [f"docs/{doc_id}" for doc_id in expected_doc_ids]
    }
    return test_case

def convert_eval_case_to_assistant(eval_case, value_text="This is an auto-generated response answering the query."):
    """
    Converts an eval_dataset.json test case into the OpenAI Assistant Message JSON format.
    Constructs annotations for each expected doc id.
    """
    expected_docs = eval_case.get("expected_doc_ids", [])
    annotations = []
    
    # Construct annotated value
    value = value_text
    
    for i, doc_id in enumerate(expected_docs):
        citation_text = f"【{i+1}:0†{doc_id}】"
        start_idx = len(value)
        value += " " + citation_text
        end_idx = len(value)
        
        annotations.append({
            "type": "file_citation",
            "text": citation_text,
            "file_citation": {
                "file_id": doc_id
            },
            "start_index": start_idx + 1,
            "end_index": end_idx
        })

    assistant_message = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": {
                    "value": value,
                    "annotations": annotations
                }
            }
        ]
    }
    return assistant_message

if __name__ == "__main__":
    # Example usage
    sample_openai = {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": {
            "value": "You are eligible for remote work after 90 days of employment.【4:0†company_policy_2026.pdf】",
            "annotations": [
              {
                "type": "file_citation",
                "text": "【4:0†company_policy_2026.pdf】",
                "file_citation": {
                  "file_id": "company_policy_2026.pdf"
                },
                "start_index": 72,
                "end_index": 102
              }
            ]
          }
        }
      ]
    }
    
    print("=== Converting OpenAI Assistant format -> eval_dataset.json case ===")
    converted_case = convert_assistant_to_eval_case(sample_openai, category="HR Policy", case_id="RET-HR-001", query="When can I work remotely?")
    print(json.dumps(converted_case, indent=2))
    
    print("\n=== Converting eval_dataset.json case -> OpenAI Assistant format ===")
    converted_assistant = convert_eval_case_to_assistant(converted_case, value_text="Remote work eligibility rules are defined in the policy document.")
    print(json.dumps(converted_assistant, indent=2))
