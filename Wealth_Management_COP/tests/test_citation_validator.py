"""
Tests for Citation Hallucination Validator — Phase 8.
"""

import json

from src.guardrails.citation_validator import (
    extract_retrieved_chunk_ids,
    validate_citations,
    validate_and_patch_brief,
)


def _make_tool_message(content: str):
    from langchain_core.messages import ToolMessage
    return ToolMessage(content=content, tool_call_id="call_123")


class TestCitationValidator:

    def test_extract_retrieved_ids_from_json(self):
        """Should parse chunk_ids from rag_retriever_tool JSON output."""
        mock_output = {
            "success": True,
            "results": [
                {"chunk_id": "doc1_0000"},
                {"chunk_id": "doc2_0001"}
            ]
        }
        msg = _make_tool_message(json.dumps(mock_output))
        ids = extract_retrieved_chunk_ids([msg])
        assert ids == {"doc1_0000", "doc2_0001"}

    def test_extract_retrieved_ids_from_regex(self):
        """Should fallback to regex if JSON parsing fails."""
        bad_json = 'Here are the results: "chunk_id": "doc3_0000", "chunk_id": "doc4_0001"'
        msg = _make_tool_message(bad_json)
        ids = extract_retrieved_chunk_ids([msg])
        assert ids == {"doc3_0000", "doc4_0001"}

    def test_validate_citations_all_valid(self):
        """When all citations match retrieved chunks, valid=True."""
        retrieved = {"doc1_0000", "doc2_0000"}
        brief = {
            "recommendations": [
                {
                    "citations": [
                        {"chunk_id": "doc1_0000"},
                        {"chunk_id": "doc2_0000"}
                    ]
                }
            ]
        }
        report = validate_citations(brief, retrieved)
        assert report["valid"] is True
        assert len(report["hallucinated"]) == 0
        assert report["safe_citations"] == 2
        assert report["compliance_override"] is None

    def test_validate_citations_with_hallucination(self):
        """When a chunk_id is not in the retrieved set, flag as hallucinated."""
        retrieved = {"doc1_0000"}
        brief = {
            "recommendations": [
                {
                    "citations": [
                        {"chunk_id": "doc1_0000"},
                        {"chunk_id": "fake_doc_0000"}  # Hallucinated
                    ]
                }
            ]
        }
        report = validate_citations(brief, retrieved)
        assert report["valid"] is False
        assert len(report["hallucinated"]) == 1
        assert report["hallucinated"][0]["chunk_id"] == "fake_doc_0000"
        assert report["compliance_override"] == "needs_review"

    def test_validate_citations_doc_id_fallback(self):
        """If chunk_id is missing, fallback to doc_id prefix match."""
        retrieved = {"policy_doc_0005"}
        brief = {
            "recommendations": [
                {
                    "citations": [
                        {"doc_id": "policy_doc"}  # No chunk_id, but doc_id matches prefix
                    ]
                }
            ]
        }
        report = validate_citations(brief, retrieved)
        assert report["valid"] is True
        assert report["safe_citations"] == 1

    def test_validate_and_patch_escalates_status(self):
        """validate_and_patch_brief should mutate the dict if hallucinations found."""
        msg = _make_tool_message(json.dumps({"results": [{"chunk_id": "real_000"}]}))
        brief = {
            "compliance_status": "cleared",
            "compliance_notes": "All good.",
            "recommendations": [
                {"citations": [{"chunk_id": "fake_123"}]}
            ]
        }

        patched = validate_and_patch_brief(brief, [msg])

        # Status should be escalated
        assert patched["compliance_status"] == "needs_review"
        assert "hallucinated" in patched["compliance_notes"].lower()
        assert "_citation_validation" in patched
        assert patched["_citation_validation"]["valid"] is False
