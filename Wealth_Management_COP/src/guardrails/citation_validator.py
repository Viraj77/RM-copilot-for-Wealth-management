"""
Citation Hallucination Validator — Phase 8: RAG Enhancement.

Verifies that every citation in the agent's final ClientBrief actually
corresponds to a chunk that was retrieved during the conversation.

This closes the hallucination loop:
  1. Retrieval happens → chunk_ids are stored in ToolMessages.
  2. LLM synthesizes a brief and may cite arbitrary chunk_ids.
  3. This validator cross-checks step 2 against step 1.

If hallucinated citations are found, the brief's compliance_status is
escalated to 'needs_review' so the RM is alerted before sharing it.
"""

from __future__ import annotations

import json
import re
from typing import Any


# ── Chunk ID extraction from agent messages ───────────────────────────────────

def extract_retrieved_chunk_ids(messages: list) -> set[str]:
    """
    Parse the agent's message history to collect every chunk_id that was
    actually returned by the RAG retriever tool.

    Scans all ToolMessages (LangChain) whose content looks like the JSON
    response from `rag_retriever_tool`, and extracts every `chunk_id` field.

    Args:
        messages: The `AgentState["messages"]` list from the LangGraph state.

    Returns:
        Set of chunk_id strings that were genuinely retrieved.
    """
    from langchain_core.messages import ToolMessage

    retrieved_ids: set[str] = set()

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        raw_content = getattr(msg, "content", "") or ""

        # Try to parse as JSON (rag_retriever_tool returns JSON)
        try:
            data = json.loads(raw_content)
        except (json.JSONDecodeError, TypeError):
            # Not JSON — scan for chunk_id patterns with regex as fallback
            for match in re.finditer(r'"chunk_id"\s*:\s*"([^"]+)"', raw_content):
                retrieved_ids.add(match.group(1))
            continue

        # Walk the results list
        results = data.get("results", [])
        for result in results:
            chunk_id = result.get("chunk_id")
            if chunk_id:
                retrieved_ids.add(chunk_id)

            # Also check nested citation dict
            citation = result.get("citation", {})
            if isinstance(citation, dict):
                cid = citation.get("chunk_id")
                if cid:
                    retrieved_ids.add(cid)

    return retrieved_ids


# ── Citation validation ───────────────────────────────────────────────────────

def validate_citations(
    brief_dict: dict[str, Any],
    retrieved_chunk_ids: set[str],
) -> dict[str, Any]:
    """
    Cross-check every citation in the brief against the retrieved chunk set.

    Walks `brief_dict["recommendations"][*]["citations"][*]` and flags any
    citation whose `chunk_id` is NOT in `retrieved_chunk_ids`.

    Args:
        brief_dict: The raw dict produced by `synthesize_brief` (not a Pydantic model).
        retrieved_chunk_ids: Set of chunk_ids from `extract_retrieved_chunk_ids()`.

    Returns:
        Dict with:
          - ``valid`` (bool): True if no hallucinations found.
          - ``hallucinated`` (list[dict]): Each flagged citation with context.
          - ``safe_citations`` (int): Number of citations that passed.
          - ``total_citations`` (int): Total citations checked.
          - ``compliance_override`` (str | None): ``"needs_review"`` if hallucinations
            found, else None.
    """
    hallucinated: list[dict[str, Any]] = []
    safe_count = 0
    total_count = 0

    recommendations = brief_dict.get("recommendations") or []

    for rec_idx, rec in enumerate(recommendations):
        if not isinstance(rec, dict):
            continue

        citations = rec.get("citations") or []
        for cit_idx, citation in enumerate(citations):
            if not isinstance(citation, dict):
                continue

            total_count += 1
            chunk_id = citation.get("chunk_id", "")
            doc_id = citation.get("doc_id", "")

            # A citation is valid if its chunk_id appears in the retrieved set.
            # If chunk_id is empty but doc_id is present, we do a prefix check
            # (doc_id is a prefix of chunk_id: e.g. "policy_doc" matches "policy_doc_0001").
            is_valid = False
            if chunk_id and chunk_id in retrieved_chunk_ids:
                is_valid = True
            elif not chunk_id and doc_id:
                # Fallback: check if any retrieved chunk belongs to this doc
                is_valid = any(cid.startswith(doc_id) for cid in retrieved_chunk_ids)

            if is_valid:
                safe_count += 1
            else:
                hallucinated.append({
                    "recommendation_index": rec_idx,
                    "citation_index": cit_idx,
                    "chunk_id": chunk_id or "(empty)",
                    "doc_id": doc_id or "(empty)",
                    "source": citation.get("source", "(unknown)"),
                    "reason": (
                        "chunk_id not found in retrieved set"
                        if chunk_id
                        else "empty chunk_id and doc_id not matched to any retrieved chunk"
                    ),
                })

    valid = len(hallucinated) == 0
    compliance_override = None if valid else "needs_review"

    return {
        "valid": valid,
        "hallucinated": hallucinated,
        "safe_citations": safe_count,
        "total_citations": total_count,
        "compliance_override": compliance_override,
    }


# ── Convenience: validate and patch brief_dict in place ──────────────────────

def validate_and_patch_brief(
    brief_dict: dict[str, Any],
    messages: list,
) -> dict[str, Any]:
    """
    One-shot helper: extracts chunk_ids from messages, validates citations,
    and patches `brief_dict` if hallucinations are detected.

    Modifies brief_dict in-place:
      - Sets ``compliance_status`` to ``"needs_review"`` if hallucinations found.
      - Appends a warning to ``compliance_notes``.
      - Adds a ``_citation_validation`` key with the full report.

    Args:
        brief_dict: The raw dict from `synthesize_brief` (modified in-place).
        messages: The full agent message history.

    Returns:
        The (possibly patched) brief_dict.
    """
    if not brief_dict:
        return brief_dict

    retrieved_ids = extract_retrieved_chunk_ids(messages)
    report = validate_citations(brief_dict, retrieved_ids)

    # Attach the validation report for downstream inspection
    brief_dict["_citation_validation"] = report

    if not report["valid"]:
        hallucination_count = len(report["hallucinated"])

        # Escalate compliance status
        current_status = brief_dict.get("compliance_status", "cleared")
        if current_status == "cleared":
            brief_dict["compliance_status"] = "needs_review"

        # Append note
        existing_notes = brief_dict.get("compliance_notes", "") or ""
        warning = (
            f"⚠️ Citation Validation: {hallucination_count} hallucinated citation(s) detected. "
            f"The cited chunk_id(s) were not found in the retrieved document set. "
            f"Please verify these citations before sharing this brief with the client."
        )
        brief_dict["compliance_notes"] = (
            f"{existing_notes}\n\n{warning}".strip()
        )

        print(
            f"  [CitationValidator] ⚠️ {hallucination_count} hallucinated citation(s) found. "
            f"compliance_status overridden to 'needs_review'."
        )
    else:
        if report["total_citations"] > 0:
            print(
                f"  [CitationValidator] ✅ All {report['safe_citations']} citation(s) verified "
                f"against retrieved set."
            )

    return brief_dict
