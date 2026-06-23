"""
Entitlement Filter — Phase 5: Guardrails.

Ensures that the final output does not explicitly leak restricted document names
or IDs if the RM shouldn't have access. (Defense in depth, as the retrieval
tool already filters this, but this catches LLM hallucinations.)

Fix 5: The list of restricted document IDs is now loaded dynamically from
ChromaDB at module startup, so new restricted docs are automatically covered
without requiring a code change. Falls back to a hardcoded list if ChromaDB
is unavailable.
"""

from src.models.documents import ENTITLEMENT_ACCESS, Sensitivity

# ── Hardcoded fallback (used when ChromaDB is unavailable) ───────────────────

_HARDCODED_RESTRICTED_FRAGMENTS = [
    "cryptocurrency_digital_assets",
    "emerging_markets_deep_dive",
    "[restricted]",
]


def _load_restricted_doc_ids() -> list[str]:
    """
    Dynamically load restricted document identifiers from ChromaDB.

    Queries the collection for all chunks with sensitivity == 'restricted'
    and returns their unique doc_ids as lowercase substrings to match against.

    Falls back to ``_HARDCODED_RESTRICTED_FRAGMENTS`` if ChromaDB is
    unavailable (e.g. during testing or before ingestion).

    Returns:
        List of lowercase restricted doc_id strings.
    """
    try:
        from config.settings import settings
        from src.ingestion.indexer import ChromaIndexer

        indexer = ChromaIndexer(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
        collection = indexer._get_collection()

        # Retrieve all restricted-sensitivity chunks (no embedding needed)
        results = collection.get(
            where={"sensitivity": {"$eq": Sensitivity.RESTRICTED.value}},
            include=["metadatas"],
        )

        doc_ids: set[str] = set()
        for meta in (results.get("metadatas") or []):
            doc_id = meta.get("doc_id", "")
            if doc_id:
                doc_ids.add(doc_id.lower())

        if doc_ids:
            print(
                f"  [EntitlementFilter] Loaded {len(doc_ids)} restricted doc IDs from ChromaDB."
            )
            return list(doc_ids)

    except Exception as exc:
        print(
            f"  [EntitlementFilter] Could not load restricted docs from ChromaDB "
            f"({exc!r}). Using hardcoded fallback list."
        )

    return list(_HARDCODED_RESTRICTED_FRAGMENTS)


# Load once at module import time; cached for the session lifetime.
_RESTRICTED_FRAGMENTS: list[str] = _load_restricted_doc_ids()


def verify_entitlements(text: str, rm_tier: str) -> bool:
    """
    Returns True if the text is safe, False if it appears to leak restricted info.

    Dynamically checks against the list of restricted document IDs loaded from
    ChromaDB (or the hardcoded fallback list). Institutional-tier RMs always pass.

    Args:
        text: The generated response text to scan.
        rm_tier: The RM's entitlement tier ('standard', 'premium', 'institutional').

    Returns:
        True if the output is safe to show; False if a restricted leak is detected.
    """
    # Allowed level for the RM
    allowed_level = ENTITLEMENT_ACCESS.get(rm_tier.lower(), 1)

    # Institutional (level 2) has access to everything — always safe
    if allowed_level >= 2:
        return True

    lower_text = text.lower()

    # Check for any restricted doc ID fragment in the output
    for fragment in _RESTRICTED_FRAGMENTS:
        if fragment in lower_text:
            return False

    return True


def reload_restricted_doc_ids() -> None:
    """
    Force-reload the restricted document ID list from ChromaDB.

    Call this after new documents have been ingested during the same session.
    """
    global _RESTRICTED_FRAGMENTS
    _RESTRICTED_FRAGMENTS = _load_restricted_doc_ids()
