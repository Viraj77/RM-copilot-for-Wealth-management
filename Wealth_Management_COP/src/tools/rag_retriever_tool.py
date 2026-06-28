"""
RAG Retriever Tool — Phase 3 / Phase 8 Enhancement.

Hybrid retrieval over the wealth management knowledge base (ChromaDB).
Supports:
  - Pure semantic similarity search (cosine via ChromaDB)
  - Hybrid search (semantic + BM25 keyword fusion via RRF) [Phase 8]
  - Cross-encoder reranking (optional, requires sentence-transformers) [Phase 8]
  - Metadata filtering by doc_type, sensitivity (entitlement), and date
  - Multi-hop retrieval with deduplication

Returns ranked DocumentChunks with citation metadata for use in recommendations.
"""

import json
from typing import Any, Optional

from config.settings import settings
from src.ingestion.embedder import embed_query
from src.ingestion.indexer import ChromaIndexer
from src.models.documents import RetrievedChunk

import threading

# ── Thread-local ChromaDB indexer ─────────────────────────────────────────────

# Streamlit creates a new thread for each run/interaction.
# ChromaDB uses SQLite, which throws a ProgrammingError if a connection is
# accessed from a thread other than the one that created it.
# We use threading.local() to give each thread its own indexer instance.
_thread_local = threading.local()


def _get_indexer() -> ChromaIndexer:
    """Return (or create) the thread-local ChromaDB indexer."""
    if not hasattr(_thread_local, "chroma_indexer"):
        _thread_local.chroma_indexer = ChromaIndexer(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
    return _thread_local.chroma_indexer


# ── Core retrieval function ───────────────────────────────────────────────────

def rag_retrieve(
    query: str,
    doc_types: Optional[list[str]] = None,
    sensitivity_max: str = "internal",
    top_k: int = 5,
    date_after: Optional[str] = None,
    use_hybrid: bool = True,
    rerank: bool = True,
    rerank_top_n: int = 5,
) -> dict[str, Any]:
    """
    Hybrid RAG retrieval over the wealth management knowledge base.

    Embeds the query, searches ChromaDB with optional metadata filters,
    optionally fuses with BM25 keyword search (Phase 8), optionally reranks
    with a cross-encoder (Phase 8), and returns ranked chunks with citations.

    Args:
        query: Natural language search query.
        doc_types: Filter to specific doc types. Options: ["product", "policy", "research"].
                   Pass None to search all types.
        sensitivity_max: Maximum sensitivity tier allowed. Controls entitlement filtering.
                         Options: "standard" (public only), "premium" (public+internal),
                         "institutional" (all including restricted).
        top_k: Number of top results to return (default: 5). When reranking is
               enabled, the retrieval candidate pool is max(top_k*2, 10) and
               results are trimmed to rerank_top_n after reranking.
        date_after: ISO date string (YYYY-MM-DD). Only return docs after this date.
        use_hybrid: If True (default), use hybrid semantic + BM25 search via RRF.
                    Falls back to pure semantic if rank_bm25 is not installed.
        rerank: If True (default), apply cross-encoder reranking to the top candidates.
                Falls back to original ordering if sentence_transformers not installed.
        rerank_top_n: Number of results to return after reranking (default: 5).
                      Ignored when rerank=False.

    Returns:
        Dict with: success, query, results (list of chunk dicts with citations),
        total_found, filters_applied, retrieval_method.
    """
    if not query.strip():
        return {
            "success": False,
            "error": "Query cannot be empty.",
            "results": [],
        }

    # Validate doc_types
    valid_types = {"product", "policy", "research"}
    if doc_types:
        invalid = set(doc_types) - valid_types
        if invalid:
            return {
                "success": False,
                "error": f"Invalid doc_types: {invalid}. Must be subset of {valid_types}",
                "results": [],
            }

    # Validate sensitivity_max
    valid_tiers = {"standard", "premium", "institutional"}
    if sensitivity_max not in valid_tiers:
        sensitivity_max = "internal"  # Safe fallback

    try:
        # Step 1: Embed the query
        query_vector = embed_query(query)

        # Step 2: Retrieve candidates
        indexer = _get_indexer()
        candidate_k = max(top_k * 2, 10) if rerank else top_k
        retrieval_method = "semantic"

        import threading
        if not hasattr(rag_retrieve, "_lock"):
            rag_retrieve._lock = threading.Lock()
            
        with rag_retrieve._lock:
            if use_hybrid:
                retrieved: list[RetrievedChunk] = indexer.query_hybrid(
                    query_embedding=query_vector,
                    query_text=query,
                    top_k=candidate_k,
                    doc_types=doc_types,
                    sensitivity_max=sensitivity_max,
                    date_after=date_after,
                )
                # Check if hybrid actually used BM25 (it returns semantic-only if rank_bm25 missing)
                retrieval_method = "hybrid_rrf" if retrieved else "semantic"
                if not retrieved:
                    retrieved = indexer.query(
                        query_embedding=query_vector,
                        top_k=candidate_k,
                        doc_types=doc_types,
                        sensitivity_max=sensitivity_max,
                        date_after=date_after,
                    )
            else:
                retrieved = indexer.query(
                    query_embedding=query_vector,
                    top_k=candidate_k,
                    doc_types=doc_types,
                    sensitivity_max=sensitivity_max,
                    date_after=date_after,
                )

        # Step 3: Optional cross-encoder reranking
        if rerank and retrieved:
            try:
                from src.ingestion.reranker import get_reranker
                reranker = get_reranker()
                if reranker.is_available:
                    retrieved = reranker.rerank(query, retrieved, top_n=rerank_top_n)
                    retrieval_method += "+reranked"
                else:
                    # Reranker unavailable — trim to top_k
                    retrieved = retrieved[:top_k]
            except Exception as exc:
                print(f"  [RAG] Reranking failed ({exc!r}); using original ordering.")
                retrieved = retrieved[:top_k]
        else:
            retrieved = retrieved[:top_k]

        # Step 4: Format results with citation metadata
        results = []
        for r in retrieved:
            results.append({
                "doc_type": r.chunk.doc_type.value,
                "source": r.chunk.source,
                "sensitivity": r.chunk.sensitivity.value,
                "content": r.chunk.content,
                "page_or_section": r.chunk.page_or_section,
                "relevance_score": r.score,
            })

        return {
            "success": True,
            "query": query,
            "total_found": len(results),
            "retrieval_method": retrieval_method,
            "filters_applied": {
                "doc_types": doc_types or "all",
                "sensitivity_max": sensitivity_max,
                "date_after": date_after or "none",
                "top_k": top_k,
                "use_hybrid": use_hybrid,
                "rerank": rerank,
            },
            "results": results,
        }

    except Exception as exc:
        import traceback
        with open("rag_error.log", "a", encoding="utf-8") as f:
            f.write(f"RAG RETRIEVE ERROR: {str(exc)}\n")
            traceback.print_exc(file=f)
        return {
            "success": False,
            "error": f"Retrieval failed: {str(exc)}",
            "results": [],
            "note": "Ensure the knowledge base has been ingested. Run: python -m src.ingestion.indexer",
        }


def rag_retrieve_multi_hop(
    queries: list[str],
    doc_types: Optional[list[str]] = None,
    sensitivity_max: str = "internal",
    top_k_per_query: int = 3,
    use_hybrid: bool = True,
    rerank: bool = False,  # Off by default for multi-hop (speed)
) -> dict[str, Any]:
    """
    Multi-hop retrieval: run multiple queries and deduplicate results.

    Useful for complex requests requiring evidence from multiple angles
    (e.g., "suitability policy AND product details for fund X").

    Args:
        queries: List of search queries (max 5).
        doc_types: Optional doc type filter.
        sensitivity_max: Entitlement filter.
        top_k_per_query: Results per query before deduplication.
        use_hybrid: Whether to use hybrid BM25+semantic search per hop.
        rerank: Whether to apply cross-encoder reranking per hop (default False for speed).

    Returns:
        Merged, deduplicated retrieval results sorted by relevance.
    """
    if not queries:
        return {"success": False, "error": "No queries provided.", "results": []}

    queries = queries[:5]  # Cap at 5 to prevent abuse

    seen_chunk_ids: set[str] = set()
    all_results: list[dict] = []

    for query in queries:
        result = rag_retrieve(
            query=query,
            doc_types=doc_types,
            sensitivity_max=sensitivity_max,
            top_k=top_k_per_query,
            use_hybrid=use_hybrid,
            rerank=rerank,
            rerank_top_n=top_k_per_query,
        )
        if result["success"]:
            for chunk_result in result["results"]:
                if chunk_result["chunk_id"] not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_result["chunk_id"])
                    all_results.append(chunk_result)

    # Sort by relevance score descending
    all_results.sort(key=lambda r: r["relevance_score"], reverse=True)

    return {
        "success": True,
        "queries": queries,
        "total_found": len(all_results),
        "retrieval_type": "multi_hop",
        "results": all_results,
    }


# ── LangChain Tool wrapper ────────────────────────────────────────────────────

def get_rag_retriever_tool():
    """Return a LangChain-compatible RAG retriever tool."""
    try:
        from langchain_core.tools import tool  # type: ignore

        @tool
        def rag_retriever_tool(
            query: str,
            rm_tier: str = "standard",
            sensitivity_max: Optional[str] = None,
            doc_types: Optional[str] = None,
            top_k: int = 3,
            use_hybrid: bool = True,
        ) -> str:
            """
            Search the wealth management knowledge base for relevant documents.

            Use this tool to retrieve product guides, compliance policies, and research
            notes relevant to a client's situation or a recommendation you are considering.

            Args:
                query: Natural language search query (e.g. 'maximum equity allocation
                       for conservative client' or 'Q2 2026 equity market outlook').
                rm_tier: Your RM entitlement tier ('standard', 'premium', or 'institutional').
                         MUST be passed exactly as given in your system prompt.
                sensitivity_max: (Deprecated) Same as rm_tier.

                doc_types: Comma-separated doc types to filter on. Options: 'product',
                           'policy', 'research'. Leave empty to search all types.
                top_k: Number of results to return (default: 3, max: 10).
                use_hybrid: If True (default), fuse semantic and BM25 keyword search
                            for more robust retrieval (especially for exact-match queries).

            Returns:
                JSON string with ranked chunks and citation metadata.
            """
            # Handle deprecated sensitivity_max argument from cached agent schemas
            actual_tier = sensitivity_max if sensitivity_max else rm_tier

            # Parse doc types if comma-separated
            parsed_types = None
            if doc_types:
                parsed_types = [t.strip() for t in doc_types.split(",") if t.strip()]

            top_k = min(int(top_k), 10)

            result = rag_retrieve(
                query=query,
                doc_types=parsed_types,
                sensitivity_max=actual_tier,
                top_k=top_k,
                use_hybrid=use_hybrid,
                rerank=False,
            )

            if result["success"]:
                return json.dumps(result, indent=2)
            else:
                return f"Error: {result.get('error')}"

        return rag_retriever_tool

    except ImportError:
        raise ImportError(
            "langchain-core is required. Install with: pip install langchain-core"
        )
