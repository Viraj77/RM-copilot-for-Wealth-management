"""
RAG Retriever Integration Tests — Fix 8.

Tests the rag_retrieve() function against an in-memory ChromaDB instance
(using chromadb.EphemeralClient) with synthetic chunks and mock embeddings.

No OpenAI API key is required — embeddings are replaced with deterministic
random unit vectors.

Run: pytest tests/test_rag_retriever.py -v
"""

import math
import random
import json
import pytest

from src.models.documents import DocType, DocumentChunk, Sensitivity


# ── Helpers ───────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 8  # Small dimension for fast tests


def _unit_vector(seed: int, dim: int = EMBEDDING_DIM) -> list[float]:
    """Return a deterministic pseudo-random unit vector."""
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def _make_chunk(
    doc_id: str,
    chunk_index: int,
    content: str,
    doc_type: DocType,
    sensitivity: Sensitivity,
) -> DocumentChunk:
    return DocumentChunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}_{chunk_index:04d}",
        doc_type=doc_type,
        source=f"{doc_id}.md",
        content=content,
        sensitivity=sensitivity,
        chunk_index=chunk_index,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ephemeral_chroma():
    """
    Create an ephemeral (in-memory) ChromaDB collection populated with
    a small set of known chunks across all sensitivity tiers and doc types.
    """
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed — skipping RAG retriever tests")

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="test_collection",
        metadata={"hnsw:space": "cosine"},
    )

    # Define test chunks
    chunks_data = [
        _make_chunk("equity_product_guide", 0,
                    "Equity funds are suitable for balanced and growth profiles.",
                    DocType.PRODUCT, Sensitivity.PUBLIC),
        _make_chunk("conservative_policy", 0,
                    "Conservative clients must not hold high yield bonds.",
                    DocType.POLICY, Sensitivity.INTERNAL),
        _make_chunk("conservative_policy", 1,
                    "Maximum equity allocation for conservative clients is 25%.",
                    DocType.POLICY, Sensitivity.INTERNAL),
        _make_chunk("q2_equity_outlook", 0,
                    "Q2 2026 equity markets show moderate growth.",
                    DocType.RESEARCH, Sensitivity.PUBLIC),
        _make_chunk("emerging_markets_deep_dive", 0,
                    "Emerging markets carry elevated political and currency risk.",
                    DocType.RESEARCH, Sensitivity.RESTRICTED),
        _make_chunk("crypto_assets_note", 0,
                    "Cryptocurrency is unsuitable for all but aggressive clients.",
                    DocType.RESEARCH, Sensitivity.RESTRICTED),
    ]

    ids = [c.chunk_id for c in chunks_data]
    embeddings = [_unit_vector(i) for i in range(len(chunks_data))]
    documents = [c.content for c in chunks_data]
    metadatas = [c.to_chroma_metadata() for c in chunks_data]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    yield collection
    # cleanup happens automatically with EphemeralClient


@pytest.fixture
def patched_indexer(ephemeral_chroma, monkeypatch):
    """
    Monkeypatches ChromaIndexer._get_collection() to return the ephemeral
    test collection, and embed_query to return a deterministic vector.
    """
    from src.ingestion.indexer import ChromaIndexer
    from src.tools import rag_retriever_tool as rrt_module

    # Reset the module-level singleton so it picks up the patched collection
    monkeypatch.setattr(rrt_module, "_chroma_indexer", None)

    def _fake_get_collection(self):
        return ephemeral_chroma

    monkeypatch.setattr(ChromaIndexer, "_get_collection", _fake_get_collection)

    # Also patch embed_query to return a deterministic vector (no API needed)
    monkeypatch.setattr(
        "src.tools.rag_retriever_tool.embed_query",
        lambda query: _unit_vector(42, EMBEDDING_DIM),
    )

    # Prevent Windows segfaults by stopping CrossEncoderReranker from importing sentence_transformers
    from src.ingestion.reranker import CrossEncoderReranker
    monkeypatch.setattr(CrossEncoderReranker, "_is_available", lambda self: False)

    # Reset thread-local cache so our patch takes effect
    from src.tools import rag_retriever_tool
    if hasattr(rag_retriever_tool._thread_local, "chroma_indexer"):
        del rag_retriever_tool._thread_local.chroma_indexer

    yield


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRAGRetrieverIntegration:
    """Integration tests for rag_retrieve() against an ephemeral ChromaDB."""

    def test_empty_query_returns_error(self, patched_indexer):
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(query="   ")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_invalid_doc_type_returns_error(self, patched_indexer):
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(query="some query", doc_types=["invalid_type"])
        assert result["success"] is False
        assert "Invalid doc_types" in result["error"]

    def test_basic_query_returns_results(self, patched_indexer):
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(query="equity allocation policy", top_k=5)
        assert result["success"] is True
        assert result["total_found"] >= 1
        assert len(result["results"]) >= 1

    def test_result_has_required_fields(self, patched_indexer):
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(query="equity fund", top_k=3)
        assert result["success"] is True
        for chunk in result["results"]:
            for field in ["chunk_id", "doc_id", "doc_type", "source", "content",
                          "relevance_score", "citation"]:
                assert field in chunk, f"Missing field '{field}' in result chunk"

    def test_doc_type_filter_policy_only(self, patched_indexer):
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(query="client suitability rules", doc_types=["policy"], top_k=5)
        assert result["success"] is True
        for chunk in result["results"]:
            assert chunk["doc_type"] == "policy", (
                f"Expected doc_type 'policy', got '{chunk['doc_type']}'"
            )

    def test_doc_type_filter_product_only(self, patched_indexer):
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(query="equity fund product", doc_types=["product"], top_k=5)
        assert result["success"] is True
        for chunk in result["results"]:
            assert chunk["doc_type"] == "product"

    def test_sensitivity_standard_excludes_restricted(self, patched_indexer):
        """Standard tier (sensitivity_max='standard') should return PUBLIC + INTERNAL (per user design)."""
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(query="policy rules", sensitivity_max="standard", top_k=10)
        assert result["success"] is True
        for chunk in result["results"]:
            assert chunk["sensitivity"] in ("public", "internal"), (
                f"Standard RM received non-public/internal chunk: sensitivity='{chunk['sensitivity']}'"
            )

    def test_sensitivity_premium_excludes_restricted(self, patched_indexer):
        """Premium tier should return PUBLIC + INTERNAL but not RESTRICTED."""
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(query="policy rules", sensitivity_max="premium", top_k=10)
        assert result["success"] is True
        for chunk in result["results"]:
            assert chunk["sensitivity"] in ("public", "internal"), (
                f"Premium RM received restricted chunk: sensitivity='{chunk['sensitivity']}'"
            )

    def test_sensitivity_institutional_can_see_restricted(self, patched_indexer):
        """Institutional tier should be able to retrieve RESTRICTED chunks."""
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(
            query="emerging markets cryptocurrency",
            sensitivity_max="institutional",
            top_k=10,
        )
        assert result["success"] is True
        sensitivities = {c["sensitivity"] for c in result["results"]}
        assert "restricted" in sensitivities, (
            "Institutional RM should be able to see restricted chunks"
        )

    def test_top_k_cap_respected(self, patched_indexer):
        """Results should not exceed the requested top_k."""
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(query="portfolio allocation", top_k=2, sensitivity_max="institutional")
        assert result["success"] is True
        assert len(result["results"]) <= 2

    def test_filters_applied_in_response(self, patched_indexer):
        """The response should echo back the filters that were applied."""
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(
            query="equity",
            doc_types=["product"],
            sensitivity_max="premium",
            top_k=3,
        )
        assert result["success"] is True
        filters = result["filters_applied"]
        assert filters["doc_types"] == ["product"]
        assert filters["sensitivity_max"] == "premium"
        assert filters["top_k"] == 3

    def test_multi_hop_retrieval_deduplication(self, patched_indexer):
        """Multi-hop retrieval should deduplicate results across queries."""
        from src.tools.rag_retriever_tool import rag_retrieve_multi_hop
        result = rag_retrieve_multi_hop(
            queries=["equity policy", "equity rules"],  # Similar queries → likely overlapping results
            sensitivity_max="institutional",
            top_k_per_query=5,
        )
        assert result["success"] is True
        chunk_ids = [c["chunk_id"] for c in result["results"]]
        assert len(chunk_ids) == len(set(chunk_ids)), "Multi-hop results should be deduplicated"

    def test_multi_hop_empty_queries(self, patched_indexer):
        from src.tools.rag_retriever_tool import rag_retrieve_multi_hop
        result = rag_retrieve_multi_hop(queries=[])
        assert result["success"] is False
        assert "No queries" in result["error"]

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 8 — Feature 1: Hybrid BM25 Search Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHybridBM25Search:
    """Tests for BM25 keyword search and RRF hybrid fusion."""

    def test_hybrid_retrieve_returns_results(self, patched_indexer):
        """use_hybrid=True should return valid results (with or without rank_bm25)."""
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(
            query="equity allocation",
            top_k=5,
            sensitivity_max="institutional",
            use_hybrid=True,
            rerank=False,
        )
        assert result["success"] is True
        assert result["total_found"] >= 1

    def test_hybrid_response_has_retrieval_method(self, patched_indexer):
        """Response dict should include a 'retrieval_method' field."""
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(
            query="conservative policy", top_k=3,
            sensitivity_max="institutional",
            use_hybrid=True, rerank=False,
        )
        assert result["success"] is True
        assert "retrieval_method" in result, "Response should contain 'retrieval_method' key"

    def test_semantic_only_returns_results(self, patched_indexer):
        """use_hybrid=False should still return valid results via pure semantic search."""
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(
            query="conservative policy", top_k=3,
            sensitivity_max="institutional",
            use_hybrid=False, rerank=False,
        )
        assert result["success"] is True
        assert result["total_found"] >= 1

    def test_query_bm25_with_rank_bm25(self, patched_indexer):
        """query_bm25() should return ranked (chunk_id, score) tuples if rank_bm25 installed."""
        pytest.importorskip("rank_bm25", reason="rank_bm25 not installed — skipping BM25 test")
        from src.ingestion.indexer import ChromaIndexer
        indexer = ChromaIndexer()
        results = indexer.query_bm25("equity allocation conservative", top_k=5)
        # Should return a list of (chunk_id, float) tuples
        assert isinstance(results, list)
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0, f"BM25 score should be normalised to [0,1], got {score}"

    def test_query_bm25_returns_empty_if_rank_bm25_missing(self, patched_indexer, monkeypatch):
        """query_bm25() should return [] gracefully if rank_bm25 not installed."""
        import builtins
        real_import = builtins.__import__

        def _block_rank_bm25(name, *args, **kwargs):
            if name == "rank_bm25":
                raise ImportError("rank_bm25 blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_rank_bm25)

        from src.ingestion.indexer import ChromaIndexer
        indexer = ChromaIndexer()
        results = indexer.query_bm25("test query", top_k=5)
        assert results == [], "Should return empty list when rank_bm25 is not available"

    def test_rrf_fusion_does_not_exceed_top_k(self, patched_indexer):
        """Hybrid search should not return more results than top_k."""
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(
            query="allocation policy research",
            top_k=2,
            sensitivity_max="institutional",
            use_hybrid=True,
            rerank=False,
        )
        assert result["success"] is True
        assert len(result["results"]) <= 2

    def test_hybrid_filters_still_respected(self, patched_indexer):
        """Sensitivity and doc_type filters should work with hybrid search."""
        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(
            query="allocation rules",
            doc_types=["policy"],
            sensitivity_max="standard",  # Only public
            top_k=5,
            use_hybrid=True,
            rerank=False,
        )
        assert result["success"] is True
        for chunk in result["results"]:
            assert chunk["doc_type"] == "policy"
            assert chunk["sensitivity"] == "public"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 8 — Feature 2: Cross-Encoder Reranking Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossEncoderReranking:
    """Tests for cross-encoder reranking integration."""

    def test_rerank_returns_top_n_with_mock_reranker(self, patched_indexer, monkeypatch):
        """With a mocked reranker, rerank=True should return at most rerank_top_n results."""
        from src.ingestion import reranker as reranker_module
        from src.models.documents import RetrievedChunk

        # Mock get_reranker to return a simple pass-through reranker
        class _MockReranker:
            is_available = True

            def rerank(self, query, chunks, top_n=5):
                # Return reversed order to verify reranker was actually called
                return list(reversed(chunks))[:top_n]

        monkeypatch.setattr(reranker_module, "_reranker_instance", _MockReranker())

        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(
            query="equity fund",
            top_k=5,
            rerank_top_n=3,
            sensitivity_max="institutional",
            use_hybrid=False,
            rerank=True,
        )
        assert result["success"] is True
        assert len(result["results"]) <= 3, (
            f"Expected ≤3 results after rerank_top_n=3, got {len(result['results'])}"
        )

    def test_rerank_false_skips_reranker(self, patched_indexer, monkeypatch):
        """rerank=False should not touch the reranker at all."""
        from src.ingestion import reranker as reranker_module

        call_count = {"n": 0}

        class _TrackingReranker:
            is_available = True

            def rerank(self, query, chunks, top_n=5):
                call_count["n"] += 1
                return chunks[:top_n]

        monkeypatch.setattr(reranker_module, "_reranker_instance", _TrackingReranker())

        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(
            query="equity fund", top_k=3,
            sensitivity_max="institutional",
            use_hybrid=False, rerank=False,
        )
        assert result["success"] is True
        assert call_count["n"] == 0, "Reranker should not be called when rerank=False"

    def test_rerank_unavailable_degrades_gracefully(self, patched_indexer, monkeypatch):
        """If reranker is not available, retrieval should still succeed."""
        from src.ingestion import reranker as reranker_module

        class _UnavailableReranker:
            is_available = False

            def rerank(self, query, chunks, top_n=5):
                return chunks[:top_n]

        monkeypatch.setattr(reranker_module, "_reranker_instance", _UnavailableReranker())

        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(
            query="equity fund", top_k=3,
            sensitivity_max="institutional",
            use_hybrid=False, rerank=True,
        )
        assert result["success"] is True, "Retrieval should succeed even when reranker unavailable"
        assert len(result["results"]) <= 3

    def test_reranker_exception_degrades_gracefully(self, patched_indexer, monkeypatch):
        """If reranker.rerank() throws, retrieval should still return results."""
        from src.ingestion import reranker as reranker_module

        class _BrokenReranker:
            is_available = True

            def rerank(self, query, chunks, top_n=5):
                raise RuntimeError("Simulated reranker failure")

        monkeypatch.setattr(reranker_module, "_reranker_instance", _BrokenReranker())

        from src.tools.rag_retriever_tool import rag_retrieve
        result = rag_retrieve(
            query="equity fund", top_k=3,
            sensitivity_max="institutional",
            use_hybrid=False, rerank=True,
        )
        assert result["success"] is True, "Retrieval should not crash when reranker throws"
