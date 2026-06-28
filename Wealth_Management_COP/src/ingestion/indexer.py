"""
ChromaDB + FAISS Indexer — Phase 2: Ingestion Pipeline.

Indexes DocumentChunks into:
  - ChromaDB (primary store, persistent, with metadata filtering)
  - FAISS (secondary store, for evaluation speed comparison)

Also exposes query interfaces used by the RAG retriever tool.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from src.models.documents import (
    ENTITLEMENT_ACCESS,
    SENSITIVITY_LEVELS,
    DocumentChunk,
    RetrievedChunk,
    Sensitivity,
)


# ── ChromaDB Indexer ──────────────────────────────────────────────────────────

class ChromaIndexer:
    """
    Manages a persistent ChromaDB collection for the wealth management
    knowledge base. Supports metadata-filtered similarity search.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name
        self._client = None
        self._collection = None

    def _get_client(self):
        """Lazily create ChromaDB persistent client."""
        if self._client is None:
            try:
                import chromadb  # type: ignore
                from chromadb.config import Settings as ChromaSettings  # type: ignore
            except ImportError:
                raise ImportError(
                    "chromadb is required. Install with: pip install chromadb"
                )
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _get_collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def index_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> None:
        """
        Add DocumentChunks and their embeddings to ChromaDB.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: Corresponding embedding vectors (same order).
            batch_size: Number of chunks per upsert call.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match."
            )

        collection = self._get_collection()
        total = len(chunks)

        for start in range(0, total, batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]

            ids = [c.chunk_id for c in batch_chunks]
            documents = [c.content for c in batch_chunks]
            metadatas = [c.to_chroma_metadata() for c in batch_chunks]

            collection.upsert(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            print(
                f"  Indexed {min(start + batch_size, total)}/{total} chunks into ChromaDB",
                end="\r",
            )

        print(f"\n  ChromaDB indexing complete: {total} chunks in '{self.collection_name}'")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_types: Optional[list[str]] = None,
        sensitivity_max: str = "internal",
        date_after: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Similarity search with optional metadata filters.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            doc_types: Filter to specific doc types (e.g. ["policy", "product"]).
            sensitivity_max: Maximum sensitivity level allowed (entitlement filter).
            date_after: Only return chunks with date after this value (ISO format).

        Returns:
            List of RetrievedChunk sorted by relevance (highest first).
        """
        collection = self._get_collection()

        # Build ChromaDB where clause
        where_clauses: list[dict] = []

        # Sensitivity filter — only return chunks ≤ allowed sensitivity level
        allowed_level = ENTITLEMENT_ACCESS.get(sensitivity_max, 1)
        allowed_sensitivities = [
            s.value
            for s, lvl in SENSITIVITY_LEVELS.items()
            if isinstance(s, Sensitivity) and lvl <= allowed_level
        ]
        if allowed_sensitivities:
            where_clauses.append({"sensitivity": {"$in": allowed_sensitivities}})

        # Doc type filter
        if doc_types:
            where_clauses.append({"doc_type": {"$in": doc_types}})

        # Date filter
        if date_after:
            where_clauses.append({"date": {"$gte": date_after}})

        # Compose where clause
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}
        else:
            where = None

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = collection.query(**query_kwargs)

        # Parse results into RetrievedChunk objects
        retrieved: list[RetrievedChunk] = []
        if not results["ids"] or not results["ids"][0]:
            return retrieved

        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            content = results["documents"][0][i]
            distance = results["distances"][0][i]
            # ChromaDB cosine distance → cosine similarity
            score = max(0.0, 1.0 - distance)

            # Reconstruct DocumentChunk from metadata
            chunk = _meta_to_chunk(chunk_id, content, meta)
            retrieved.append(RetrievedChunk(chunk=chunk, score=round(score, 4)))

        return sorted(retrieved, key=lambda r: r.score, reverse=True)

    def count(self) -> int:
        """Return total number of documents in the collection."""
        return self._get_collection().count()

    def reset(self) -> None:
        """Delete and recreate the collection (use with caution)."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            print(f"  Deleted collection '{self.collection_name}'")
        except Exception:
            pass
        self._collection = None
        self._get_collection()
        print(f"  Recreated collection '{self.collection_name}'")

    # ── BM25 Keyword Search ───────────────────────────────────────────────────

    def query_bm25(
        self,
        query_text: str,
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[tuple[str, float]]:
        """
        Keyword-based BM25 search over the ChromaDB collection.

        Loads all documents matching the metadata filter from ChromaDB,
        builds an in-memory BM25Okapi index, and returns ranked
        (chunk_id, normalised_bm25_score) pairs.

        Falls back to an empty list if ``rank_bm25`` is not installed.

        Args:
            query_text: Raw search query string.
            top_k: Number of results to return.
            where: Optional ChromaDB ``where`` clause for metadata pre-filtering.

        Returns:
            List of (chunk_id, normalised_score) tuples, highest first.
            Scores normalised to [0, 1] relative to the top BM25 score.
        """
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError:
            print(
                "  [BM25] rank_bm25 not installed — BM25 keyword search disabled. "
                "Install with: pip install rank-bm25"
            )
            return []

        collection = self._get_collection()

        get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if where:
            get_kwargs["where"] = where

        try:
            results = collection.get(**get_kwargs)
        except Exception as exc:
            print(f"  [BM25] collection.get() failed: {exc!r}")
            return []

        ids = results.get("ids") or []
        documents = results.get("documents") or []

        if not ids or not documents:
            return []

        tokenised_corpus = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenised_corpus)
        query_tokens = query_text.lower().split()
        raw_scores = bm25.get_scores(query_tokens)

        id_score_pairs = sorted(
            zip(ids, raw_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        if not id_score_pairs:
            return []

        max_score = id_score_pairs[0][1]
        if max_score <= 0:
            return [(cid, 0.0) for cid, _ in id_score_pairs[:top_k]]

        return [
            (cid, round(score / max_score, 4))
            for cid, score in id_score_pairs[:top_k]
        ]

    def query_hybrid(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 5,
        doc_types: Optional[list[str]] = None,
        sensitivity_max: str = "internal",
        date_after: Optional[str] = None,
        rrf_k: int = 60,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> list[RetrievedChunk]:
        """
        Hybrid search: fuse semantic (cosine) + BM25 keyword rankings via
        Reciprocal Rank Fusion (RRF).

        Strategy:
          1. Semantic search with top 2*top_k candidates.
          2. BM25 keyword search with top 2*top_k candidates.
          3. Fuse rankings with weighted RRF: score = sum(weight / (rrf_k + rank)).
          4. Return top_k chunks by fused RRF score.

        Falls back to pure semantic search if rank_bm25 is not installed.

        Args:
            query_embedding: Dense query vector for semantic search.
            query_text: Raw query string for BM25.
            top_k: Final number of results to return.
            doc_types: Optional doc type filter.
            sensitivity_max: Entitlement tier filter.
            date_after: Optional ISO date lower bound.
            rrf_k: RRF smoothing constant (default 60).
            semantic_weight: Weight for semantic RRF contributions (default 0.6).
            bm25_weight: Weight for BM25 RRF contributions (default 0.4).

        Returns:
            List of up to top_k RetrievedChunk objects sorted by fused RRF score.
        """
        candidate_k = max(top_k * 2, 10)

        # Step 1: Semantic retrieval
        semantic_results = self.query(
            query_embedding=query_embedding,
            top_k=candidate_k,
            doc_types=doc_types,
            sensitivity_max=sensitivity_max,
            date_after=date_after,
        )

        # Step 2: Build metadata where clause for BM25 (mirrors query() logic)
        where_clauses: list[dict] = []
        allowed_level = ENTITLEMENT_ACCESS.get(sensitivity_max, 1)
        allowed_sensitivities = [
            s.value
            for s, lvl in SENSITIVITY_LEVELS.items()
            if isinstance(s, Sensitivity) and lvl <= allowed_level
        ]
        if allowed_sensitivities:
            where_clauses.append({"sensitivity": {"$in": allowed_sensitivities}})
        if doc_types:
            where_clauses.append({"doc_type": {"$in": doc_types}})
        if date_after:
            where_clauses.append({"date": {"$gte": date_after}})

        bm25_where: Optional[dict] = None
        if len(where_clauses) == 1:
            bm25_where = where_clauses[0]
        elif len(where_clauses) > 1:
            bm25_where = {"$and": where_clauses}

        # Step 3: BM25 retrieval
        bm25_results = self.query_bm25(query_text, top_k=candidate_k, where=bm25_where)

        # No BM25 available → fall back to pure semantic
        if not bm25_results:
            return semantic_results[:top_k]

        # Step 4: Reciprocal Rank Fusion
        semantic_lookup: dict[str, RetrievedChunk] = {
            r.chunk.chunk_id: r for r in semantic_results
        }
        fused_scores: dict[str, float] = {}

        for rank, r in enumerate(semantic_results, start=1):
            cid = r.chunk.chunk_id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + semantic_weight / (rrf_k + rank)

        for rank, (cid, _) in enumerate(bm25_results, start=1):
            fused_scores[cid] = fused_scores.get(cid, 0.0) + bm25_weight / (rrf_k + rank)

        ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

        # Step 5: Build final result list (only chunks with full data from semantic pass)
        final_results: list[RetrievedChunk] = []
        for cid in ranked_ids[:top_k]:
            if cid in semantic_lookup:
                r = semantic_lookup[cid]
                final_results.append(
                    RetrievedChunk(
                        chunk=r.chunk,
                        score=round(fused_scores[cid], 6),
                    )
                )

        return final_results


# ── FAISS Indexer ─────────────────────────────────────────────────────────────

class FAISSIndexer:
    """
    FAISS flat index for evaluation / speed comparison.
    Stores embeddings + chunk metadata in memory and optionally on disk.
    """

    def __init__(self, index_path: Optional[str] = None) -> None:
        self.index_path = index_path or settings.faiss_index_path
        self._index = None
        self._chunks: list[DocumentChunk] = []

    def _get_faiss(self):
        try:
            import faiss  # type: ignore
            return faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            )

    def index_chunks(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        """Build FAISS flat L2 index from chunks and embeddings."""
        import numpy as np  # type: ignore

        faiss = self._get_faiss()
        dim = len(embeddings[0])
        self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalise)

        # Normalize for cosine similarity
        vecs = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vecs)
        self._index.add(vecs)
        self._chunks = list(chunks)

        print(f"  FAISS index built: {self._index.ntotal} vectors (dim={dim})")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Search FAISS index, returns top_k results."""
        if self._index is None or not self._chunks:
            raise RuntimeError("FAISS index is empty. Run index_chunks() first.")

        import numpy as np  # type: ignore
        faiss = self._get_faiss()

        vec = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(vec)

        scores, indices = self._index.search(vec, top_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    score=round(float(score), 4),
                )
            )
        return results

    def save(self) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        import faiss  # type: ignore

        path = Path(self.index_path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "index.faiss"))

        # Save chunk metadata as JSON
        meta = [c.to_chroma_metadata() for c in self._chunks]
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"  FAISS index saved to {path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _meta_to_chunk(chunk_id: str, content: str, meta: dict) -> DocumentChunk:
    """Reconstruct a DocumentChunk from ChromaDB metadata dict."""
    from src.models.documents import DocType, Sensitivity

    pub_date = None
    if meta.get("date"):
        try:
            from datetime import date
            parts = meta["date"].split("-")
            pub_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
        except Exception:
            pass

    return DocumentChunk(
        doc_id=meta.get("doc_id", ""),
        chunk_id=chunk_id,
        doc_type=DocType(meta.get("doc_type", "research")),
        source=meta.get("source", ""),
        content=content,
        pub_date=pub_date,
        sensitivity=Sensitivity(meta.get("sensitivity", "internal")),
        chunk_index=int(meta.get("chunk_index", 0)),
        page_or_section=meta.get("page_or_section") or None,
        metadata={k: v for k, v in meta.items()},
    )


# ── Single-file ingestion runner ──────────────────────────────────────────────

def ingest_single_file(filepath: Path, doc_type: str) -> None:
    """
    Ingest a single document into the existing ChromaDB collection.
    
    Args:
        filepath: Path to the document.
        doc_type: Type of document (e.g., 'product', 'policy', 'research').
    """
    from src.models.documents import DocType
    from src.ingestion.loader import load_file
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.embedder import embed_chunks

    print("\n" + "=" * 60)
    print(f"  Ingesting single file: {filepath.name}")
    print("=" * 60)
    
    # 1. Load
    doc = load_file(filepath, DocType(doc_type))
    
    # 2. Chunk
    chunks = chunk_documents(
        [doc],
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    
    if not chunks:
        print("  No chunks generated. Skipping.")
        return

    # 3. Embed
    chunks, embeddings = embed_chunks(chunks)

    # 4. Index
    chroma = ChromaIndexer()
    chroma.index_chunks(chunks, embeddings)
    
    print(f"  ✅ Single file ingestion complete! {len(chunks)} chunks indexed.")


# ── End-to-end ingestion runner ───────────────────────────────────────────────

def run_ingestion(
    raw_docs_dir: Optional[str] = None,
    reset: bool = False,
    build_faiss: bool = True,
) -> tuple[ChromaIndexer, Optional[FAISSIndexer]]:
    """
    Full ingestion pipeline: Load → Chunk → Embed → Index.

    Args:
        raw_docs_dir: Path to raw documents directory.
        reset: If True, wipe and recreate the ChromaDB collection first.
        build_faiss: If True, also build a FAISS index.

    Returns:
        (ChromaIndexer, FAISSIndexer | None)
    """
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.embedder import embed_chunks
    from src.ingestion.loader import load_documents

    raw_dir = raw_docs_dir or settings.raw_docs_dir

    print("\n" + "=" * 60)
    print("  STEP 1: Loading documents")
    print("=" * 60)
    docs = load_documents(raw_dir)

    print("\n" + "=" * 60)
    print("  STEP 2: Chunking documents")
    print("=" * 60)
    chunks = chunk_documents(
        docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    print("\n" + "=" * 60)
    print("  STEP 3: Embedding chunks")
    print("=" * 60)
    chunks, embeddings = embed_chunks(chunks)

    print("\n" + "=" * 60)
    print("  STEP 4: Indexing into ChromaDB")
    print("=" * 60)
    chroma = ChromaIndexer()
    if reset:
        chroma.reset()
    chroma.index_chunks(chunks, embeddings)

    faiss_indexer: Optional[FAISSIndexer] = None
    if build_faiss and chunks:
        print("\n" + "=" * 60)
        print("  STEP 5: Building FAISS index (evaluation)")
        print("=" * 60)
        faiss_indexer = FAISSIndexer()
        faiss_indexer.index_chunks(chunks, embeddings)
        faiss_indexer.save()

    print("\n" + "=" * 60)
    print(f"  [OK] Ingestion complete! {len(chunks)} chunks indexed.")
    print("=" * 60 + "\n")

    return chroma, faiss_indexer


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the knowledge ingestion pipeline")
    parser.add_argument(
        "--reset", action="store_true", help="Wipe and recreate the ChromaDB collection"
    )
    parser.add_argument(
        "--no-faiss", action="store_true", help="Skip building the FAISS index"
    )
    parser.add_argument(
        "--raw-dir", type=str, default=None, help="Override raw documents directory"
    )
    args = parser.parse_args()

    run_ingestion(
        raw_docs_dir=args.raw_dir,
        reset=args.reset,
        build_faiss=not args.no_faiss,
    )
