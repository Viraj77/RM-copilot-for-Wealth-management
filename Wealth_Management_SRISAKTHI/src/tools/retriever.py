"""Hybrid RAG retriever: BM25 keyword + vector semantic with metadata filters."""

import re
from datetime import datetime

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents
from src.ingestion.indexer import build_vector_store, get_vector_store
from src.models import DocumentType, RetrievedChunk, Sensitivity

DOC_CODE_PATTERN = re.compile(r"\b(PG|CMP|RN)-\d{3}\b", re.IGNORECASE)
RRF_K = 60


class HybridRAGRetriever:
    """BM25 + vector retrieval with entitlement, type, and freshness filtering."""

    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        self._bm25: BM25Retriever | None = None
        self._chunk_docs: list[Document] = []

    def _ensure_store(self):
        if self.vector_store is None:
            try:
                self.vector_store = get_vector_store()
                if self.vector_store._collection.count() == 0:
                    self.vector_store = build_vector_store()
            except Exception as exc:
                raise RuntimeError(
                    "Vector store unavailable. Set OPENAI_API_KEY and run "
                    "`python scripts/ingest.py` to build the knowledge index."
                ) from exc

    def _ensure_bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            self._chunk_docs = chunk_documents(load_documents())
            self._bm25 = BM25Retriever.from_documents(self._chunk_docs)
            self._bm25.k = 15
        return self._bm25

    def _passes_filters(
        self,
        metadata: dict,
        entitlements: list[Sensitivity],
        doc_types: list[DocumentType] | None,
        min_date: str | None,
    ) -> bool:
        sensitivity = Sensitivity(metadata.get("sensitivity", "public"))
        if sensitivity not in entitlements:
            return False
        doc_type_str = metadata.get("type", "research")
        if doc_types and DocumentType(doc_type_str) not in doc_types:
            return False
        if min_date:
            doc_date = metadata.get("date", "1900-01-01")
            if doc_date < min_date:
                return False
        return True

    def _doc_key(self, metadata: dict, content: str) -> str:
        return f"{metadata.get('doc_id', '')}:{content[:80]}"

    def _vector_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        self._ensure_store()
        return self.vector_store.similarity_search_with_relevance_scores(query, k=k * 3)

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        bm25 = self._ensure_bm25()
        bm25.k = k * 3
        return bm25.invoke(query)

    def _rrf_fuse(
        self,
        vector_results: list[tuple[Document, float]],
        bm25_docs: list[Document],
        entitlements: list[Sensitivity],
        doc_types: list[DocumentType] | None,
        min_date: str | None,
        k: int,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion of vector + BM25 results."""
        fused_scores: dict[str, float] = {}
        doc_map: dict[str, tuple[Document, float]] = {}

        for rank, (doc, score) in enumerate(vector_results):
            if not self._passes_filters(doc.metadata, entitlements, doc_types, min_date):
                continue
            key = self._doc_key(doc.metadata, doc.page_content)
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (RRF_K + rank + 1)
            doc_map[key] = (doc, score)

        for rank, doc in enumerate(bm25_docs):
            if not self._passes_filters(doc.metadata, entitlements, doc_types, min_date):
                continue
            key = self._doc_key(doc.metadata, doc.page_content)
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (RRF_K + rank + 1)
            if key not in doc_map:
                doc_map[key] = (doc, 0.5)

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        chunks: list[RetrievedChunk] = []
        for key, rrf_score in ranked:
            doc, vec_score = doc_map[key]
            chunks.append(
                RetrievedChunk(
                    doc_id=doc.metadata.get("doc_id", "unknown"),
                    content=doc.page_content,
                    doc_type=DocumentType(doc.metadata.get("type", "research")),
                    source=doc.metadata.get("source", "unknown"),
                    date=doc.metadata.get("date", "unknown"),
                    sensitivity=Sensitivity(doc.metadata.get("sensitivity", "public")),
                    score=round(rrf_score + vec_score * 0.1, 4),
                )
            )
        return chunks

    def retrieve(
        self,
        query: str,
        k: int = 5,
        doc_types: list[DocumentType] | None = None,
        rm_entitlements: list[Sensitivity] | None = None,
        min_score: float = 0.0,
        min_date: str | None = "2025-01-01",
    ) -> list[RetrievedChunk]:
        """Hybrid retrieve: BM25 + vector with metadata filtering."""
        entitlements = rm_entitlements or [Sensitivity.PUBLIC, Sensitivity.INTERNAL]
        vector_results = self._vector_search(query, k)
        bm25_docs = self._bm25_search(query, k)
        chunks = self._rrf_fuse(vector_results, bm25_docs, entitlements, doc_types, min_date, k)
        return [c for c in chunks if c.score >= min_score] or chunks

    def _extract_entities(self, query: str, chunks: list[RetrievedChunk]) -> list[str]:
        """Extract PG/CMP/RN codes and client IDs for multi-hop follow-up."""
        entities: set[str] = set()
        for match in DOC_CODE_PATTERN.finditer(query):
            entities.add(match.group(0).upper())
        for chunk in chunks:
            for match in DOC_CODE_PATTERN.finditer(chunk.content[:500]):
                entities.add(match.group(0).upper())
            code = chunk.doc_id.split("_")[0] if "_" in chunk.doc_id else chunk.doc_id
            if DOC_CODE_PATTERN.match(code):
                entities.add(code.upper())
        client_match = re.search(r"C-\d+", query, re.IGNORECASE)
        if client_match:
            entities.add(client_match.group(0).upper())
        return list(entities)[:8]

    def multi_hop_retrieve(
        self,
        query: str,
        k: int = 5,
        rm_entitlements: list[Sensitivity] | None = None,
    ) -> list[RetrievedChunk]:
        """Multi-hop: initial hybrid search + entity-driven follow-up."""
        initial = self.retrieve(query, k=k, rm_entitlements=rm_entitlements)
        if not initial:
            return initial

        entities = self._extract_entities(query, initial)
        follow_up_query = f"{query} {' '.join(entities)}"
        follow_up = self.retrieve(follow_up_query, k=k, rm_entitlements=rm_entitlements)

        seen = {c.doc_id + c.content[:50] for c in initial}
        combined = list(initial)
        for chunk in follow_up:
            key = chunk.doc_id + chunk.content[:50]
            if key not in seen:
                combined.append(chunk)
                seen.add(key)

        combined.sort(key=lambda c: c.score, reverse=True)
        return combined[: k * 2]

    def format_chunks_for_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks as context for the LLM."""
        if not chunks:
            return "No relevant documents retrieved."
        lines = []
        for i, chunk in enumerate(chunks, 1):
            citation = f"[{chunk.doc_id} | {chunk.source} | {chunk.date}]"
            lines.append(f"--- Document {i} {citation} ---")
            lines.append(chunk.content)
            lines.append("")
        return "\n".join(lines)


def chunks_to_citations(chunks: list[RetrievedChunk]) -> list[str]:
    """Extract citation strings from chunks."""
    return [
        f"{c.doc_id} ({c.source}, {c.date}, type={c.doc_type.value})"
        for c in chunks
    ]
