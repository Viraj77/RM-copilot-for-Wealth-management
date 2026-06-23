"""
Document and vector store metadata models — DocumentChunk, DocType, Sensitivity.
Used throughout the ingestion pipeline and RAG retriever.
"""

from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class DocType(str, Enum):
    """Categorises knowledge documents by functional type."""
    PRODUCT = "product"       # Product guides, fund factsheets
    POLICY = "policy"         # Suitability rules, compliance frameworks
    RESEARCH = "research"     # Market research, economic outlook notes


class Sensitivity(str, Enum):
    """
    Data sensitivity level.
    Controls which RM entitlement tiers can access a document.

    Access rules:
        standard    → public only
        premium     → public + internal
        institutional → public + internal + restricted
    """
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


# ── Sensitivity access map (also used in entitlement_filter.py) ──────────────

SENSITIVITY_LEVELS: dict[str, int] = {
    Sensitivity.PUBLIC: 0,
    Sensitivity.INTERNAL: 1,
    Sensitivity.RESTRICTED: 2,
}

ENTITLEMENT_ACCESS: dict[str, int] = {
    "standard": 1,       # public + internal
    "premium": 1,        # public + internal
    "institutional": 2,  # all
}


# ── Raw Document ──────────────────────────────────────────────────────────────

class RawDocument(BaseModel):
    """Represents a loaded source document before chunking."""
    doc_id: str = Field(..., description="Unique document identifier")
    doc_type: DocType
    source: str = Field(..., description="Original filename or URL")
    content: str = Field(..., description="Full document text")
    pub_date: Optional[date] = Field(None, description="Document publish / effective date")
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    extra_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata from the source"
    )


# ── Document Chunk ────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """
    A single chunk produced by the chunking pipeline.
    Carries parent document metadata for filtering and citation.
    """
    doc_id: str = Field(..., description="Parent document identifier")
    chunk_id: str = Field(..., description="Unique chunk ID: '{doc_id}_{chunk_index}'")
    doc_type: DocType
    source: str = Field(..., description="Original filename / document title")
    content: str = Field(..., description="Chunk text content")
    pub_date: Optional[date] = None
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    chunk_index: int = Field(..., ge=0, description="Position within the parent document")
    page_or_section: Optional[str] = Field(
        None, description="Page number or section heading of the chunk"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata passed to ChromaDB"
    )

    def to_chroma_metadata(self) -> dict[str, Any]:
        """Serialize to ChromaDB-compatible metadata dict (str values only)."""
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "doc_type": self.doc_type.value,
            "source": self.source,
            "date": self.pub_date.isoformat() if self.pub_date else "",
            "sensitivity": self.sensitivity.value,
            "chunk_index": str(self.chunk_index),
            "page_or_section": self.page_or_section or "",
            **{k: str(v) for k, v in self.metadata.items()},
        }


# ── Retrieval Result ──────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """
    A chunk returned from a vector store similarity search,
    augmented with a relevance score.
    """
    chunk: DocumentChunk
    score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")

    @property
    def as_citation_dict(self) -> dict[str, str]:
        """Format for inclusion in a Recommendation.citations list."""
        return {
            "doc_id": self.chunk.doc_id,
            "chunk_id": self.chunk.chunk_id,
            "doc_type": self.chunk.doc_type.value,
            "source": self.chunk.source,
            "chunk_text": self.chunk.content[:500],
            "page_or_section": self.chunk.page_or_section or "",
        }
