"""
Structure-Aware Chunker — Phase 2: Ingestion Pipeline.

Splits RawDocuments into DocumentChunks using RecursiveCharacterTextSplitter
with structure-aware separators. Attaches parent metadata to every chunk.
"""

import hashlib
import re
from typing import Optional

from src.models.documents import DocType, DocumentChunk, RawDocument, Sensitivity

# ── Default chunking parameters ───────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 800          # Characters (approx ~200 tokens)
DEFAULT_CHUNK_OVERLAP = 200       # Characters overlap between chunks

# Separator hierarchy — respects markdown / document structure
STRUCTURE_SEPARATORS = [
    "\n## ",    # H2 heading boundary
    "\n### ",   # H3 heading boundary
    "\n#### ",  # H4 heading boundary
    "\n\n",     # Paragraph boundary
    "\n",       # Line boundary
    ". ",       # Sentence boundary
    " ",        # Word boundary
    "",         # Character fallback
]


# ── Section heading extractor ─────────────────────────────────────────────────

def _extract_section_heading(text: str) -> Optional[str]:
    """
    Extract the first markdown heading from a text chunk.
    Used to annotate chunk.page_or_section for citation purposes.
    """
    match = re.search(r"^#{1,4}\s+(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


# ── Core chunker ──────────────────────────────────────────────────────────────

def _split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text using a recursive separator strategy.

    Uses the built-in pure-Python splitter by default for maximum compatibility.
    Set USE_LANGCHAIN_SPLITTER=1 env var to use LangChain's splitter instead.
    """
    import os
    if os.getenv("USE_LANGCHAIN_SPLITTER", "0") == "1":
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
            splitter = RecursiveCharacterTextSplitter(
                separators=STRUCTURE_SEPARATORS,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            return splitter.split_text(text)
        except Exception:
            pass  # Fall through to built-in splitter

    return _simple_split(text, chunk_size, chunk_overlap)


def _simple_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Fallback chunker: splits on paragraph boundaries first,
    then falls back to character-level fixed-size chunks.
    """
    # Try paragraph-level split first
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) <= chunk_size:
                current = para
            else:
                # Para itself is too long — split by characters with overlap
                for i in range(0, len(para), chunk_size - chunk_overlap):
                    chunks.append(para[i : i + chunk_size])
                current = ""

    if current:
        chunks.append(current)

    return chunks if chunks else [text]


# ── Prefix injector ───────────────────────────────────────────────────────────

def _inject_context_prefix(chunk_text: str, doc: RawDocument) -> str:
    """
    Prepend a lightweight context header to each chunk so that
    retrieved chunks are self-contained (useful for RAG without parent context).

    Format: "[{doc_type} | {source}] {chunk_text}"
    """
    header = f"[{doc.doc_type.value.upper()} | {doc.source}]\n"
    # Don't add header if the chunk already starts with a markdown heading
    if chunk_text.lstrip().startswith("#"):
        return chunk_text
    return header + chunk_text


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document(
    doc: RawDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    inject_prefix: bool = True,
) -> list[DocumentChunk]:
    """
    Split a RawDocument into DocumentChunks.

    Args:
        doc: The source RawDocument.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        inject_prefix: If True, prepend doc_type/source context header.

    Returns:
        List of DocumentChunk objects with inherited metadata.
    """
    raw_chunks = _split_text(doc.content, chunk_size, chunk_overlap)

    document_chunks: list[DocumentChunk] = []
    for idx, raw_text in enumerate(raw_chunks):
        text = raw_text.strip()
        if not text:
            continue

        # Optionally inject context prefix
        content = _inject_context_prefix(text, doc) if inject_prefix else text

        # Extract section heading for citation annotation
        section = _extract_section_heading(text)

        chunk = DocumentChunk(
            doc_id=doc.doc_id,
            chunk_id=f"{doc.doc_id}_{idx:04d}",
            doc_type=doc.doc_type,
            source=doc.source,
            content=content,
            pub_date=doc.pub_date,
            sensitivity=doc.sensitivity,
            chunk_index=idx,
            page_or_section=section,
            metadata={
                "doc_id": doc.doc_id,
                "doc_type": doc.doc_type.value,
                "sensitivity": doc.sensitivity.value,
                "source": doc.source,
            },
        )
        document_chunks.append(chunk)

    return document_chunks


def chunk_documents(
    docs: list[RawDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    inject_prefix: bool = True,
) -> list[DocumentChunk]:
    """
    Chunk a list of RawDocuments into DocumentChunks.

    Args:
        docs: List of RawDocument objects.
        chunk_size: Max chunk size in characters.
        chunk_overlap: Overlap between chunks.
        inject_prefix: Whether to inject context prefix.

    Returns:
        Flat list of all DocumentChunks across all documents.
    """
    all_chunks: list[DocumentChunk] = []
    for doc in docs:
        chunks = chunk_document(doc, chunk_size, chunk_overlap, inject_prefix)
        all_chunks.extend(chunks)
        print(f"  Chunked '{doc.source}' -> {len(chunks)} chunks")

    print(f"\n  Total: {len(all_chunks)} chunks from {len(docs)} documents.")
    return all_chunks
