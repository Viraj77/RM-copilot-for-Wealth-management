# src/ingestion/__init__.py
from src.ingestion.loader import load_documents, load_file, load_market_data, load_client_data
from src.ingestion.chunker import chunk_document, chunk_documents
from src.ingestion.embedder import embed_texts, embed_chunks, embed_query
from src.ingestion.indexer import ChromaIndexer, FAISSIndexer, run_ingestion

__all__ = [
    "load_documents", "load_file", "load_market_data", "load_client_data",
    "chunk_document", "chunk_documents",
    "embed_texts", "embed_chunks", "embed_query",
    "ChromaIndexer", "FAISSIndexer", "run_ingestion",
]
