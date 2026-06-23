"""
OpenAI Embedding Wrapper — Phase 2: Ingestion Pipeline.

Wraps OpenAI's text-embedding-3-small model with batching,
retry logic, and a simple caching interface.
"""

import time
from typing import Optional

from config.settings import settings
from src.models.documents import DocumentChunk

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds


# ── Embedding client ──────────────────────────────────────────────────────────

def _get_openai_client():
    """Lazily initialise OpenAI client (avoids import error if key not set)."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise ImportError(
            "openai package is required. Install with: pip install openai"
        )
    api_key = settings.openai_api_key
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please configure it in your .env file."
        )
    return OpenAI(api_key=api_key)


# ── Core embedding function ───────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """
    Embed a list of texts using OpenAI embedding model.

    Args:
        texts: List of text strings to embed.
        model: OpenAI embedding model name.
        batch_size: Number of texts per API call.

    Returns:
        List of embedding vectors (one per input text).
    """
    if not texts:
        return []

    client = _get_openai_client()
    embeddings: list[list[float]] = []

    # Process in batches
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]

        for attempt in range(MAX_RETRIES):
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

                progress = min(batch_start + batch_size, len(texts))
                print(f"  Embedded {progress}/{len(texts)} texts", end="\r")
                break

            except Exception as exc:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"\n  [WARN] Embedding error (attempt {attempt+1}): {exc}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Failed to embed batch starting at index {batch_start} "
                        f"after {MAX_RETRIES} attempts: {exc}"
                    ) from exc

    print(f"\n  Embedding complete: {len(embeddings)} vectors generated.")
    return embeddings


def embed_chunks(
    chunks: list[DocumentChunk],
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[DocumentChunk], list[list[float]]]:
    """
    Embed a list of DocumentChunks.

    Args:
        chunks: List of DocumentChunk objects.
        model: OpenAI embedding model.
        batch_size: API batch size.

    Returns:
        Tuple of (chunks, embeddings) — same order, same length.
    """
    texts = [chunk.content for chunk in chunks]
    embeddings = embed_texts(texts, model=model, batch_size=batch_size)
    return chunks, embeddings


def embed_query(
    query: str,
    model: str = DEFAULT_MODEL,
) -> list[float]:
    """
    Embed a single query string for similarity search.

    Args:
        query: Search query text.
        model: OpenAI embedding model.

    Returns:
        Single embedding vector.
    """
    results = embed_texts([query], model=model, batch_size=1)
    return results[0]
