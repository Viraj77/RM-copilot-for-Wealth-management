"""
Cross-Encoder Reranker — Phase 8: RAG Enhancement.

Reranks a set of retrieved chunks using a cross-encoder model that reads
(query, passage) pairs jointly — producing more accurate relevance scores
than cosine similarity alone.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Small (22M params), fast on CPU, open-source, no API key needed.
  - Achieves strong reranking quality on passage retrieval benchmarks.

The CrossEncoder is loaded lazily on first use and cached as a module-level
singleton. If `sentence_transformers` is not installed, all reranking calls
gracefully degrade to returning the original ordering unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.models.documents import RetrievedChunk

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Module-level singleton ────────────────────────────────────────────────────

_reranker_instance: Optional["CrossEncoderReranker"] = None


def get_reranker() -> "CrossEncoderReranker":
    """Return (or lazily create) the module-level CrossEncoderReranker singleton."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance


# ── Reranker class ────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder for passage reranking.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, chunks, top_n=5)

    If `sentence_transformers` is not installed, `rerank()` returns the
    original chunks unchanged (graceful degradation — no exception raised).
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._available: Optional[bool] = None  # Cached availability check

    def _is_available(self) -> bool:
        """Check whether sentence_transformers is importable (cached)."""
        if self._available is None:
            try:
                import sentence_transformers  # noqa: F401  # type: ignore
                self._available = True
            except ImportError:
                self._available = False
                print(
                    "  [Reranker] sentence_transformers not installed — "
                    "reranking disabled. Install with: pip install sentence-transformers"
                )
        return self._available

    def _get_model(self):
        """Lazily load the CrossEncoder model."""
        if self._model is None:
            if not self._is_available():
                return None
            from sentence_transformers import CrossEncoder  # type: ignore
            print(f"  [Reranker] Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            print(f"  [Reranker] Cross-encoder loaded.")
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Rerank a list of retrieved chunks using the cross-encoder.

        Reads each (query, chunk.content) pair jointly and produces a
        relevance score. Returns the top_n chunks sorted by cross-encoder score.

        If sentence_transformers is not installed or the model fails to load,
        returns the original chunks (first top_n) unchanged.

        Args:
            query: The original search query.
            chunks: Retrieved chunks to rerank (typically top-10 from embedding search).
            top_n: Number of chunks to return after reranking.

        Returns:
            List of up to top_n RetrievedChunk objects, sorted by cross-encoder score
            (highest first).
        """
        if not chunks:
            return chunks

        model = self._get_model()
        if model is None:
            # Graceful degradation: return original ordering
            return chunks[:top_n]

        try:
            # Build (query, passage) pairs
            pairs = [(query, r.chunk.content) for r in chunks]

            # Score all pairs — returns numpy array of floats
            scores = model.predict(pairs)

            # Attach cross-encoder scores and sort
            scored = sorted(
                zip(scores, chunks),
                key=lambda x: float(x[0]),
                reverse=True,
            )

            return [chunk for _, chunk in scored[:top_n]]

        except Exception as exc:
            print(f"  [Reranker] Reranking failed ({exc!r}); returning original order.")
            return chunks[:top_n]

    @property
    def is_available(self) -> bool:
        """True if sentence_transformers is installed and the model can be loaded."""
        return self._is_available()
