"""Embed and index documents in ChromaDB."""

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import get_openai_api_key, settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents
from src.utils.http_client import get_async_http_client, get_sync_http_client


def get_embeddings() -> OpenAIEmbeddings:
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to the project root .env file:\n"
            "  OPENAI_API_KEY=your-key-here"
        )
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=api_key,
        check_embedding_ctx_length=False,
        http_client=get_sync_http_client(),
        http_async_client=get_async_http_client(),
    )


def build_vector_store(
    documents: list[Document] | None = None,
    persist: bool = True,
    reset: bool = True,
) -> Chroma:
    """Build or rebuild the ChromaDB vector store."""
    docs = documents or load_documents()
    chunks = chunk_documents(docs)
    embeddings = get_embeddings()

    persist_dir = settings.chroma_persist_dir if persist else None

    if reset and persist_dir:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=persist_dir)
            client.delete_collection(settings.collection_name)
        except Exception:
            pass

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=persist_dir,
    )
    return vector_store


def get_vector_store() -> Chroma:
    """Load existing vector store or build from scratch."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
