"""
Centralized configuration using pydantic-settings.
All values loaded from environment variables / .env file.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_temperature: float = 0.1
    suitability_temperature: float = 0.0

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "wealth-manager-copilot"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "wealth_management_knowledge"

    # ── FAISS ─────────────────────────────────────────────────────────────────
    faiss_index_path: str = "./data/faiss_index"

    # ── Agent Controls ────────────────────────────────────────────────────────
    agent_max_steps: int = 10
    agent_max_total_steps: int = 30

    # ── Data Paths ────────────────────────────────────────────────────────────
    client_data_path: str = "./data/clients/clients.json"
    market_data_path: str = "./data/raw/market_data.json"
    raw_docs_dir: str = "./data/raw"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 200
    embedding_batch_size: int = 100

    # ── Streamlit ─────────────────────────────────────────────────────────────
    streamlit_port: int = 8501

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def chroma_persist_path(self) -> Path:
        return PROJECT_ROOT / self.chroma_persist_dir

    @property
    def raw_docs_path(self) -> Path:
        return PROJECT_ROOT / self.raw_docs_dir

    @property
    def client_data_file(self) -> Path:
        return PROJECT_ROOT / self.client_data_path

    @property
    def market_data_file(self) -> Path:
        return PROJECT_ROOT / self.market_data_path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Convenience alias
settings = get_settings()
