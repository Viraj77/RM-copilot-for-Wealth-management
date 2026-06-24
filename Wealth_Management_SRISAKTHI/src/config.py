"""Application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CLIENTS_DIR = DATA_DIR / "clients"
CHROMA_DIR = DATA_DIR / "chroma_db"
GOLDEN_SET_DIR = DATA_DIR / "golden_set"

# Load .env before anything reads environment variables
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Propagate SSL_VERIFY to environment for requests/tiktoken
if os.environ.get("SSL_VERIFY", "").lower() in ("0", "false", "no", "off"):
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
elif not os.environ.get("SSL_CERT_FILE"):
    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "rm-copilot"
    max_agent_steps: int = 10
    llm_temperature: float = 0.2
    chroma_persist_dir: str = str(CHROMA_DIR)
    chunk_size: int = 800
    chunk_overlap: int = 100
    collection_name: str = "rm_knowledge"

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def strip_api_key_quotes(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip().strip('"').strip("'")
        return v


def get_openai_api_key() -> str:
    """Resolve API key from settings or environment."""
    key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    return key.strip().strip('"').strip("'")


settings = Settings()
