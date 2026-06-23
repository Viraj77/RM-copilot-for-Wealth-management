"""
Configuration management for Wealth Manager Copilot
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Vector Store
    vector_store_type: str = os.getenv("VECTOR_STORE_TYPE", "faiss")
    vector_store_dir: str = os.getenv("VECTOR_STORE_DIR", "./data/vector_store")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 100))
    
    # Knowledge
    knowledge_dir: str = os.getenv("KNOWLEDGE_DIR", "./data/sample_knowledge")
    
    # Agent
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", 20))
    agent_temperature: float = float(os.getenv("AGENT_TEMPERATURE", 0.2))
    agent_model: str = os.getenv("AGENT_MODEL", "gpt-4o")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "./logs/wealth_manager.log")
    
    # Streamlit
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", 8501))
    streamlit_host: str = os.getenv("STREAMLIT_HOST", "localhost")
    
    # LangSmith (optional)
    langsmith_api_key: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "wealth-manager-copilot")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def validate_config(self) -> bool:
        """Validate critical configuration."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        # Create necessary directories
        Path(self.vector_store_dir).mkdir(parents=True, exist_ok=True)
        Path(self.knowledge_dir).mkdir(parents=True, exist_ok=True)
        Path(Path(self.log_file).parent).mkdir(parents=True, exist_ok=True)
        
        return True


# Global settings instance
settings = Settings()
