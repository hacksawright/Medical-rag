import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App Settings
    ENV: str = "production"
    LOG_LEVEL: str = "INFO"
    
    # Language Detection Fallback
    DEFAULT_LANGUAGE: str = "vi"

    # Translation Service Integration
    ENVIT5_API_URL: str = "https://subplot-strep-ragweed.ngrok-free.dev/predict"
    # Vector Database Settings
    CHROMA_DB_DIR: str = str(Path(__file__).resolve().parent.parent / "data" / "chromadb")
    CHROMA_COLLECTION_NAME: str = "medical_knowledge_vi"
    
    # Embedding Model Settings
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3" 
    EMBEDDING_DEVICE: str = "cuda" if os.environ.get("USE_CUDA") == "true" else "cpu"
    
    # LLM Orchestration Configuration
    LLM_PROVIDER: Literal["gemini", "qwen"] = "qwen"
    
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    
    QWEN_API_KEY: str = ""
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL_NAME: str = "qwen-max"

settings = Settings()