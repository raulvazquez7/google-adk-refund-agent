"""
Centralized configuration management using Pydantic Settings.

This module provides type-safe, validated configuration for the entire application.
All environment variables are loaded and validated here.
"""
from typing import Literal
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration with validation.

    All settings are loaded from environment variables (.env file).
    Pydantic validates types and provides defaults.
    """

    # Google Cloud Platform
    gcp_project_id: str = Field(..., description="GCP Project ID")
    gcp_location: str = Field(default="us-central1", description="GCP region")

    # Firestore
    firestore_database_id: str = Field(
        default="orders",
        description="Firestore database name"
    )

    # AI Models
    embeddings_model: str = Field(
        default="text-embedding-004",
        description="Vertex AI embedding model"
    )
    agent_model: str = Field(
        default="gemini-2.5-flash",
        description="LLM model for agents"
    )

    # Agent Configuration
    llm_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout for LLM calls in seconds"
    )
    llm_max_retries: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Max retry attempts for LLM calls"
    )
    llm_rate_limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max concurrent LLM calls (prevents API saturation)"
    )
    embeddings_rate_limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max concurrent embedding API calls (faster than LLM, higher limit)"
    )
    firestore_rate_limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Max concurrent Firestore operations (least restrictive)"
    )

    # RAG Configuration
    rag_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of chunks to retrieve in RAG"
    )
    embeddings_cache_size: int = Field(
        default=100,
        ge=0,
        le=1000,
        description="Max embeddings to cache (LRU eviction). Set to 0 to disable cache."
    )

    # Langfuse Observability
    langfuse_public_key: str = Field(..., description="Langfuse public key")
    langfuse_secret_key: str = Field(..., description="Langfuse secret key")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse host URL"
    )

    # Application
    app_name: str = Field(
        default="barefoot_refund_agent",
        description="Application name for tracing"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from .env that aren't defined here
    )


# Global settings instance (loaded once)
settings = Settings()
