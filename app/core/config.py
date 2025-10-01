"""
Configuration management for cloud-agnostic AI-powered CV matching platform.

This module provides centralized configuration management supporting:
- Environment variables
- Cloud-agnostic OpenAI integration (direct or Azure)
- Local development defaults
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)

# Cloud-agnostic configuration - no Azure SDK dependencies required


class Settings(BaseSettings):
    """
    Main application settings with smart defaults for local development
    and full Azure support when needed.
    """
    
    # Environment Detection
    ENVIRONMENT: str = Field(
        default="local",
        description="Environment (local/development/production)"
    )
    DEBUG: bool = Field(
        default=False,
        description="Debug mode"
    )
    
    # Core Application Settings
    APP_NAME: str = Field(
        default="CV Matching Platform",
        description="Application name"
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    # Security (Required)
    SECRET_KEY: str = Field(
        ...,
        description="JWT signing secret key (min 32 chars)"
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT algorithm"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=15,
        description="Access token expiry in minutes"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        description="Refresh token expiry in days"
    )
    
    # Password Security
    BCRYPT_ROUNDS: int = Field(
        default=12,
        description="BCrypt hashing rounds"
    )
    PASSWORD_MIN_LENGTH: int = Field(
        default=8,
        description="Minimum password length"
    )
    PASSWORD_RESET_TOKEN_TTL_MINUTES: int = Field(
        default=30,
        description="Minutes password reset tokens remain valid"
    )
    PASSWORD_RESET_TOKEN_BYTES: int = Field(
        default=32,
        description="Random bytes used when generating password reset tokens"
    )
    PASSWORD_RESET_REQUEST_INTERVAL_SECONDS: int = Field(
        default=60,
        description="Cooldown between password reset requests for the same user"
    )
    
    # Security Settings
    MAX_LOGIN_ATTEMPTS: int = Field(
        default=5,
        description="Maximum login attempts before lockout"
    )
    LOGIN_ATTEMPT_WINDOW_MINUTES: int = Field(
        default=15,
        description="Login attempt window in minutes"
    )
    SESSION_LIMIT_PER_USER: int = Field(
        default=5,
        description="Maximum concurrent sessions per user"
    )
    API_KEY_LENGTH: int = Field(
        default=32,
        description="API key length in bytes"
    )
    
    # Server Configuration
    HOST: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    PORT: int = Field(
        default=8000,
        description="Server port"
    )
    WORKERS: int = Field(
        default=1,
        description="Number of workers"
    )
    
    # CORS Configuration
    CORS_ORIGINS: str = Field(
        default="*",
        description="Comma-separated CORS origins"
    )
    
    # Logging
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    LOG_FORMAT: str = Field(
        default="console",
        description="Log format (json/console)"
    )
    
    # Cache Configuration (Redis or Memory)
    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Redis URL (optional, uses memory cache if not set)"
    )
    CACHE_TTL: int = Field(
        default=3600,
        description="Default cache TTL in seconds"
    )
    
    # OpenAI Configuration (Cloud-Agnostic)
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key (direct or Azure)"
    )
    OPENAI_BASE_URL: Optional[str] = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL (default: OpenAI, can be Azure endpoint)"
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4",
        description="OpenAI chat model name"
    )
    OPENAI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model name"
    )
    OPENAI_API_VERSION: Optional[str] = Field(
        default=None,
        description="API version (for Azure OpenAI compatibility)"
    )
    OPENAI_MAX_TOKENS: int = Field(
        default=4000,
        description="Maximum tokens for chat completions"
    )
    OPENAI_TEMPERATURE: float = Field(
        default=0.7,
        description="Default temperature for chat completions"
    )
    
    # Azure OpenAI Configuration (Legacy - for backward compatibility)
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(
        default=None,
        description="Azure OpenAI endpoint (legacy)"
    )
    AZURE_OPENAI_KEY: Optional[str] = Field(
        default=None,
        description="Azure OpenAI API key (legacy)"
    )
    AZURE_OPENAI_DEPLOYMENT: str = Field(
        default="gpt-4",
        description="OpenAI deployment name (legacy)"
    )
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = Field(
        default="text-embedding-ada-002",
        description="Embedding model deployment (legacy)"
    )
    
    
    # PostgreSQL Configuration (Local Development)
    POSTGRES_URL: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection URL"
    )
    POSTGRES_HOST: str = Field(
        default="localhost",
        description="PostgreSQL host"
    )
    POSTGRES_PORT: int = Field(
        default=5432,
        description="PostgreSQL port"
    )
    POSTGRES_USER: str = Field(
        default="cv_user",
        description="PostgreSQL user"
    )
    POSTGRES_PASSWORD: str = Field(
        default="cv_password",
        description="PostgreSQL password"
    )
    POSTGRES_DB: str = Field(
        default="cv-analytic",
        description="PostgreSQL database name"
    )
    
    
    
    
    
    
    # File Storage Configuration
    LOCAL_STORAGE_PATH: str = Field(
        default="./storage",
        description="Local file storage path"
    )
    MAX_FILE_SIZE: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum file size in bytes"
    )
    
    # AI Performance and Processing Settings
    REQUEST_TIMEOUT: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    MAX_REQUEST_SIZE: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum request size in bytes"
    )
    
    # Document Processing Configuration
    MAX_DOCUMENT_SIZE: int = Field(
        default=25 * 1024 * 1024,  # 25MB
        description="Maximum document size for processing"
    )
    DOCUMENT_CHUNK_SIZE: int = Field(
        default=1000,
        description="Document chunk size for embeddings"
    )
    DOCUMENT_CHUNK_OVERLAP: int = Field(
        default=200,
        description="Document chunk overlap size"
    )
    
    # Semantic Search Configuration
    SEARCH_SIMILARITY_THRESHOLD: float = Field(
        default=0.7,
        description="Minimum similarity score for search results"
    )
    SEARCH_MAX_RESULTS: int = Field(
        default=20,
        description="Maximum search results to return"
    )
    SEARCH_RERANK_ENABLED: bool = Field(
        default=True,
        description="Enable AI-powered result reranking"
    )
    
    # Embedding Configuration
    EMBEDDING_DIMENSION: int = Field(
        default=3072,
        description="Embedding vector dimension (text-embedding-3-large)"
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=100,
        description="Batch size for embedding generation"
    )
    
    # Semantic Caching Configuration
    SEMANTIC_CACHE_ENABLED: bool = Field(
        default=True,
        description="Enable semantic similarity caching"
    )
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = Field(
        default=0.95,
        description="Similarity threshold for cache hits"
    )
    SEMANTIC_CACHE_TTL: int = Field(
        default=7200,  # 2 hours
        description="Semantic cache TTL in seconds"
    )
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    RATE_LIMIT_REQUESTS: int = Field(
        default=100,
        description="Rate limit requests per minute"
    )
    
    # Health Check
    HEALTH_CHECK_INTERVAL: int = Field(
        default=30,
        description="Health check interval in seconds"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator('SECRET_KEY')
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate that secret key is secure enough."""
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v
    
    @field_validator('ENVIRONMENT')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate and normalize environment name."""
        v = v.lower()
        if v not in ('local', 'development', 'staging', 'production'):
            logger.warning(f"Unknown environment: {v}, defaulting to 'local'")
            return 'local'
        return v
    
    def is_legacy_azure_configured(self) -> bool:
        """Check if legacy Azure OpenAI configuration is used."""
        return bool(self.AZURE_OPENAI_ENDPOINT and self.AZURE_OPENAI_KEY)
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == 'production'
    
    def is_local(self) -> bool:
        """Check if running in local environment."""
        return self.ENVIRONMENT == 'local'
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins as a list."""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    def get_postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        if self.POSTGRES_URL:
            return self.POSTGRES_URL
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    def is_postgres_configured(self) -> bool:
        """Check if PostgreSQL is configured."""
        return bool(self.POSTGRES_URL or (self.POSTGRES_HOST and self.POSTGRES_USER and self.POSTGRES_DB))
    
    def is_openai_configured(self) -> bool:
        """Check if OpenAI is configured (direct or Azure)."""
        return bool(self.OPENAI_API_KEY or self.AZURE_OPENAI_KEY)
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration with fallback to Azure settings."""
        # Prefer direct OpenAI configuration
        if self.OPENAI_API_KEY:
            config = {
                "api_key": self.OPENAI_API_KEY,
                "base_url": self.OPENAI_BASE_URL,
                "model": self.OPENAI_MODEL,
                "embedding_model": self.OPENAI_EMBEDDING_MODEL,
                "max_tokens": self.OPENAI_MAX_TOKENS,
                "temperature": self.OPENAI_TEMPERATURE,
                "provider": "openai"
            }
            # Add API version if specified (for Azure compatibility)
            if self.OPENAI_API_VERSION:
                config["api_version"] = self.OPENAI_API_VERSION
            return config
        
        # Fallback to Azure configuration
        if self.AZURE_OPENAI_KEY and self.AZURE_OPENAI_ENDPOINT:
            return {
                "api_key": self.AZURE_OPENAI_KEY,
                "base_url": self.AZURE_OPENAI_ENDPOINT,
                "model": self.AZURE_OPENAI_DEPLOYMENT,
                "embedding_model": self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                "max_tokens": self.OPENAI_MAX_TOKENS,
                "temperature": self.OPENAI_TEMPERATURE,
                "api_version": "2024-02-01",
                "provider": "azure"
            }
        
        raise ValueError("No OpenAI configuration found. Set OPENAI_API_KEY or Azure credentials.")
    
    def is_document_processing_enabled(self) -> bool:
        """Check if document processing is enabled."""
        return self.is_openai_configured()
    
    def is_semantic_search_enabled(self) -> bool:
        """Check if semantic search is enabled."""
        return self.is_openai_configured()
    
    def is_semantic_caching_enabled(self) -> bool:
        """Check if semantic caching is enabled."""
        return self.SEMANTIC_CACHE_ENABLED and bool(self.REDIS_URL)






@lru_cache()
def get_settings() -> Settings:
    """
    Get cloud-agnostic application settings.
    
    This function:
    1. Loads settings from environment variables and .env file
    2. Returns a validated Settings instance
    3. Supports both direct OpenAI and Azure OpenAI configurations
    """
    
    # Load base settings from environment
    settings = Settings()
    
    # Log environment detection
    logger.info(
        "Settings loaded",
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG,
        legacy_azure_configured=settings.is_legacy_azure_configured()
    )
    
    # Validate that we have minimum required configuration
    if not settings.SECRET_KEY:
        raise ValueError("SECRET_KEY must be configured in environment or .env file")
    
    # Log final configuration state
    logger.info(
        "Configuration ready",
        environment=settings.ENVIRONMENT,
        redis_configured=bool(settings.REDIS_URL),
        openai_configured=settings.is_openai_configured(),
        semantic_search_enabled=settings.is_semantic_search_enabled(),
        semantic_caching_enabled=settings.is_semantic_caching_enabled(),
        document_processing_enabled=settings.is_document_processing_enabled(),
        legacy_azure_openai_configured=bool(settings.AZURE_OPENAI_ENDPOINT)
    )
    
    return settings

