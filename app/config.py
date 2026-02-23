"""Configuration settings for the document processing system"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # AWS Configuration
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "us-east-1"
    s3_bucket_name: str
    sqs_queue_url: str
    
    # LlamaCloud Configuration
    llama_cloud_api_key: str
    
    # OpenAI Configuration (for RAG embeddings and chat)
    openai_api_key: Optional[str] = None  # Make optional for now
    
    # PostgreSQL Configuration
    postgres_host: str = "127.0.0.1"  # Use IPv4 explicitly to avoid IPv6 connection issues
    postgres_port: int = 5433  # Using 5433 to avoid conflict with local PostgreSQL
    postgres_db: str = "document_processor"
    postgres_user: str = "docuser"
    postgres_password: str = "docpass_dev_2026"
    
    # Redis Configuration (still used for task metadata and rate limiting)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Database settings
    debug_sql: bool = False  # Set to True to log SQL queries
    
    # Storage Configuration (legacy - keeping for backward compatibility)
    storage_path: str = "./storage/uploads"
    max_file_size_mb: int = 50
    max_files_per_request: int = 100
    
    # Celery Configuration (deprecated - will be removed)
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    
    # Rate Limiting
    rate_limit_requests: int = 10
    rate_limit_window: int = 60
    
    # Task Configuration
    task_result_ttl: int = 3600  # 1 hour in seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL database URL (sync, psycopg2)"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def async_database_url(self) -> str:
        """Construct async PostgreSQL database URL (asyncpg driver)"""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    def ensure_storage_path(self):
        """Create storage directory if it doesn't exist"""
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_storage_path()
