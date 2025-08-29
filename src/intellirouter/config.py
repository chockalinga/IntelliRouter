"""Configuration management for DynaRoute."""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_version: str = Field(default="v1", description="API version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/dynaroute",
        description="Database URL"
    )
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    
    # Authentication
    secret_key: str = Field(..., description="Secret key for JWT")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=1440, description="Token expiration")
    
    # AI Provider API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    google_api_key: Optional[str] = Field(default=None, description="Google API key")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret key")
    aws_region: str = Field(default="us-east-1", description="AWS region")
    azure_openai_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint")
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview", 
        description="Azure OpenAI API version"
    )
    
    # Router Configuration
    default_model: str = Field(default="gpt-4o-mini", description="Default model")
    fallback_model: str = Field(default="gpt-3.5-turbo", description="Fallback model")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    
    # Analytics
    enable_analytics: bool = Field(default=True, description="Enable analytics")
    analytics_retention_days: int = Field(default=30, description="Analytics retention period")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Logging format")
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(
        default=60, 
        description="Rate limit per minute"
    )
    rate_limit_burst: int = Field(default=10, description="Rate limit burst")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
