"""Provider factory for creating AI model provider instances."""

from typing import Dict, Optional
from ..models.providers import ProviderType
from ..config import settings
from .base import BaseProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .azure_openai_provider import AzureOpenAIProvider
from .aws_bedrock_provider import AWSBedrockProvider


class ProviderFactory:
    """Factory for creating provider instances."""
    
    def __init__(self):
        """Initialize the provider factory."""
        self._providers: Dict[ProviderType, BaseProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available providers based on configuration."""
        
        # OpenAI Provider (direct API)
        if settings.openai_api_key:
            self._providers[ProviderType.OPENAI] = OpenAIProvider(
                api_key=settings.openai_api_key
            )
        
        # Anthropic Provider (direct API)
        if settings.anthropic_api_key:
            self._providers[ProviderType.ANTHROPIC] = AnthropicProvider(
                api_key=settings.anthropic_api_key
            )
        
        # Azure OpenAI Provider (enterprise)
        if settings.azure_openai_api_key and settings.azure_openai_endpoint:
            self._providers[ProviderType.AZURE_OPENAI] = AzureOpenAIProvider(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=getattr(settings, 'azure_openai_api_version', '2024-02-15-preview')
            )
        
        # AWS Bedrock Provider (enterprise)
        if (settings.aws_access_key_id and settings.aws_secret_access_key):
            self._providers[ProviderType.AWS_BEDROCK] = AWSBedrockProvider(
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                aws_region=getattr(settings, 'aws_region', 'us-east-1')
            )
    
    def get_provider(self, provider_type: ProviderType) -> BaseProvider:
        """Get a provider instance.
        
        Args:
            provider_type: Type of provider to get
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider is not available
        """
        if provider_type not in self._providers:
            raise ValueError(f"Provider {provider_type} is not configured or available")
        
        return self._providers[provider_type]
    
    def get_available_providers(self) -> list[ProviderType]:
        """Get list of available providers."""
        return list(self._providers.keys())
    
    def is_provider_available(self, provider_type: ProviderType) -> bool:
        """Check if a provider is available."""
        return provider_type in self._providers
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Health check all providers."""
        results = {}
        
        for provider_type, provider in self._providers.items():
            try:
                health = await provider.health_check()
                results[provider_type] = health
            except Exception as e:
                results[provider_type] = False
        
        return results
    
    def get_provider_stats(self) -> Dict[str, dict]:
        """Get statistics for all providers."""
        stats = {}
        
        for provider_type, provider in self._providers.items():
            stats[provider_type] = provider.get_stats()
        
        return stats
