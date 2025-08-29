"""Base provider interface for AI model providers."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
import time

from ..models.chat import ChatCompletionRequest, ChatCompletionResponse
from ..models.providers import ModelMetadata


class BaseProvider(ABC):
    """Abstract base class for AI model providers."""
    
    def __init__(self, api_key: str, api_base: Optional[str] = None):
        """Initialize the provider.
        
        Args:
            api_key: API key for authentication
            api_base: Optional custom API base URL
        """
        self.api_key = api_key
        self.api_base = api_base
        self._request_count = 0
        self._total_cost = 0.0
        self._total_latency = 0.0
    
    @abstractmethod
    async def complete_chat(
        self,
        request: ChatCompletionRequest,
        model: ModelMetadata,
        request_id: str
    ) -> ChatCompletionResponse:
        """Complete a chat request.
        
        Args:
            request: The chat completion request
            model: Model metadata
            request_id: Unique request identifier
            
        Returns:
            Chat completion response
        """
        pass
    
    @abstractmethod
    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        model: ModelMetadata,
        request_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion.
        
        Args:
            request: The chat completion request
            model: Model metadata
            request_id: Unique request identifier
            
        Yields:
            JSON chunks for streaming response
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    def _track_request(self, latency_ms: int, cost: float):
        """Track request metrics."""
        self._request_count += 1
        self._total_cost += cost
        self._total_latency += latency_ms
    
    def get_stats(self) -> dict:
        """Get provider statistics."""
        avg_latency = (
            self._total_latency / self._request_count 
            if self._request_count > 0 else 0
        )
        
        return {
            "request_count": self._request_count,
            "total_cost": self._total_cost,
            "average_latency_ms": avg_latency,
            "average_cost_per_request": (
                self._total_cost / self._request_count 
                if self._request_count > 0 else 0
            )
        }
    
    def _calculate_cost(
        self,
        model: ModelMetadata,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate cost for a request."""
        
        input_cost = (prompt_tokens / 1000) * model.input_price_per_1k
        output_cost = (completion_tokens / 1000) * model.output_price_per_1k
        
        return input_cost + output_cost
