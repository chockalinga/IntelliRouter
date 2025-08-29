"""Azure OpenAI provider for OpenAI models."""

import json
import uuid
from typing import AsyncGenerator, Optional, Dict, Any
import httpx
import asyncio
import time

from .base import BaseProvider
from ..models.chat import (
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    ChatChoice, Usage, RouterInfo
)
from ..models.providers import ModelMetadata


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI provider for OpenAI models."""
    
    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str = "2024-02-15-preview",
        deployment_name: Optional[str] = None
    ):
        """Initialize Azure OpenAI provider.
        
        Args:
            api_key: Azure OpenAI API key
            azure_endpoint: Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com/)
            api_version: API version (default: 2024-02-15-preview)
            deployment_name: Optional deployment name override
        """
        super().__init__(api_key=api_key)
        self.azure_endpoint = azure_endpoint.rstrip('/')
        self.api_version = api_version
        self.deployment_name = deployment_name
        
        # Initialize httpx client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={
                "api-key": api_key,
                "Content-Type": "application/json"
            }
        )
    
    def _get_deployment_name(self, model_id: str) -> str:
        """Get deployment name for a model."""
        if self.deployment_name:
            return self.deployment_name
        
        # Map model IDs to common deployment names
        deployment_map = {
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-4": "gpt-4",
            "gpt-4-32k": "gpt-4-32k",
            "gpt-35-turbo": "gpt-35-turbo",
            "gpt-3.5-turbo": "gpt-35-turbo",  # Azure uses gpt-35-turbo
            "text-embedding-ada-002": "text-embedding-ada-002"
        }
        
        return deployment_map.get(model_id, model_id.replace(".", ""))
    
    def _build_url(self, model_id: str, endpoint: str = "chat/completions") -> str:
        """Build Azure OpenAI API URL."""
        deployment = self._get_deployment_name(model_id)
        return (
            f"{self.azure_endpoint}/openai/deployments/{deployment}/"
            f"{endpoint}?api-version={self.api_version}"
        )
    
    def _convert_request_to_azure_format(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Convert request to Azure OpenAI format."""
        
        body = {
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content
                }
                for msg in request.messages
            ]
        }
        
        # Add optional parameters
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        
        if request.temperature is not None:
            body["temperature"] = request.temperature
        
        if request.top_p is not None:
            body["top_p"] = request.top_p
        
        if request.frequency_penalty is not None:
            body["frequency_penalty"] = request.frequency_penalty
        
        if request.presence_penalty is not None:
            body["presence_penalty"] = request.presence_penalty
        
        if request.stop is not None:
            body["stop"] = request.stop
        
        if request.stream is not None:
            body["stream"] = request.stream
        
        # Add tools if specified
        if hasattr(request, 'tools') and request.tools:
            body["tools"] = request.tools
        
        if hasattr(request, 'tool_choice') and request.tool_choice:
            body["tool_choice"] = request.tool_choice
        
        return body
    
    async def complete_chat(
        self,
        request: ChatCompletionRequest,
        model: ModelMetadata,
        request_id: str
    ) -> ChatCompletionResponse:
        """Complete a chat request using Azure OpenAI."""
        
        start_time = time.time()
        
        try:
            # Build request
            url = self._build_url(model.id)
            body = self._convert_request_to_azure_format(request)
            
            # Make request
            response = await self.client.post(url, json=body)
            response.raise_for_status()
            
            response_data = response.json()
            
            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            usage_data = response_data.get("usage", {})
            cost = self._calculate_cost(
                model,
                usage_data.get("prompt_tokens", 0),
                usage_data.get("completion_tokens", 0)
            )
            
            # Track metrics
            self._track_request(latency_ms, cost)
            
            # Create router info
            router_info = RouterInfo(
                selected_model=model.id,
                reason=f"Azure OpenAI: {model.name}",
                latency_ms=latency_ms,
                cost=cost,
                provider="azure_openai",
                classification={}
            )
            
            return ChatCompletionResponse(
                id=response_data.get("id", f"chatcmpl-{request_id}"),
                object=response_data.get("object", "chat.completion"),
                created=response_data.get("created", int(time.time())),
                model=model.id,
                choices=[
                    ChatChoice(
                        index=choice.get("index", 0),
                        message=ChatMessage(
                            role=choice["message"]["role"],
                            content=choice["message"]["content"]
                        ),
                        finish_reason=choice.get("finish_reason", "stop")
                    )
                    for choice in response_data.get("choices", [])
                ],
                usage=Usage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0)
                ),
                router_info=router_info
            )
            
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except:
                error_detail = str(e)
            
            raise Exception(f"Azure OpenAI HTTP error: {error_detail}")
        except Exception as e:
            raise Exception(f"Azure OpenAI provider error: {str(e)}")
    
    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        model: ModelMetadata,
        request_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion using Azure OpenAI."""
        
        try:
            # Build request with streaming enabled
            url = self._build_url(model.id)
            body = self._convert_request_to_azure_format(request)
            body["stream"] = True
            
            # Make streaming request
            async with self.client.stream("POST", url, json=body) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        
                        if data == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        
                        try:
                            # Parse and potentially modify the chunk
                            chunk_data = json.loads(data)
                            
                            # Ensure model ID is correct
                            chunk_data["model"] = model.id
                            
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                        except json.JSONDecodeError:
                            # Skip invalid JSON chunks
                            continue
                        
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except:
                error_detail = str(e)
            
            # Yield error as a chunk
            error_chunk = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model.id,
                "error": {
                    "message": f"Azure OpenAI streaming error: {error_detail}",
                    "type": "api_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            # Yield error as a chunk
            error_chunk = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model.id,
                "error": {
                    "message": f"Azure OpenAI provider error: {str(e)}",
                    "type": "provider_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
    
    async def health_check(self) -> bool:
        """Check if Azure OpenAI is accessible."""
        try:
            # Try to make a simple request to test connectivity
            # We'll use a minimal chat completion request
            url = self._build_url("gpt-35-turbo")  # Use common deployment name
            body = {
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }
            
            response = await self.client.post(url, json=body)
            return response.status_code in [200, 400]  # 400 is OK for health check (bad request but service is up)
        except Exception:
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
