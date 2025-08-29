"""OpenAI provider implementation."""

import json
import time
from typing import AsyncGenerator, Optional
import openai
from openai import AsyncOpenAI

from .base import BaseProvider
from ..models.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatChoice,
    Usage,
    StreamingChunk,
)
from ..models.providers import ModelMetadata


class OpenAIProvider(BaseProvider):
    """OpenAI provider for chat completions."""
    
    def __init__(self, api_key: str, api_base: Optional[str] = None):
        """Initialize OpenAI provider."""
        super().__init__(api_key, api_base)
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        
        self.client = AsyncOpenAI(**client_kwargs)
    
    async def complete_chat(
        self,
        request: ChatCompletionRequest,
        model: ModelMetadata,
        request_id: str
    ) -> ChatCompletionResponse:
        """Complete a chat request using OpenAI."""
        
        start_time = time.time()
        
        # Convert to OpenAI format
        openai_messages = self._convert_messages(request.messages)
        
        # Prepare OpenAI request
        openai_request = {
            "model": model.id,
            "messages": openai_messages,
            "stream": False,
        }
        
        # Add optional parameters
        if request.temperature is not None:
            openai_request["temperature"] = request.temperature
        if request.max_tokens is not None:
            openai_request["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            openai_request["top_p"] = request.top_p
        if request.stop:
            openai_request["stop"] = request.stop
        if request.presence_penalty is not None:
            openai_request["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            openai_request["frequency_penalty"] = request.frequency_penalty
        if request.tools:
            openai_request["tools"] = self._convert_tools(request.tools)
        if request.tool_choice:
            openai_request["tool_choice"] = request.tool_choice
        if request.response_format:
            openai_request["response_format"] = {"type": request.response_format.type}
        
        try:
            # Make the request
            response = await self.client.chat.completions.create(**openai_request)
            
            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            cost = self._calculate_cost(
                model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            
            # Track metrics
            self._track_request(latency_ms, cost)
            
            # Convert to our format
            return ChatCompletionResponse(
                id=response.id,
                created=response.created,
                model=response.model,
                choices=[
                    ChatChoice(
                        index=choice.index,
                        message=ChatMessage(
                            role=choice.message.role,
                            content=choice.message.content,
                            tool_calls=choice.message.tool_calls
                        ),
                        finish_reason=choice.finish_reason,
                        logprobs=choice.logprobs
                    )
                    for choice in response.choices
                ],
                usage=Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                ),
                system_fingerprint=getattr(response, 'system_fingerprint', None)
            )
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        model: ModelMetadata,
        request_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion using OpenAI."""
        
        # Convert to OpenAI format
        openai_messages = self._convert_messages(request.messages)
        
        # Prepare OpenAI request
        openai_request = {
            "model": model.id,
            "messages": openai_messages,
            "stream": True,
        }
        
        # Add optional parameters (same as complete_chat)
        if request.temperature is not None:
            openai_request["temperature"] = request.temperature
        if request.max_tokens is not None:
            openai_request["max_tokens"] = request.max_tokens
        # ... (other parameters)
        
        try:
            stream = await self.client.chat.completions.create(**openai_request)
            
            async for chunk in stream:
                # Convert chunk to our format
                streaming_chunk = StreamingChunk(
                    id=chunk.id,
                    created=chunk.created,
                    model=chunk.model,
                    choices=[
                        {
                            "index": choice.index,
                            "delta": {
                                "role": choice.delta.role if choice.delta.role else None,
                                "content": choice.delta.content if choice.delta.content else None,
                                "tool_calls": choice.delta.tool_calls if choice.delta.tool_calls else None
                            },
                            "finish_reason": choice.finish_reason
                        }
                        for choice in chunk.choices
                    ],
                    system_fingerprint=getattr(chunk, 'system_fingerprint', None)
                )
                
                yield streaming_chunk.json()
                
        except Exception as e:
            raise Exception(f"OpenAI streaming error: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            # Simple health check - list models
            models = await self.client.models.list()
            return len(models.data) > 0
        except:
            return False
    
    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert our message format to OpenAI format."""
        openai_messages = []
        
        for message in messages:
            openai_msg = {
                "role": message.role,
                "content": message.content
            }
            
            if message.name:
                openai_msg["name"] = message.name
            if message.tool_calls:
                openai_msg["tool_calls"] = message.tool_calls
            if message.tool_call_id:
                openai_msg["tool_call_id"] = message.tool_call_id
            
            openai_messages.append(openai_msg)
        
        return openai_messages
    
    def _convert_tools(self, tools: list) -> list[dict]:
        """Convert our tools format to OpenAI format."""
        openai_tools = []
        
        for tool in tools:
            openai_tool = {
                "type": tool.type,
                "function": {
                    "name": tool.function.name,
                    "parameters": tool.function.parameters
                }
            }
            
            if tool.function.description:
                openai_tool["function"]["description"] = tool.function.description
            
            openai_tools.append(openai_tool)
        
        return openai_tools
