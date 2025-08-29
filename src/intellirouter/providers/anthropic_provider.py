"""Anthropic provider implementation."""

import json
import time
from typing import AsyncGenerator, Optional
import anthropic
from anthropic import AsyncAnthropic

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


class AnthropicProvider(BaseProvider):
    """Anthropic provider for chat completions."""
    
    def __init__(self, api_key: str, api_base: Optional[str] = None):
        """Initialize Anthropic provider."""
        super().__init__(api_key, api_base)
        
        # Initialize Anthropic client
        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        
        self.client = AsyncAnthropic(**client_kwargs)
    
    async def complete_chat(
        self,
        request: ChatCompletionRequest,
        model: ModelMetadata,
        request_id: str
    ) -> ChatCompletionResponse:
        """Complete a chat request using Anthropic."""
        
        start_time = time.time()
        
        # Convert to Anthropic format
        system_message, anthropic_messages = self._convert_messages(request.messages)
        
        # Prepare Anthropic request
        anthropic_request = {
            "model": model.id,
            "messages": anthropic_messages,
            "max_tokens": request.max_tokens or 1000,
        }
        
        # Add system message if present
        if system_message:
            anthropic_request["system"] = system_message
        
        # Add optional parameters
        if request.temperature is not None:
            anthropic_request["temperature"] = request.temperature
        if request.top_p is not None:
            anthropic_request["top_p"] = request.top_p
        if request.stop:
            anthropic_request["stop_sequences"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )
        
        try:
            # Make the request
            response = await self.client.messages.create(**anthropic_request)
            
            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract token usage
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens
            
            cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
            
            # Track metrics
            self._track_request(latency_ms, cost)
            
            # Convert to OpenAI-compatible format
            return ChatCompletionResponse(
                id=response.id,
                created=int(time.time()),
                model=response.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=self._extract_content(response.content)
                        ),
                        finish_reason=self._map_stop_reason(response.stop_reason)
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )
            )
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        model: ModelMetadata,
        request_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion using Anthropic."""
        
        # Convert to Anthropic format
        system_message, anthropic_messages = self._convert_messages(request.messages)
        
        # Prepare Anthropic request
        anthropic_request = {
            "model": model.id,
            "messages": anthropic_messages,
            "max_tokens": request.max_tokens or 1000,
            "stream": True,
        }
        
        # Add system message if present
        if system_message:
            anthropic_request["system"] = system_message
        
        # Add optional parameters
        if request.temperature is not None:
            anthropic_request["temperature"] = request.temperature
        if request.top_p is not None:
            anthropic_request["top_p"] = request.top_p
        
        try:
            stream = await self.client.messages.create(**anthropic_request)
            
            async for event in stream:
                if event.type == "content_block_delta":
                    # Convert to OpenAI-compatible streaming format
                    streaming_chunk = StreamingChunk(
                        id=request_id,
                        created=int(time.time()),
                        model=model.id,
                        choices=[{
                            "index": 0,
                            "delta": {
                                "role": None,
                                "content": event.delta.text if hasattr(event.delta, 'text') else None
                            },
                            "finish_reason": None
                        }]
                    )
                    
                    yield streaming_chunk.json()
                
                elif event.type == "message_stop":
                    # Send final chunk with finish reason
                    final_chunk = StreamingChunk(
                        id=request_id,
                        created=int(time.time()),
                        model=model.id,
                        choices=[{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    )
                    
                    yield final_chunk.json()
                
        except Exception as e:
            raise Exception(f"Anthropic streaming error: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check Anthropic API health."""
        try:
            # Simple health check - make a minimal request
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return response.id is not None
        except:
            return False
    
    def _convert_messages(self, messages: list[ChatMessage]) -> tuple[Optional[str], list[dict]]:
        """Convert our message format to Anthropic format."""
        system_message = None
        anthropic_messages = []
        
        for message in messages:
            if message.role == "system":
                # Anthropic handles system messages separately
                system_message = message.content
            elif message.role in ["user", "assistant"]:
                anthropic_msg = {
                    "role": message.role,
                    "content": message.content or ""
                }
                anthropic_messages.append(anthropic_msg)
        
        return system_message, anthropic_messages
    
    def _extract_content(self, content_blocks) -> str:
        """Extract text content from Anthropic response."""
        if not content_blocks:
            return ""
        
        # Handle both single content and list of content blocks
        if isinstance(content_blocks, list):
            text_parts = []
            for block in content_blocks:
                if hasattr(block, 'text'):
                    text_parts.append(block.text)
                elif isinstance(block, dict) and 'text' in block:
                    text_parts.append(block['text'])
            return "".join(text_parts)
        else:
            # Single content block
            if hasattr(content_blocks, 'text'):
                return content_blocks.text
            return str(content_blocks)
    
    def _map_stop_reason(self, stop_reason: str) -> str:
        """Map Anthropic stop reason to OpenAI format."""
        mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        return mapping.get(stop_reason, "stop")
