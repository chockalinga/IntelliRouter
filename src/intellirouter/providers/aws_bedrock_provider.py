"""AWS Bedrock provider for Anthropic models."""

import json
import uuid
from typing import AsyncGenerator, Optional, Dict, Any
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import asyncio
import time

from .base import BaseProvider
from ..models.chat import (
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    ChatChoice, Usage, RouterInfo
)
from ..models.providers import ModelMetadata


class AWSBedrockProvider(BaseProvider):
    """AWS Bedrock provider for Anthropic models."""
    
    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_region: str = "us-east-1",
        aws_session_token: Optional[str] = None
    ):
        """Initialize AWS Bedrock provider.
        
        Args:
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_region: AWS region (default: us-east-1)
            aws_session_token: Optional session token for temporary credentials
        """
        super().__init__(api_key=aws_access_key_id)
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region
        self.aws_session_token = aws_session_token
        
        # Initialize Bedrock client
        session_kwargs = {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
            'region_name': aws_region
        }
        
        if aws_session_token:
            session_kwargs['aws_session_token'] = aws_session_token
        
        self.session = boto3.Session(**session_kwargs)
        self.bedrock_client = self.session.client('bedrock-runtime')
    
    def _convert_to_anthropic_format(self, messages: list[ChatMessage]) -> Dict[str, Any]:
        """Convert OpenAI format messages to Anthropic format for Bedrock."""
        
        # Find system message
        system_message = None
        conversation_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Anthropic expects alternating user/assistant messages
        # Ensure it starts with user
        if conversation_messages and conversation_messages[0]["role"] != "user":
            conversation_messages.insert(0, {
                "role": "user",
                "content": "Please help me with the following:"
            })
        
        body = {
            "messages": conversation_messages,
            "max_tokens": 4000,  # Default max tokens
            "anthropic_version": "bedrock-2023-05-31"
        }
        
        if system_message:
            body["system"] = system_message
        
        return body
    
    def _convert_from_anthropic_format(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic Bedrock response to OpenAI format."""
        
        content = ""
        if "content" in response and response["content"]:
            # Anthropic returns content as a list
            if isinstance(response["content"], list):
                content = response["content"][0].get("text", "")
            else:
                content = response["content"]
        
        usage = response.get("usage", {})
        
        return {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": response.get("stop_reason", "stop")
            }],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            }
        }
    
    async def complete_chat(
        self,
        request: ChatCompletionRequest,
        model: ModelMetadata,
        request_id: str
    ) -> ChatCompletionResponse:
        """Complete a chat request using AWS Bedrock."""
        
        start_time = time.time()
        
        try:
            # Convert to Anthropic format
            body = self._convert_to_anthropic_format(request.messages)
            
            # Override max_tokens if specified in request
            if request.max_tokens:
                body["max_tokens"] = request.max_tokens
            
            if request.temperature is not None:
                body["temperature"] = request.temperature
            
            if request.top_p is not None:
                body["top_p"] = request.top_p
            
            # Make request to Bedrock
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.bedrock_client.invoke_model(
                    modelId=model.id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json"
                )
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Convert to OpenAI format
            openai_response = self._convert_from_anthropic_format(response_body)
            
            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            usage = openai_response["usage"]
            cost = self._calculate_cost(model, usage["prompt_tokens"], usage["completion_tokens"])
            
            # Track metrics
            self._track_request(latency_ms, cost)
            
            # Create router info
            router_info = RouterInfo(
                selected_model=model.id,
                reason=f"AWS Bedrock: {model.name}",
                latency_ms=latency_ms,
                cost=cost,
                provider="aws_bedrock",
                classification={}
            )
            
            return ChatCompletionResponse(
                id=f"chatcmpl-{request_id}",
                object="chat.completion",
                created=int(time.time()),
                model=model.id,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=openai_response["choices"][0]["message"]["content"]
                        ),
                        finish_reason=openai_response["choices"][0]["finish_reason"]
                    )
                ],
                usage=Usage(
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"]
                ),
                router_info=router_info
            )
            
        except (ClientError, BotoCoreError) as e:
            raise Exception(f"AWS Bedrock error: {str(e)}")
        except Exception as e:
            raise Exception(f"Bedrock provider error: {str(e)}")
    
    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        model: ModelMetadata,
        request_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion using AWS Bedrock."""
        
        # Note: AWS Bedrock streaming requires different API calls
        # For now, fall back to non-streaming and yield the complete response
        response = await self.complete_chat(request, model, request_id)
        
        # Simulate streaming by yielding chunks
        content = response.choices[0].message.content
        words = content.split()
        
        for i, word in enumerate(words):
            chunk_data = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model.id,
                "choices": [{
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": None
                }]
            }
            
            yield f"data: {json.dumps(chunk_data)}\n\n"
            await asyncio.sleep(0.05)  # Small delay for streaming effect
        
        # Final chunk
        final_chunk = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model.id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    async def health_check(self) -> bool:
        """Check if AWS Bedrock is accessible."""
        try:
            # Try to list available models
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.bedrock_client.list_foundation_models()
            )
            return True
        except Exception:
            return False
