"""Chat completion models for OpenAI-compatible API."""

from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class ChatMessage(BaseModel):
    """A message in a chat completion request."""
    
    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="The role of the message author"
    )
    content: Optional[str] = Field(
        default=None,
        description="The content of the message"
    )
    name: Optional[str] = Field(
        default=None,
        description="An optional name for the participant"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tool calls made by the assistant"
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Tool call ID for tool messages"
    )


class Function(BaseModel):
    """Function definition for tool use."""
    
    name: str = Field(description="Function name")
    description: Optional[str] = Field(default=None, description="Function description")
    parameters: Dict[str, Any] = Field(description="Function parameters schema")


class Tool(BaseModel):
    """Tool definition for function calling."""
    
    type: Literal["function"] = Field(default="function", description="Tool type")
    function: Function = Field(description="Function definition")


class ResponseFormat(BaseModel):
    """Response format specification."""
    
    type: Literal["text", "json_object"] = Field(
        default="text",
        description="Response format type"
    )


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    
    model: str = Field(
        default="auto",
        description="Model to use, or 'auto' for automatic routing"
    )
    messages: List[ChatMessage] = Field(
        min_items=1,
        description="List of messages comprising the conversation"
    )
    temperature: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of tokens to generate"
    )
    top_p: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    n: Optional[int] = Field(
        default=1,
        ge=1,
        le=1,
        description="Number of completions to generate"
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream partial results"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Stop sequences"
    )
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty"
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty"
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        default=None,
        description="Logit bias modifications"
    )
    user: Optional[str] = Field(
        default=None,
        description="Unique identifier for end-user"
    )
    tools: Optional[List[Tool]] = Field(
        default=None,
        description="Available tools for function calling"
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Tool choice specification"
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description="Response format specification"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for deterministic outputs"
    )
    
    # DynaRoute-specific fields
    batch_id: Optional[str] = Field(
        default=None,
        description="Optional batch ID for request grouping"
    )
    cache_key: Optional[str] = Field(
        default=None,
        description="Optional semantic cache key"
    )
    routing_preferences: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Routing preferences (cost vs quality vs latency)"
    )


class Usage(BaseModel):
    """Token usage information."""
    
    prompt_tokens: int = Field(description="Tokens in the prompt")
    completion_tokens: int = Field(description="Tokens in the completion")
    total_tokens: int = Field(description="Total tokens used")


class RouterInfo(BaseModel):
    """Information about routing decision."""
    
    selected_model: str = Field(description="The model that was selected")
    reason: str = Field(description="Reason for model selection")
    latency_ms: int = Field(description="Response latency in milliseconds")
    cost: float = Field(description="Cost of the request in USD")
    savings_vs_baseline: Optional[float] = Field(
        default=None,
        description="Cost savings compared to baseline model"
    )
    baseline_model: Optional[str] = Field(
        default=None,
        description="Baseline model for cost comparison"
    )
    classification: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Prompt classification details"
    )
    fallback_used: bool = Field(
        default=False,
        description="Whether fallback was used"
    )


class ChatChoice(BaseModel):
    """A chat completion choice."""
    
    index: int = Field(description="Choice index")
    message: ChatMessage = Field(description="The generated message")
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "end_turn"]] = Field(
        default=None,
        description="Reason for completion finish"
    )
    logprobs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Log probabilities (if requested)"
    )


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    
    id: str = Field(description="Unique completion ID")
    object: Literal["chat.completion"] = Field(
        default="chat.completion",
        description="Response object type"
    )
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used for completion")
    choices: List[ChatChoice] = Field(description="Completion choices")
    usage: Usage = Field(description="Token usage information")
    system_fingerprint: Optional[str] = Field(
        default=None,
        description="System fingerprint"
    )
    
    # DynaRoute-specific fields
    router_info: Optional[RouterInfo] = Field(
        default=None,
        description="Router decision information"
    )


class StreamingChunk(BaseModel):
    """Streaming response chunk."""
    
    id: str = Field(description="Unique completion ID")
    object: Literal["chat.completion.chunk"] = Field(
        default="chat.completion.chunk",
        description="Response object type"
    )
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used for completion")
    choices: List[Dict[str, Any]] = Field(description="Streaming choices")
    system_fingerprint: Optional[str] = Field(
        default=None,
        description="System fingerprint"
    )


class StreamingResponse(BaseModel):
    """Streaming response wrapper."""
    
    chunk: StreamingChunk = Field(description="Streaming chunk")
    router_info: Optional[RouterInfo] = Field(
        default=None,
        description="Router information (sent with first chunk)"
    )
