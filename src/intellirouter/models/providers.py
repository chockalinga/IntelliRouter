"""Provider and model metadata models."""

from typing import Dict, List, Optional, Set, Any
from pydantic import BaseModel, Field
from enum import Enum


class ProviderType(str, Enum):
    """Supported AI providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AWS_BEDROCK = "aws_bedrock"
    AZURE_OPENAI = "azure_openai"
    SELF_HOSTED = "self_hosted"


class TaskType(str, Enum):
    """Types of tasks for quality scoring."""
    
    CHAT = "chat"
    REASONING = "reasoning"
    CODING = "coding"
    RAG = "rag"
    STRUCTURED_OUTPUT = "structured_output"
    CREATIVE_WRITING = "creative_writing"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"


class QualityScores(BaseModel):
    """Quality scores for different task types."""
    
    reasoning: float = Field(ge=0.0, le=1.0, description="Reasoning task quality")
    coding: float = Field(ge=0.0, le=1.0, description="Code generation quality")
    chat: float = Field(ge=0.0, le=1.0, description="Chat conversation quality")
    rag: float = Field(ge=0.0, le=1.0, description="RAG/QA task quality")
    structured_output: float = Field(ge=0.0, le=1.0, description="Structured output quality")
    creative_writing: float = Field(ge=0.0, le=1.0, description="Creative writing quality")
    translation: float = Field(ge=0.0, le=1.0, description="Translation quality")
    summarization: float = Field(ge=0.0, le=1.0, description="Summarization quality")


class ModelCapabilities(BaseModel):
    """Model capabilities and features."""
    
    supports_tools: bool = Field(default=False, description="Function calling support")
    supports_streaming: bool = Field(default=True, description="Streaming support")
    supports_json_mode: bool = Field(default=False, description="JSON mode support")
    supports_vision: bool = Field(default=False, description="Vision/image support")
    supports_system_message: bool = Field(default=True, description="System message support")
    max_output_tokens: Optional[int] = Field(default=None, description="Max output tokens")
    input_modalities: List[str] = Field(default=["text"], description="Supported input types")
    output_modalities: List[str] = Field(default=["text"], description="Supported output types")


class ModelMetadata(BaseModel):
    """Comprehensive model metadata."""
    
    id: str = Field(description="Model identifier")
    name: str = Field(description="Human-readable model name")
    provider: ProviderType = Field(description="Provider type")
    
    # Pricing information (per 1K tokens)
    input_price_per_1k: float = Field(ge=0.0, description="Input token price per 1K")
    output_price_per_1k: float = Field(ge=0.0, description="Output token price per 1K")
    
    # Model specifications
    context_window: int = Field(gt=0, description="Context window size in tokens")
    training_cutoff: Optional[str] = Field(default=None, description="Training data cutoff")
    
    # Performance metrics
    avg_latency_ms: int = Field(gt=0, description="Average latency in milliseconds")
    reliability_score: float = Field(ge=0.0, le=1.0, description="Reliability score")
    
    # Quality scores for different tasks
    quality_scores: QualityScores = Field(description="Quality scores by task type")
    
    # Capabilities
    capabilities: ModelCapabilities = Field(description="Model capabilities")
    
    # Geographic and compliance
    regions: List[str] = Field(default=[], description="Available regions")
    data_residency: List[str] = Field(default=[], description="Data residency options")
    compliance_certifications: List[str] = Field(
        default=[],
        description="Compliance certifications (SOC2, HIPAA, etc.)"
    )
    
    # Operational
    is_active: bool = Field(default=True, description="Whether model is currently active")
    rate_limit_rpm: Optional[int] = Field(default=None, description="Rate limit per minute")
    rate_limit_tpm: Optional[int] = Field(default=None, description="Token limit per minute")
    
    # Additional metadata
    description: Optional[str] = Field(default=None, description="Model description")
    tags: List[str] = Field(default=[], description="Model tags for categorization")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class ProviderConfig(BaseModel):
    """Provider-specific configuration."""
    
    provider: ProviderType = Field(description="Provider type")
    api_key: Optional[str] = Field(default=None, description="API key")
    api_base: Optional[str] = Field(default=None, description="API base URL")
    api_version: Optional[str] = Field(default=None, description="API version")
    
    # Rate limiting
    requests_per_minute: int = Field(default=60, description="Requests per minute limit")
    tokens_per_minute: int = Field(default=10000, description="Tokens per minute limit")
    
    # Retry configuration
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Base retry delay in seconds")
    
    # Timeout configuration
    connect_timeout: float = Field(default=10.0, description="Connection timeout")
    read_timeout: float = Field(default=30.0, description="Read timeout")
    
    # Provider-specific settings
    custom_headers: Dict[str, str] = Field(default={}, description="Custom headers")
    extra_config: Dict[str, Any] = Field(default={}, description="Provider-specific config")
    
    # Regional settings
    preferred_regions: List[str] = Field(default=[], description="Preferred regions")
    
    # Health check
    health_check_enabled: bool = Field(default=True, description="Enable health checks")
    health_check_interval: int = Field(default=300, description="Health check interval in seconds")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class ModelRegistry(BaseModel):
    """Registry of all available models."""
    
    models: Dict[str, ModelMetadata] = Field(default={}, description="Model metadata by ID")
    providers: Dict[ProviderType, ProviderConfig] = Field(
        default={},
        description="Provider configurations"
    )
    
    def get_models_by_provider(self, provider: ProviderType) -> List[ModelMetadata]:
        """Get all models for a specific provider."""
        return [
            model for model in self.models.values()
            if model.provider == provider and model.is_active
        ]
    
    def get_models_by_capability(self, capability: str) -> List[ModelMetadata]:
        """Get models that support a specific capability."""
        models = []
        for model in self.models.values():
            if not model.is_active:
                continue
            
            capabilities = model.capabilities
            if capability == "tools" and capabilities.supports_tools:
                models.append(model)
            elif capability == "streaming" and capabilities.supports_streaming:
                models.append(model)
            elif capability == "json_mode" and capabilities.supports_json_mode:
                models.append(model)
            elif capability == "vision" and capabilities.supports_vision:
                models.append(model)
        
        return models
    
    def get_models_by_task_quality(
        self, 
        task_type: TaskType, 
        min_quality: float = 0.0
    ) -> List[ModelMetadata]:
        """Get models above a quality threshold for a task type."""
        models = []
        for model in self.models.values():
            if not model.is_active:
                continue
            
            quality_score = getattr(model.quality_scores, task_type, 0.0)
            if quality_score >= min_quality:
                models.append(model)
        
        return sorted(models, key=lambda m: getattr(m.quality_scores, task_type), reverse=True)
    
    def get_models_by_cost_range(
        self, 
        max_input_cost: Optional[float] = None,
        max_output_cost: Optional[float] = None
    ) -> List[ModelMetadata]:
        """Get models within a cost range."""
        models = []
        for model in self.models.values():
            if not model.is_active:
                continue
            
            if max_input_cost and model.input_price_per_1k > max_input_cost:
                continue
            if max_output_cost and model.output_price_per_1k > max_output_cost:
                continue
            
            models.append(model)
        
        return models
    
    def get_best_model_for_task(
        self,
        task_type: TaskType,
        max_cost_per_1k: Optional[float] = None,
        required_capabilities: Optional[List[str]] = None,
        max_latency_ms: Optional[int] = None
    ) -> Optional[ModelMetadata]:
        """Get the best model for a task considering constraints."""
        candidates = self.get_models_by_task_quality(task_type, min_quality=0.0)
        
        # Apply cost constraint
        if max_cost_per_1k:
            candidates = [
                m for m in candidates 
                if max(m.input_price_per_1k, m.output_price_per_1k) <= max_cost_per_1k
            ]
        
        # Apply capability constraints
        if required_capabilities:
            for capability in required_capabilities:
                candidates = [
                    m for m in candidates
                    if self._has_capability(m, capability)
                ]
        
        # Apply latency constraint
        if max_latency_ms:
            candidates = [
                m for m in candidates
                if m.avg_latency_ms <= max_latency_ms
            ]
        
        # Return the highest quality model that meets constraints
        return candidates[0] if candidates else None
    
    def _has_capability(self, model: ModelMetadata, capability: str) -> bool:
        """Check if model has a specific capability."""
        capabilities = model.capabilities
        
        capability_map = {
            "tools": capabilities.supports_tools,
            "streaming": capabilities.supports_streaming,
            "json_mode": capabilities.supports_json_mode,
            "vision": capabilities.supports_vision,
            "system_message": capabilities.supports_system_message,
        }
        
        return capability_map.get(capability, False)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
