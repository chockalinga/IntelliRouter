"""Routing decision and classification models."""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

from .providers import TaskType, ModelMetadata


class ComplexityLevel(str, Enum):
    """Complexity levels for prompt classification."""
    
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class LatencyRequirement(str, Enum):
    """Latency requirement levels."""
    
    REAL_TIME = "real_time"  # < 1s
    FAST = "fast"  # < 3s
    NORMAL = "normal"  # < 10s
    SLOW = "slow"  # > 10s


class CostPriority(str, Enum):
    """Cost optimization priorities."""
    
    MINIMIZE_COST = "minimize_cost"
    BALANCE = "balance"
    MAXIMIZE_QUALITY = "maximize_quality"


class PromptClassification(BaseModel):
    """Classification results for a prompt."""
    
    task_type: TaskType = Field(description="Detected task type")
    complexity_level: ComplexityLevel = Field(description="Complexity assessment")
    estimated_tokens: int = Field(description="Estimated token count")
    requires_tools: bool = Field(default=False, description="Whether tools are needed")
    requires_json_mode: bool = Field(default=False, description="Whether JSON mode is needed")
    requires_vision: bool = Field(default=False, description="Whether vision is needed")
    
    # Content analysis
    has_code: bool = Field(default=False, description="Contains code")
    has_math: bool = Field(default=False, description="Contains mathematical content")
    has_structured_data: bool = Field(default=False, description="Contains structured data")
    is_conversational: bool = Field(default=False, description="Conversational context")
    
    # Inferred requirements
    latency_requirement: LatencyRequirement = Field(description="Inferred latency needs")
    quality_threshold: float = Field(ge=0.0, le=1.0, description="Minimum quality threshold")
    
    # Classification metadata
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")
    classification_time_ms: int = Field(description="Time taken for classification")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class RoutingConstraints(BaseModel):
    """Constraints for model routing."""
    
    # Cost constraints
    max_input_cost_per_1k: Optional[float] = Field(default=None, description="Max input cost")
    max_output_cost_per_1k: Optional[float] = Field(default=None, description="Max output cost")
    max_total_cost: Optional[float] = Field(default=None, description="Max total request cost")
    
    # Performance constraints
    max_latency_ms: Optional[int] = Field(default=None, description="Max latency")
    min_reliability_score: float = Field(default=0.95, description="Min reliability")
    
    # Capability requirements
    required_capabilities: List[str] = Field(default=[], description="Required capabilities")
    
    # Quality requirements
    min_quality_score: float = Field(default=0.0, description="Minimum quality score")
    
    # Geographic constraints
    allowed_regions: Optional[List[str]] = Field(default=None, description="Allowed regions")
    data_residency_requirements: Optional[List[str]] = Field(
        default=None,
        description="Data residency requirements"
    )
    
    # Compliance constraints
    required_certifications: List[str] = Field(
        default=[],
        description="Required compliance certifications"
    )
    
    # Provider constraints
    excluded_providers: List[str] = Field(default=[], description="Excluded providers")
    preferred_providers: List[str] = Field(default=[], description="Preferred providers")
    
    # Context constraints
    max_context_window: Optional[int] = Field(
        default=None,
        description="Maximum context window needed"
    )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class RoutingPolicy(BaseModel):
    """Policy configuration for routing decisions."""
    
    # Optimization strategy
    cost_priority: CostPriority = Field(default=CostPriority.BALANCE, description="Cost priority")
    
    # Weights for scoring (must sum to 1.0)
    quality_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Quality weight")
    cost_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Cost weight")
    latency_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Latency weight")
    reliability_weight: float = Field(default=0.1, ge=0.0, le=1.0, description="Reliability weight")
    
    # Fallback configuration
    enable_fallback: bool = Field(default=True, description="Enable fallback on errors")
    fallback_models: List[str] = Field(default=[], description="Fallback model order")
    max_fallback_attempts: int = Field(default=2, description="Max fallback attempts")
    
    # Session and caching
    enable_session_routing: bool = Field(default=True, description="Enable session-aware routing")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    
    # A/B testing
    enable_canary_routing: bool = Field(default=False, description="Enable canary routing")
    canary_percentage: float = Field(default=0.05, ge=0.0, le=1.0, description="Canary traffic %")
    canary_models: List[str] = Field(default=[], description="Canary models to test")
    
    # Safety and guardrails
    enable_safety_checks: bool = Field(default=True, description="Enable safety checks")
    pii_detection: bool = Field(default=True, description="Enable PII detection")
    content_filtering: bool = Field(default=True, description="Enable content filtering")
    
    def validate_weights(self) -> bool:
        """Validate that weights sum to 1.0."""
        total = self.quality_weight + self.cost_weight + self.latency_weight + self.reliability_weight
        return abs(total - 1.0) < 0.01
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class ModelScore(BaseModel):
    """Scoring information for a model candidate."""
    
    model_id: str = Field(description="Model identifier")
    total_score: float = Field(description="Total weighted score")
    
    # Individual scores
    quality_score: float = Field(description="Quality score")
    cost_score: float = Field(description="Cost score (inverse of cost)")
    latency_score: float = Field(description="Latency score (inverse of latency)")
    reliability_score: float = Field(description="Reliability score")
    
    # Cost estimates
    estimated_input_cost: float = Field(description="Estimated input cost")
    estimated_output_cost: float = Field(description="Estimated output cost")
    estimated_total_cost: float = Field(description="Estimated total cost")
    
    # Constraint satisfaction
    meets_constraints: bool = Field(description="Whether model meets all constraints")
    constraint_violations: List[str] = Field(default=[], description="Constraint violations")
    
    # Additional metadata
    reasoning: str = Field(description="Reasoning for score")


class RoutingDecision(BaseModel):
    """Complete routing decision with reasoning."""
    
    # Selected model
    selected_model: ModelMetadata = Field(description="Selected model")
    selected_model_score: ModelScore = Field(description="Score for selected model")
    
    # Decision context
    classification: PromptClassification = Field(description="Prompt classification")
    constraints: RoutingConstraints = Field(description="Applied constraints")
    policy: RoutingPolicy = Field(description="Routing policy used")
    
    # Alternative models considered
    candidate_models: List[ModelScore] = Field(
        default=[],
        description="All models considered with scores"
    )
    
    # Decision metadata
    decision_time_ms: int = Field(description="Time taken for routing decision")
    decision_confidence: float = Field(ge=0.0, le=1.0, description="Decision confidence")
    
    # Cost analysis
    baseline_model: Optional[str] = Field(default=None, description="Baseline for comparison")
    baseline_cost: Optional[float] = Field(default=None, description="Baseline cost")
    estimated_savings: Optional[float] = Field(default=None, description="Estimated savings")
    savings_percentage: Optional[float] = Field(default=None, description="Savings percentage")
    
    # Routing strategy used
    routing_strategy: str = Field(description="Strategy used for routing")
    fallback_available: bool = Field(default=False, description="Whether fallback is available")
    
    # Session context
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    conversation_context: bool = Field(default=False, description="Part of conversation")
    
    # Quality assurance
    quality_check_passed: bool = Field(default=True, description="Quality checks passed")
    safety_check_passed: bool = Field(default=True, description="Safety checks passed")
    
    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Decision timestamp")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class RoutingResult(BaseModel):
    """Result of a routing operation including execution."""
    
    decision: RoutingDecision = Field(description="Routing decision")
    
    # Execution results
    execution_successful: bool = Field(description="Whether execution was successful")
    actual_latency_ms: int = Field(description="Actual response latency")
    actual_cost: float = Field(description="Actual cost incurred")
    
    # Token usage
    actual_prompt_tokens: int = Field(description="Actual prompt tokens")
    actual_completion_tokens: int = Field(description="Actual completion tokens")
    actual_total_tokens: int = Field(description="Actual total tokens")
    
    # Quality metrics
    response_quality_score: Optional[float] = Field(
        default=None,
        description="Response quality score (if available)"
    )
    
    # Error handling
    errors_encountered: List[str] = Field(default=[], description="Errors during execution")
    fallback_used: bool = Field(default=False, description="Whether fallback was used")
    fallback_model: Optional[str] = Field(default=None, description="Fallback model used")
    
    # Performance tracking
    cache_hit: bool = Field(default=False, description="Whether response was cached")
    provider_response_time_ms: int = Field(description="Provider response time")
    
    # Validation
    response_validated: bool = Field(default=True, description="Response validation passed")
    validation_errors: List[str] = Field(default=[], description="Validation errors")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
