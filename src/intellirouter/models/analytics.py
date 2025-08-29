"""Analytics and logging models for DynaRoute."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from .providers import ProviderType, TaskType
from .routing import ComplexityLevel, LatencyRequirement


class RequestStatus(str, Enum):
    """Request status types."""
    
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    FALLBACK_USED = "fallback_used"


class ErrorType(str, Enum):
    """Error type categories."""
    
    PROVIDER_ERROR = "provider_error"
    ROUTING_ERROR = "routing_error"
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


class RequestLog(BaseModel):
    """Detailed log entry for each request."""
    
    # Request identification
    request_id: str = Field(description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    api_key_hash: Optional[str] = Field(default=None, description="Hashed API key for tracking")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    
    # Request details
    model_requested: str = Field(description="Model requested by user")
    model_used: str = Field(description="Actual model used")
    provider: ProviderType = Field(description="Provider used")
    
    # Classification and routing
    task_type: TaskType = Field(description="Detected task type")
    complexity_level: ComplexityLevel = Field(description="Complexity level")
    latency_requirement: LatencyRequirement = Field(description="Latency requirement")
    routing_strategy: str = Field(description="Routing strategy used")
    routing_reason: str = Field(description="Reason for model selection")
    
    # Token usage
    prompt_tokens: int = Field(description="Input tokens")
    completion_tokens: int = Field(description="Output tokens")
    total_tokens: int = Field(description="Total tokens")
    
    # Cost information
    input_cost: float = Field(description="Input cost in USD")
    output_cost: float = Field(description="Output cost in USD")
    total_cost: float = Field(description="Total cost in USD")
    
    # Performance metrics
    classification_time_ms: int = Field(description="Classification time")
    routing_time_ms: int = Field(description="Routing decision time")
    provider_response_time_ms: int = Field(description="Provider response time")
    total_latency_ms: int = Field(description="Total request latency")
    
    # Status and errors
    status: RequestStatus = Field(description="Request status")
    error_type: Optional[ErrorType] = Field(default=None, description="Error type if any")
    error_message: Optional[str] = Field(default=None, description="Error message")
    fallback_used: bool = Field(default=False, description="Whether fallback was used")
    fallback_model: Optional[str] = Field(default=None, description="Fallback model used")
    
    # Quality and satisfaction
    response_quality_score: Optional[float] = Field(
        default=None,
        description="Response quality score"
    )
    user_satisfaction: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="User satisfaction rating (1-5)"
    )
    
    # Caching
    cache_hit: bool = Field(default=False, description="Whether response was cached")
    cached_response_used: bool = Field(default=False, description="Used cached response")
    
    # Cost savings
    baseline_model: Optional[str] = Field(default=None, description="Baseline model for comparison")
    baseline_cost: Optional[float] = Field(default=None, description="Baseline cost")
    cost_savings: Optional[float] = Field(default=None, description="Cost savings achieved")
    savings_percentage: Optional[float] = Field(default=None, description="Savings percentage")
    
    # Request context
    has_tools: bool = Field(default=False, description="Request included tools")
    has_vision: bool = Field(default=False, description="Request included vision")
    stream_requested: bool = Field(default=False, description="Streaming was requested")
    json_mode_requested: bool = Field(default=False, description="JSON mode was requested")
    
    # Geographic and compliance
    region: Optional[str] = Field(default=None, description="Request region")
    data_residency: Optional[str] = Field(default=None, description="Data residency")
    
    # Additional metadata
    user_agent: Optional[str] = Field(default=None, description="User agent")
    client_version: Optional[str] = Field(default=None, description="Client version")
    request_source: Optional[str] = Field(default=None, description="Request source")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class UsageMetrics(BaseModel):
    """Aggregated usage metrics for analytics."""
    
    # Time period
    period_start: datetime = Field(description="Metrics period start")
    period_end: datetime = Field(description="Metrics period end")
    period_type: str = Field(description="Period type (hour, day, week, month)")
    
    # Request volume
    total_requests: int = Field(description="Total number of requests")
    successful_requests: int = Field(description="Number of successful requests")
    failed_requests: int = Field(description="Number of failed requests")
    error_rate: float = Field(ge=0.0, le=1.0, description="Error rate percentage")
    
    # Token usage
    total_prompt_tokens: int = Field(description="Total input tokens")
    total_completion_tokens: int = Field(description="Total output tokens")
    total_tokens: int = Field(description="Total tokens processed")
    avg_tokens_per_request: float = Field(description="Average tokens per request")
    
    # Cost metrics
    total_cost: float = Field(description="Total cost in USD")
    total_savings: float = Field(description="Total cost savings in USD")
    avg_cost_per_request: float = Field(description="Average cost per request")
    avg_savings_per_request: float = Field(description="Average savings per request")
    savings_percentage: float = Field(description="Overall savings percentage")
    
    # Performance metrics
    avg_latency_ms: float = Field(description="Average response latency")
    p50_latency_ms: float = Field(description="50th percentile latency")
    p95_latency_ms: float = Field(description="95th percentile latency")
    p99_latency_ms: float = Field(description="99th percentile latency")
    
    # Model usage distribution
    model_usage: Dict[str, int] = Field(
        default={},
        description="Number of requests per model"
    )
    provider_usage: Dict[str, int] = Field(
        default={},
        description="Number of requests per provider"
    )
    
    # Task distribution
    task_type_distribution: Dict[str, int] = Field(
        default={},
        description="Distribution of task types"
    )
    complexity_distribution: Dict[str, int] = Field(
        default={},
        description="Distribution of complexity levels"
    )
    
    # Quality metrics
    avg_quality_score: Optional[float] = Field(
        default=None,
        description="Average quality score"
    )
    avg_user_satisfaction: Optional[float] = Field(
        default=None,
        description="Average user satisfaction"
    )
    
    # Cache metrics
    cache_hit_rate: float = Field(ge=0.0, le=1.0, description="Cache hit rate")
    cache_savings: float = Field(description="Cost savings from caching")
    
    # Fallback metrics
    fallback_rate: float = Field(ge=0.0, le=1.0, description="Fallback usage rate")
    fallback_success_rate: float = Field(ge=0.0, le=1.0, description="Fallback success rate")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class CostAnalysis(BaseModel):
    """Detailed cost analysis and breakdown."""
    
    # Time period
    analysis_period: str = Field(description="Analysis period")
    period_start: datetime = Field(description="Analysis start time")
    period_end: datetime = Field(description="Analysis end time")
    
    # Overall cost metrics
    total_actual_cost: float = Field(description="Total actual cost incurred")
    total_baseline_cost: float = Field(description="Cost if using baseline model")
    total_savings: float = Field(description="Total cost savings")
    savings_percentage: float = Field(description="Percentage savings")
    
    # Cost breakdown by provider
    cost_by_provider: Dict[str, float] = Field(
        default={},
        description="Cost breakdown by provider"
    )
    savings_by_provider: Dict[str, float] = Field(
        default={},
        description="Savings breakdown by provider"
    )
    
    # Cost breakdown by model
    cost_by_model: Dict[str, float] = Field(
        default={},
        description="Cost breakdown by model"
    )
    usage_by_model: Dict[str, int] = Field(
        default={},
        description="Usage count by model"
    )
    
    # Cost breakdown by task type
    cost_by_task_type: Dict[str, float] = Field(
        default={},
        description="Cost breakdown by task type"
    )
    
    # Cost trends over time
    daily_costs: List[Dict[str, Any]] = Field(
        default=[],
        description="Daily cost breakdown"
    )
    
    # Top cost contributors
    top_expensive_requests: List[Dict[str, Any]] = Field(
        default=[],
        description="Most expensive individual requests"
    )
    
    # Optimization opportunities
    potential_additional_savings: float = Field(
        description="Potential additional savings"
    )
    optimization_recommendations: List[str] = Field(
        default=[],
        description="Cost optimization recommendations"
    )
    
    # Quality vs cost analysis
    quality_cost_correlation: Optional[float] = Field(
        default=None,
        description="Correlation between quality and cost"
    )
    
    # Budget tracking
    budget_used: Optional[float] = Field(
        default=None,
        description="Percentage of budget used"
    )
    projected_monthly_cost: Optional[float] = Field(
        default=None,
        description="Projected monthly cost"
    )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class PerformanceMetrics(BaseModel):
    """Performance metrics and SLA tracking."""
    
    # Time period
    period_start: datetime = Field(description="Metrics period start")
    period_end: datetime = Field(description="Metrics period end")
    
    # Latency metrics
    avg_total_latency: float = Field(description="Average total latency")
    avg_routing_latency: float = Field(description="Average routing latency")
    avg_provider_latency: float = Field(description="Average provider latency")
    
    # Latency percentiles
    latency_p50: float = Field(description="50th percentile latency")
    latency_p90: float = Field(description="90th percentile latency")
    latency_p95: float = Field(description="95th percentile latency")
    latency_p99: float = Field(description="99th percentile latency")
    
    # Throughput
    requests_per_second: float = Field(description="Average requests per second")
    peak_requests_per_second: float = Field(description="Peak requests per second")
    tokens_per_second: float = Field(description="Average tokens per second")
    
    # Reliability
    uptime_percentage: float = Field(ge=0.0, le=100.0, description="Uptime percentage")
    success_rate: float = Field(ge=0.0, le=1.0, description="Success rate")
    error_rate: float = Field(ge=0.0, le=1.0, description="Error rate")
    
    # SLA compliance
    sla_target_latency: Optional[float] = Field(default=None, description="SLA latency target")
    sla_compliance_rate: Optional[float] = Field(
        default=None,
        description="SLA compliance rate"
    )
    sla_violations: int = Field(default=0, description="Number of SLA violations")
    
    # Resource utilization
    cpu_utilization: Optional[float] = Field(default=None, description="CPU utilization")
    memory_utilization: Optional[float] = Field(default=None, description="Memory utilization")
    
    # Provider performance
    provider_latency: Dict[str, float] = Field(
        default={},
        description="Average latency by provider"
    )
    provider_error_rates: Dict[str, float] = Field(
        default={},
        description="Error rates by provider"
    )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class AlertRule(BaseModel):
    """Alert rule configuration."""
    
    id: str = Field(description="Alert rule ID")
    name: str = Field(description="Alert rule name")
    description: Optional[str] = Field(default=None, description="Alert description")
    
    # Alert conditions
    metric: str = Field(description="Metric to monitor")
    threshold: float = Field(description="Alert threshold")
    operator: str = Field(description="Comparison operator (>, <, =, etc.)")
    
    # Alert settings
    severity: str = Field(description="Alert severity (low, medium, high, critical)")
    enabled: bool = Field(default=True, description="Whether alert is enabled")
    
    # Notification settings
    notification_channels: List[str] = Field(
        default=[],
        description="Notification channels"
    )
    
    # Time settings
    evaluation_interval: int = Field(
        default=60,
        description="Evaluation interval in seconds"
    )
    for_duration: int = Field(
        default=300,
        description="Alert must be true for this duration"
    )
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(default=None, description="Creator user ID")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
