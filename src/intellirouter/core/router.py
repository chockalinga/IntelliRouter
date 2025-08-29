"""Main model router for intelligent AI request routing."""

import time
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime

from ..models.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatChoice,
    Usage,
    RouterInfo,
    StreamingChunk,
)
from ..models.providers import ModelRegistry, ModelMetadata, TaskType
from ..models.routing import (
    PromptClassification,
    RoutingConstraints,
    RoutingPolicy,
    RoutingDecision,
    RoutingResult,
    ModelScore,
    CostPriority,
)
from ..models.analytics import RequestLog, RequestStatus
from .classifier import PromptClassifier
from .policy_engine import PolicyEngine
from .fallback_manager import FallbackManager
from ..providers.factory import ProviderFactory
from ..config import settings


class ModelRouter:
    """Main router for intelligent model selection and request routing."""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        policy: Optional[RoutingPolicy] = None
    ):
        """Initialize the model router.
        
        Args:
            model_registry: Registry of available models
            policy: Routing policy configuration
        """
        self.model_registry = model_registry
        self.policy = policy or RoutingPolicy()
        
        # Core components
        self.classifier = PromptClassifier()
        self.policy_engine = PolicyEngine(model_registry, self.policy)
        self.fallback_manager = FallbackManager(model_registry)
        self.provider_factory = ProviderFactory()
        
        # Session tracking for sticky routing
        self.session_models: Dict[str, str] = {}
        
        # Performance tracking
        self.request_count = 0
        self.total_cost = 0.0
        self.total_savings = 0.0
    
    async def route_request(
        self,
        request: ChatCompletionRequest,
        constraints: Optional[RoutingConstraints] = None,
        session_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Route a chat completion request to the optimal model.
        
        Args:
            request: The chat completion request
            constraints: Optional routing constraints
            session_id: Optional session ID for sticky routing
            
        Returns:
            Chat completion response with routing information
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Step 1: Classify the prompt
            classification = self.classifier.classify(request.messages)
            
            # Step 2: Apply constraints
            effective_constraints = self._merge_constraints(
                constraints, 
                request, 
                classification
            )
            
            # Step 3: Make routing decision
            routing_decision = await self._make_routing_decision(
                classification,
                effective_constraints,
                session_id
            )
            
            # Step 4: Execute the request
            response = await self._execute_request(
                request,
                routing_decision,
                request_id,
                start_time
            )
            
            # Step 5: Log analytics
            await self._log_request(
                request_id,
                request,
                routing_decision,
                response,
                RequestStatus.SUCCESS
            )
            
            return response
            
        except Exception as e:
            # Handle errors with fallback
            return await self._handle_error(
                request,
                request_id,
                start_time,
                str(e),
                session_id
            )
    
    async def route_streaming_request(
        self,
        request: ChatCompletionRequest,
        constraints: Optional[RoutingConstraints] = None,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Route a streaming chat completion request.
        
        Args:
            request: The chat completion request
            constraints: Optional routing constraints
            session_id: Optional session ID for sticky routing
            
        Yields:
            Server-sent events for streaming response
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Classification and routing (same as non-streaming)
            classification = self.classifier.classify(request.messages)
            effective_constraints = self._merge_constraints(
                constraints, 
                request, 
                classification
            )
            
            routing_decision = await self._make_routing_decision(
                classification,
                effective_constraints,
                session_id
            )
            
            # Execute streaming request
            async for chunk in self._execute_streaming_request(
                request,
                routing_decision,
                request_id,
                start_time
            ):
                yield chunk
                
        except Exception as e:
            # Send error as streaming chunk
            error_chunk = self._create_error_chunk(request_id, str(e))
            yield f"data: {error_chunk}\n\n"
    
    def _merge_constraints(
        self,
        user_constraints: Optional[RoutingConstraints],
        request: ChatCompletionRequest,
        classification: PromptClassification
    ) -> RoutingConstraints:
        """Merge user constraints with inferred constraints."""
        
        # Start with default constraints
        constraints = RoutingConstraints()
        
        # Apply user constraints if provided
        if user_constraints:
            constraints = user_constraints
        
        # Infer constraints from request
        required_capabilities = []
        
        if request.tools:
            required_capabilities.append("tools")
        
        if request.response_format and request.response_format.type == "json_object":
            required_capabilities.append("json_mode")
        
        if classification.requires_vision:
            required_capabilities.append("vision")
        
        # Update constraints with inferred capabilities
        constraints.required_capabilities.extend(required_capabilities)
        
        # Set quality threshold from classification
        if constraints.min_quality_score == 0.0:
            constraints.min_quality_score = classification.quality_threshold
        
        # Set latency constraints based on classification
        if not constraints.max_latency_ms:
            latency_map = {
                "real_time": 1000,
                "fast": 3000,
                "normal": 10000,
                "slow": 30000
            }
            constraints.max_latency_ms = latency_map.get(
                classification.latency_requirement,
                10000
            )
        
        return constraints
    
    async def _make_routing_decision(
        self,
        classification: PromptClassification,
        constraints: RoutingConstraints,
        session_id: Optional[str] = None
    ) -> RoutingDecision:
        """Make the routing decision using the policy engine."""
        
        start_time = time.time()
        
        # Check for sticky routing
        if session_id and self.policy.enable_session_routing:
            if session_id in self.session_models:
                sticky_model_id = self.session_models[session_id]
                if sticky_model_id in self.model_registry.models:
                    sticky_model = self.model_registry.models[sticky_model_id]
                    # Verify the sticky model still meets constraints
                    if self._model_meets_constraints(sticky_model, constraints):
                        return self._create_sticky_decision(
                            sticky_model,
                            classification,
                            constraints,
                            int((time.time() - start_time) * 1000)
                        )
        
        # Use policy engine for decision
        decision = await self.policy_engine.make_decision(
            classification,
            constraints,
            session_id
        )
        
        # Update session tracking
        if session_id and self.policy.enable_session_routing:
            self.session_models[session_id] = decision.selected_model.id
        
        return decision
    
    async def _execute_request(
        self,
        request: ChatCompletionRequest,
        decision: RoutingDecision,
        request_id: str,
        start_time: float
    ) -> ChatCompletionResponse:
        """Execute the request using the selected model."""
        
        provider = self.provider_factory.get_provider(
            decision.selected_model.provider
        )
        
        try:
            # Execute the request
            response = await provider.complete_chat(
                request,
                decision.selected_model,
                request_id
            )
            
            # Calculate actual metrics
            actual_latency = int((time.time() - start_time) * 1000)
            
            # Add router information
            router_info = RouterInfo(
                selected_model=decision.selected_model.id,
                reason=decision.selected_model_score.reasoning,
                latency_ms=actual_latency,
                cost=response.usage.total_tokens * decision.selected_model.input_price_per_1k / 1000,
                savings_vs_baseline=decision.estimated_savings,
                baseline_model=decision.baseline_model,
                classification=decision.classification.dict(),
                fallback_used=False
            )
            
            response.router_info = router_info
            
            # Update tracking
            self.request_count += 1
            self.total_cost += router_info.cost
            if router_info.savings_vs_baseline:
                self.total_savings += router_info.savings_vs_baseline
            
            return response
            
        except Exception as e:
            # Try fallback if enabled
            if self.policy.enable_fallback:
                return await self._try_fallback(
                    request,
                    decision,
                    request_id,
                    start_time,
                    str(e)
                )
            else:
                raise
    
    async def _execute_streaming_request(
        self,
        request: ChatCompletionRequest,
        decision: RoutingDecision,
        request_id: str,
        start_time: float
    ) -> AsyncGenerator[str, None]:
        """Execute a streaming request."""
        
        provider = self.provider_factory.get_provider(
            decision.selected_model.provider
        )
        
        try:
            # Send initial router info
            router_info = RouterInfo(
                selected_model=decision.selected_model.id,
                reason=decision.selected_model_score.reasoning,
                latency_ms=0,  # Will be updated
                cost=0.0,  # Will be calculated
                savings_vs_baseline=decision.estimated_savings,
                baseline_model=decision.baseline_model,
                classification=decision.classification.dict(),
                fallback_used=False
            )
            
            yield f"data: {{'router_info': {router_info.json()}}}\n\n"
            
            # Stream the response
            async for chunk in provider.stream_chat(
                request,
                decision.selected_model,
                request_id
            ):
                yield f"data: {chunk}\n\n"
                
        except Exception as e:
            if self.policy.enable_fallback:
                async for chunk in self._try_streaming_fallback(
                    request,
                    decision,
                    request_id,
                    start_time,
                    str(e)
                ):
                    yield chunk
            else:
                error_chunk = self._create_error_chunk(request_id, str(e))
                yield f"data: {error_chunk}\n\n"
    
    async def _try_fallback(
        self,
        request: ChatCompletionRequest,
        original_decision: RoutingDecision,
        request_id: str,
        start_time: float,
        error: str
    ) -> ChatCompletionResponse:
        """Try fallback models when primary model fails."""
        
        fallback_model = await self.fallback_manager.get_fallback_model(
            original_decision.selected_model,
            original_decision.constraints
        )
        
        if not fallback_model:
            raise Exception(f"No fallback available. Original error: {error}")
        
        provider = self.provider_factory.get_provider(fallback_model.provider)
        
        try:
            response = await provider.complete_chat(
                request,
                fallback_model,
                request_id
            )
            
            # Mark as fallback response
            actual_latency = int((time.time() - start_time) * 1000)
            
            router_info = RouterInfo(
                selected_model=fallback_model.id,
                reason=f"Fallback used due to error: {error}",
                latency_ms=actual_latency,
                cost=response.usage.total_tokens * fallback_model.input_price_per_1k / 1000,
                fallback_used=True
            )
            
            response.router_info = router_info
            return response
            
        except Exception as fallback_error:
            raise Exception(f"Fallback failed: {fallback_error}. Original error: {error}")
    
    async def _try_streaming_fallback(
        self,
        request: ChatCompletionRequest,
        original_decision: RoutingDecision,
        request_id: str,
        start_time: float,
        error: str
    ) -> AsyncGenerator[str, None]:
        """Try fallback for streaming requests."""
        
        fallback_model = await self.fallback_manager.get_fallback_model(
            original_decision.selected_model,
            original_decision.constraints
        )
        
        if not fallback_model:
            error_chunk = self._create_error_chunk(
                request_id, 
                f"No fallback available. Original error: {error}"
            )
            yield f"data: {error_chunk}\n\n"
            return
        
        provider = self.provider_factory.get_provider(fallback_model.provider)
        
        try:
            # Send fallback notification
            fallback_info = RouterInfo(
                selected_model=fallback_model.id,
                reason=f"Fallback used due to error: {error}",
                latency_ms=0,
                cost=0.0,
                fallback_used=True
            )
            
            yield f"data: {{'fallback_info': {fallback_info.json()}}}\n\n"
            
            # Stream fallback response
            async for chunk in provider.stream_chat(
                request,
                fallback_model,
                request_id
            ):
                yield f"data: {chunk}\n\n"
                
        except Exception as fallback_error:
            error_chunk = self._create_error_chunk(
                request_id,
                f"Fallback failed: {fallback_error}. Original error: {error}"
            )
            yield f"data: {error_chunk}\n\n"
    
    async def _handle_error(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        start_time: float,
        error: str,
        session_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Handle routing errors."""
        
        # Log the error
        await self._log_request(
            request_id,
            request,
            None,
            None,
            RequestStatus.ERROR,
            error
        )
        
        # Create error response
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model="error",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=f"I encountered an error processing your request: {error}"
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            ),
            router_info=RouterInfo(
                selected_model="error",
                reason=f"Error occurred: {error}",
                latency_ms=int((time.time() - start_time) * 1000),
                cost=0.0,
                fallback_used=False
            )
        )
    
    def _model_meets_constraints(
        self,
        model: ModelMetadata,
        constraints: RoutingConstraints
    ) -> bool:
        """Check if a model meets the given constraints."""
        
        # Cost constraints
        if constraints.max_input_cost_per_1k:
            if model.input_price_per_1k > constraints.max_input_cost_per_1k:
                return False
        
        if constraints.max_output_cost_per_1k:
            if model.output_price_per_1k > constraints.max_output_cost_per_1k:
                return False
        
        # Latency constraints
        if constraints.max_latency_ms:
            if model.avg_latency_ms > constraints.max_latency_ms:
                return False
        
        # Capability constraints
        for capability in constraints.required_capabilities:
            if not self.model_registry._has_capability(model, capability):
                return False
        
        # Quality constraints
        if model.reliability_score < constraints.min_reliability_score:
            return False
        
        return True
    
    def _create_sticky_decision(
        self,
        model: ModelMetadata,
        classification: PromptClassification,
        constraints: RoutingConstraints,
        decision_time_ms: int
    ) -> RoutingDecision:
        """Create a routing decision for sticky routing."""
        
        model_score = ModelScore(
            model_id=model.id,
            total_score=1.0,
            quality_score=1.0,
            cost_score=1.0,
            latency_score=1.0,
            reliability_score=model.reliability_score,
            estimated_input_cost=0.0,
            estimated_output_cost=0.0,
            estimated_total_cost=0.0,
            meets_constraints=True,
            reasoning="Sticky routing - continuing with same model from session"
        )
        
        return RoutingDecision(
            selected_model=model,
            selected_model_score=model_score,
            classification=classification,
            constraints=constraints,
            policy=self.policy,
            decision_time_ms=decision_time_ms,
            decision_confidence=1.0,
            routing_strategy="sticky_session",
            fallback_available=len(self.policy.fallback_models) > 0
        )
    
    def _create_error_chunk(self, request_id: str, error: str) -> str:
        """Create an error chunk for streaming responses."""
        
        error_chunk = StreamingChunk(
            id=request_id,
            created=int(time.time()),
            model="error",
            choices=[{
                "index": 0,
                "delta": {"content": f"Error: {error}"},
                "finish_reason": "stop"
            }]
        )
        
        return error_chunk.json()
    
    async def _log_request(
        self,
        request_id: str,
        request: ChatCompletionRequest,
        decision: Optional[RoutingDecision],
        response: Optional[ChatCompletionResponse],
        status: RequestStatus,
        error: Optional[str] = None
    ):
        """Log request for analytics."""
        
        # This would integrate with the analytics system
        # For now, we'll just track basic metrics
        
        if status == RequestStatus.SUCCESS and response:
            self.request_count += 1
            if response.router_info:
                self.total_cost += response.router_info.cost
                if response.router_info.savings_vs_baseline:
                    self.total_savings += response.router_info.savings_vs_baseline
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        
        return {
            "request_count": self.request_count,
            "total_cost": self.total_cost,
            "total_savings": self.total_savings,
            "average_cost_per_request": self.total_cost / max(self.request_count, 1),
            "savings_percentage": (self.total_savings / max(self.total_cost + self.total_savings, 1)) * 100,
            "active_models": len([m for m in self.model_registry.models.values() if m.is_active]),
            "active_sessions": len(self.session_models)
        }
