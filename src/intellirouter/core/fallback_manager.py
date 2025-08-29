"""Fallback manager for handling model failures and errors."""

from typing import List, Optional, Dict, Any
import asyncio
import time

from ..models.providers import ModelRegistry, ModelMetadata, ProviderType
from ..models.routing import RoutingConstraints


class FallbackManager:
    """Manages fallback strategies when models fail."""
    
    def __init__(self, model_registry: ModelRegistry):
        """Initialize the fallback manager.
        
        Args:
            model_registry: Registry of available models
        """
        self.model_registry = model_registry
        
        # Track model health and failures
        self.model_health: Dict[str, Dict[str, Any]] = {}
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_time: Dict[str, float] = {}
        
        # Circuit breaker settings
        self.failure_threshold = 5  # Failures before circuit opens
        self.recovery_timeout = 300  # 5 minutes before retry
        
        # Default fallback order by reliability and cost
        self.default_fallback_order = [
            "gpt-3.5-turbo",
            "gpt-4o-mini", 
            "claude-3-haiku",
            "gpt-4o",
            "claude-3-sonnet"
        ]
    
    async def get_fallback_model(
        self,
        failed_model: ModelMetadata,
        constraints: RoutingConstraints,
        exclude_providers: Optional[List[ProviderType]] = None
    ) -> Optional[ModelMetadata]:
        """Get the best fallback model for a failed model.
        
        Args:
            failed_model: The model that failed
            constraints: Original routing constraints
            exclude_providers: Providers to exclude from fallback
            
        Returns:
            Best fallback model or None if no suitable fallback
        """
        
        # Record the failure
        self._record_failure(failed_model.id)
        
        # Get candidate fallback models
        candidates = self._get_fallback_candidates(
            failed_model,
            constraints,
            exclude_providers
        )
        
        if not candidates:
            return None
        
        # Sort by fallback priority
        candidates = self._sort_by_fallback_priority(candidates, failed_model)
        
        # Return the best available fallback
        return candidates[0] if candidates else None
    
    def _record_failure(self, model_id: str):
        """Record a model failure for circuit breaker logic."""
        
        current_time = time.time()
        
        # Increment failure count
        self.failure_counts[model_id] = self.failure_counts.get(model_id, 0) + 1
        self.last_failure_time[model_id] = current_time
        
        # Update health status
        if model_id not in self.model_health:
            self.model_health[model_id] = {
                "status": "healthy",
                "failure_count": 0,
                "circuit_open": False,
                "last_failure": None
            }
        
        health = self.model_health[model_id]
        health["failure_count"] = self.failure_counts[model_id]
        health["last_failure"] = current_time
        
        # Open circuit if failure threshold exceeded
        if health["failure_count"] >= self.failure_threshold:
            health["status"] = "unhealthy"
            health["circuit_open"] = True
    
    def _get_fallback_candidates(
        self,
        failed_model: ModelMetadata,
        constraints: RoutingConstraints,
        exclude_providers: Optional[List[ProviderType]] = None
    ) -> List[ModelMetadata]:
        """Get candidate models for fallback."""
        
        candidates = []
        exclude_providers = exclude_providers or []
        
        for model in self.model_registry.models.values():
            # Skip if not active
            if not model.is_active:
                continue
            
            # Skip the failed model
            if model.id == failed_model.id:
                continue
            
            # Skip excluded providers
            if model.provider in exclude_providers:
                continue
            
            # Skip if provider same as failed model (might be provider issue)
            if model.provider == failed_model.provider:
                continue
            
            # Skip if circuit is open for this model
            if self._is_circuit_open(model.id):
                continue
            
            # Check if model meets basic constraints
            if not self._model_meets_fallback_constraints(model, constraints):
                continue
            
            candidates.append(model)
        
        return candidates
    
    def _model_meets_fallback_constraints(
        self,
        model: ModelMetadata,
        constraints: RoutingConstraints
    ) -> bool:
        """Check if model meets relaxed constraints for fallback."""
        
        # For fallback, we relax some constraints but keep critical ones
        
        # Always check capability requirements
        for capability in constraints.required_capabilities:
            if not self._has_capability(model, capability):
                return False
        
        # Relax cost constraints for fallback (allow 2x the original limit)
        if constraints.max_input_cost_per_1k:
            if model.input_price_per_1k > constraints.max_input_cost_per_1k * 2:
                return False
        
        if constraints.max_output_cost_per_1k:
            if model.output_price_per_1k > constraints.max_output_cost_per_1k * 2:
                return False
        
        # Relax latency constraints (allow 1.5x the original limit)
        if constraints.max_latency_ms:
            if model.avg_latency_ms > constraints.max_latency_ms * 1.5:
                return False
        
        # Keep quality constraints but lower threshold
        min_quality = max(0.5, constraints.min_quality_score - 0.2)
        if model.reliability_score < min_quality:
            return False
        
        # Keep geographic and compliance constraints strict
        if constraints.allowed_regions:
            if not any(region in model.regions for region in constraints.allowed_regions):
                return False
        
        if constraints.excluded_providers:
            if model.provider in constraints.excluded_providers:
                return False
        
        return True
    
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
    
    def _sort_by_fallback_priority(
        self,
        candidates: List[ModelMetadata],
        failed_model: ModelMetadata
    ) -> List[ModelMetadata]:
        """Sort candidates by fallback priority."""
        
        def fallback_score(model: ModelMetadata) -> tuple:
            """Calculate fallback priority score."""
            
            # Priority factors (lower is better for sorting):
            # 1. Reliability (higher is better)
            # 2. Different provider (reduce correlated failures)
            # 3. Cost (lower is better for fallback)
            # 4. Historical failure rate
            # 5. Predefined fallback order
            
            reliability_score = -model.reliability_score  # Negative for ascending sort
            
            provider_penalty = 0 if model.provider != failed_model.provider else 1
            
            cost_score = model.input_price_per_1k + model.output_price_per_1k
            
            failure_rate = self._get_failure_rate(model.id)
            
            # Predefined order bonus
            order_bonus = 0
            if model.id in self.default_fallback_order:
                order_bonus = self.default_fallback_order.index(model.id)
            else:
                order_bonus = len(self.default_fallback_order)
            
            return (
                reliability_score,
                provider_penalty,
                failure_rate,
                cost_score,
                order_bonus
            )
        
        candidates.sort(key=fallback_score)
        return candidates
    
    def _is_circuit_open(self, model_id: str) -> bool:
        """Check if circuit breaker is open for a model."""
        
        if model_id not in self.model_health:
            return False
        
        health = self.model_health[model_id]
        
        if not health.get("circuit_open", False):
            return False
        
        # Check if recovery timeout has passed
        last_failure = health.get("last_failure", 0)
        if time.time() - last_failure > self.recovery_timeout:
            # Reset circuit breaker
            health["circuit_open"] = False
            health["status"] = "recovering"
            health["failure_count"] = 0
            return False
        
        return True
    
    def _get_failure_rate(self, model_id: str) -> float:
        """Get recent failure rate for a model."""
        
        if model_id not in self.model_health:
            return 0.0
        
        health = self.model_health[model_id]
        failure_count = health.get("failure_count", 0)
        
        # Simple failure rate based on recent failures
        # In a production system, this would use a sliding window
        return min(1.0, failure_count / 10.0)
    
    def record_success(self, model_id: str):
        """Record a successful request to help with circuit breaker recovery."""
        
        if model_id in self.model_health:
            health = self.model_health[model_id]
            
            # Reduce failure count on success
            health["failure_count"] = max(0, health["failure_count"] - 1)
            
            # If failure count is low, mark as healthy
            if health["failure_count"] <= 1:
                health["status"] = "healthy"
                health["circuit_open"] = False
    
    def get_model_health_status(self, model_id: str) -> Dict[str, Any]:
        """Get health status for a model."""
        
        if model_id not in self.model_health:
            return {
                "status": "healthy",
                "failure_count": 0,
                "circuit_open": False,
                "last_failure": None
            }
        
        return self.model_health[model_id].copy()
    
    def get_all_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all models."""
        
        status = {}
        
        for model_id in self.model_registry.models:
            status[model_id] = self.get_model_health_status(model_id)
        
        return status
    
    def reset_model_health(self, model_id: str):
        """Reset health status for a model (admin function)."""
        
        if model_id in self.model_health:
            self.model_health[model_id] = {
                "status": "healthy",
                "failure_count": 0,
                "circuit_open": False,
                "last_failure": None
            }
        
        if model_id in self.failure_counts:
            del self.failure_counts[model_id]
        
        if model_id in self.last_failure_time:
            del self.last_failure_time[model_id]
    
    def configure_fallback_order(self, model_ids: List[str]):
        """Configure custom fallback order."""
        
        # Validate that all model IDs exist
        valid_models = []
        for model_id in model_ids:
            if model_id in self.model_registry.models:
                valid_models.append(model_id)
        
        self.default_fallback_order = valid_models
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get fallback usage statistics."""
        
        total_failures = sum(self.failure_counts.values())
        unhealthy_models = len([
            h for h in self.model_health.values()
            if h.get("status") != "healthy"
        ])
        
        open_circuits = len([
            h for h in self.model_health.values()
            if h.get("circuit_open", False)
        ])
        
        return {
            "total_failures_recorded": total_failures,
            "unhealthy_models": unhealthy_models,
            "open_circuits": open_circuits,
            "models_with_failures": len(self.failure_counts),
            "fallback_order": self.default_fallback_order,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout
        }
