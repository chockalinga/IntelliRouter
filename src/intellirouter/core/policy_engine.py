"""Policy engine for intelligent model selection."""

import time
from typing import List, Optional, Dict, Any
import numpy as np

from ..models.providers import ModelRegistry, ModelMetadata, TaskType
from ..models.routing import (
    PromptClassification,
    RoutingConstraints,
    RoutingPolicy,
    RoutingDecision,
    ModelScore,
    CostPriority,
)
from ..config import settings


class PolicyEngine:
    """Engine for making intelligent routing decisions based on policy."""
    
    def __init__(self, model_registry: ModelRegistry, policy: RoutingPolicy):
        """Initialize the policy engine.
        
        Args:
            model_registry: Registry of available models
            policy: Routing policy configuration
        """
        self.model_registry = model_registry
        self.policy = policy
        
        # Validate policy weights
        if not policy.validate_weights():
            raise ValueError("Policy weights must sum to 1.0")
    
    async def make_decision(
        self,
        classification: PromptClassification,
        constraints: RoutingConstraints,
        session_id: Optional[str] = None
    ) -> RoutingDecision:
        """Make a routing decision based on classification and constraints.
        
        Args:
            classification: Prompt classification results
            constraints: Routing constraints
            session_id: Optional session ID
            
        Returns:
            Routing decision with selected model and reasoning
        """
        start_time = time.time()
        
        # Get candidate models that meet constraints
        candidates = self._get_candidate_models(classification, constraints)
        
        if not candidates:
            raise Exception("No models available that meet the specified constraints")
        
        # Score all candidate models
        model_scores = []
        for model in candidates:
            score = self._score_model(model, classification, constraints)
            model_scores.append(score)
        
        # Sort by total score (highest first)
        model_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Select the best model
        selected_score = model_scores[0]
        selected_model = next(
            m for m in candidates if m.id == selected_score.model_id
        )
        
        # Calculate cost savings
        baseline_model_id = settings.default_model
        baseline_cost, estimated_savings, savings_percentage = self._calculate_savings(
            selected_model,
            baseline_model_id,
            classification.estimated_tokens
        )
        
        # Determine decision confidence
        confidence = self._calculate_decision_confidence(model_scores)
        
        decision_time = int((time.time() - start_time) * 1000)
        
        return RoutingDecision(
            selected_model=selected_model,
            selected_model_score=selected_score,
            classification=classification,
            constraints=constraints,
            policy=self.policy,
            candidate_models=model_scores[:5],  # Top 5 alternatives
            decision_time_ms=decision_time,
            decision_confidence=confidence,
            baseline_model=baseline_model_id,
            baseline_cost=baseline_cost,
            estimated_savings=estimated_savings,
            savings_percentage=savings_percentage,
            routing_strategy=self._get_routing_strategy(),
            fallback_available=len(self.policy.fallback_models) > 0,
            session_id=session_id,
            conversation_context=classification.is_conversational
        )
    
    def _get_candidate_models(
        self,
        classification: PromptClassification,
        constraints: RoutingConstraints
    ) -> List[ModelMetadata]:
        """Get models that meet the constraints and requirements."""
        
        candidates = []
        
        for model in self.model_registry.models.values():
            if not model.is_active:
                continue
            
            # Check basic constraints
            if not self._model_meets_constraints(model, constraints):
                continue
            
            # Check capability requirements
            if not self._model_has_required_capabilities(model, classification, constraints):
                continue
            
            # Check quality threshold
            task_quality = getattr(model.quality_scores, classification.task_type, 0.0)
            if task_quality < constraints.min_quality_score:
                continue
            
            candidates.append(model)
        
        return candidates
    
    def _model_meets_constraints(
        self,
        model: ModelMetadata,
        constraints: RoutingConstraints
    ) -> bool:
        """Check if model meets basic constraints."""
        
        # Cost constraints
        if constraints.max_input_cost_per_1k:
            if model.input_price_per_1k > constraints.max_input_cost_per_1k:
                return False
        
        if constraints.max_output_cost_per_1k:
            if model.output_price_per_1k > constraints.max_output_cost_per_1k:
                return False
        
        # Performance constraints
        if constraints.max_latency_ms:
            if model.avg_latency_ms > constraints.max_latency_ms:
                return False
        
        if model.reliability_score < constraints.min_reliability_score:
            return False
        
        # Geographic constraints
        if constraints.allowed_regions:
            if not any(region in model.regions for region in constraints.allowed_regions):
                return False
        
        # Provider constraints
        if constraints.excluded_providers:
            if model.provider.value in constraints.excluded_providers:
                return False
        
        # Context window constraint
        if constraints.max_context_window:
            if model.context_window < constraints.max_context_window:
                return False
        
        return True
    
    def _model_has_required_capabilities(
        self,
        model: ModelMetadata,
        classification: PromptClassification,
        constraints: RoutingConstraints
    ) -> bool:
        """Check if model has required capabilities."""
        
        capabilities = model.capabilities
        
        # Check explicit capability requirements
        for capability in constraints.required_capabilities:
            if capability == "tools" and not capabilities.supports_tools:
                return False
            elif capability == "streaming" and not capabilities.supports_streaming:
                return False
            elif capability == "json_mode" and not capabilities.supports_json_mode:
                return False
            elif capability == "vision" and not capabilities.supports_vision:
                return False
        
        # Check inferred requirements from classification
        if classification.requires_tools and not capabilities.supports_tools:
            return False
        
        if classification.requires_json_mode and not capabilities.supports_json_mode:
            return False
        
        if classification.requires_vision and not capabilities.supports_vision:
            return False
        
        return True
    
    def _score_model(
        self,
        model: ModelMetadata,
        classification: PromptClassification,
        constraints: RoutingConstraints
    ) -> ModelScore:
        """Score a model based on policy weights and requirements."""
        
        # Get quality score for the specific task
        task_quality = getattr(model.quality_scores, classification.task_type, 0.0)
        
        # Calculate individual scores (0-1 scale)
        quality_score = task_quality
        cost_score = self._calculate_cost_score(model, classification.estimated_tokens)
        latency_score = self._calculate_latency_score(model, classification.latency_requirement)
        reliability_score = model.reliability_score
        
        # Apply policy weights
        total_score = (
            quality_score * self.policy.quality_weight +
            cost_score * self.policy.cost_weight +
            latency_score * self.policy.latency_weight +
            reliability_score * self.policy.reliability_weight
        )
        
        # Apply cost priority adjustments
        if self.policy.cost_priority == CostPriority.MINIMIZE_COST:
            total_score = total_score * 0.7 + cost_score * 0.3
        elif self.policy.cost_priority == CostPriority.MAXIMIZE_QUALITY:
            total_score = total_score * 0.7 + quality_score * 0.3
        
        # Calculate cost estimates
        estimated_input_cost = (
            classification.estimated_tokens * model.input_price_per_1k / 1000
        )
        estimated_output_cost = (
            classification.estimated_tokens * 0.3 * model.output_price_per_1k / 1000
        )  # Assume output is ~30% of input
        estimated_total_cost = estimated_input_cost + estimated_output_cost
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            model, quality_score, cost_score, latency_score, total_score
        )
        
        return ModelScore(
            model_id=model.id,
            total_score=total_score,
            quality_score=quality_score,
            cost_score=cost_score,
            latency_score=latency_score,
            reliability_score=reliability_score,
            estimated_input_cost=estimated_input_cost,
            estimated_output_cost=estimated_output_cost,
            estimated_total_cost=estimated_total_cost,
            meets_constraints=True,  # Already filtered
            reasoning=reasoning
        )
    
    def _calculate_cost_score(self, model: ModelMetadata, estimated_tokens: int) -> float:
        """Calculate cost score (higher is better, meaning lower cost)."""
        
        # Get the maximum cost among all models for normalization
        max_cost = max(
            m.input_price_per_1k + m.output_price_per_1k
            for m in self.model_registry.models.values()
            if m.is_active
        )
        
        model_cost = model.input_price_per_1k + model.output_price_per_1k
        
        # Invert cost (higher cost = lower score)
        if max_cost > 0:
            return 1.0 - (model_cost / max_cost)
        return 1.0
    
    def _calculate_latency_score(self, model: ModelMetadata, latency_req) -> float:
        """Calculate latency score (higher is better, meaning lower latency)."""
        
        # Get the maximum latency among all models for normalization
        max_latency = max(
            m.avg_latency_ms
            for m in self.model_registry.models.values()
            if m.is_active
        )
        
        # Invert latency (higher latency = lower score)
        if max_latency > 0:
            return 1.0 - (model.avg_latency_ms / max_latency)
        return 1.0
    
    def _calculate_savings(
        self,
        selected_model: ModelMetadata,
        baseline_model_id: str,
        estimated_tokens: int
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate cost savings compared to baseline model."""
        
        if baseline_model_id not in self.model_registry.models:
            return None, None, None
        
        baseline_model = self.model_registry.models[baseline_model_id]
        
        # Calculate costs
        selected_cost = (
            estimated_tokens * selected_model.input_price_per_1k / 1000 +
            estimated_tokens * 0.3 * selected_model.output_price_per_1k / 1000
        )
        
        baseline_cost = (
            estimated_tokens * baseline_model.input_price_per_1k / 1000 +
            estimated_tokens * 0.3 * baseline_model.output_price_per_1k / 1000
        )
        
        savings = baseline_cost - selected_cost
        savings_percentage = (savings / baseline_cost * 100) if baseline_cost > 0 else 0
        
        return baseline_cost, savings, savings_percentage
    
    def _calculate_decision_confidence(self, model_scores: List[ModelScore]) -> float:
        """Calculate confidence in the routing decision."""
        
        if len(model_scores) < 2:
            return 1.0
        
        # Calculate confidence based on score difference
        top_score = model_scores[0].total_score
        second_score = model_scores[1].total_score
        
        score_gap = top_score - second_score
        
        # Higher gap = higher confidence
        confidence = min(1.0, 0.5 + score_gap * 2)
        
        return confidence
    
    def _generate_reasoning(
        self,
        model: ModelMetadata,
        quality_score: float,
        cost_score: float,
        latency_score: float,
        total_score: float
    ) -> str:
        """Generate human-readable reasoning for model selection."""
        
        reasons = []
        
        # Identify strongest factors
        scores = {
            "quality": quality_score,
            "cost": cost_score,
            "latency": latency_score
        }
        
        best_factor = max(scores, key=scores.get)
        
        if best_factor == "quality" and quality_score > 0.8:
            reasons.append(f"High quality score ({quality_score:.2f}) for this task type")
        
        if best_factor == "cost" and cost_score > 0.7:
            reasons.append("Excellent cost efficiency")
        
        if best_factor == "latency" and latency_score > 0.8:
            reasons.append("Fast response time")
        
        if model.reliability_score > 0.95:
            reasons.append("High reliability")
        
        # Combine reasons
        if reasons:
            return f"Selected for: {', '.join(reasons)}"
        else:
            return f"Best overall balance of quality, cost, and performance (score: {total_score:.2f})"
    
    def _get_routing_strategy(self) -> str:
        """Get the routing strategy description."""
        
        if self.policy.cost_priority == CostPriority.MINIMIZE_COST:
            return "cost_optimized"
        elif self.policy.cost_priority == CostPriority.MAXIMIZE_QUALITY:
            return "quality_optimized"
        else:
            return "balanced"
