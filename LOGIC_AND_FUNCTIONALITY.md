# 🧠 IntelliRouter Logic & Functionality Guide

## **🔄 Complete Request Flow**

This document explains exactly how IntelliRouter processes every request and makes intelligent routing decisions through its sophisticated decision engine.

### **Step 1: Request Analysis (PromptClassifier)**

When a request comes in, the **PromptClassifier** analyzes it using multiple heuristics:

```python
# Example: "Write a Python function to sort a list"

1. **Task Type Detection**: Scans for keywords
   - Found: ["code", "function", "python"] 
   - Result: TaskType.CODING

2. **Complexity Assessment**: Multiple factors
   - Keywords: "function" (+1), "sort" (+1)
   - Length: 42 characters (-1, short)  
   - Result: ComplexityLevel.MEDIUM

3. **Token Estimation**: ~10 tokens
4. **Capability Requirements**: 
   - requires_tools: False
   - requires_json_mode: False
   - requires_vision: False

5. **Content Analysis**:
   - has_code: True (function mention)
   - has_math: False
   - is_conversational: False

6. **Inferred Requirements**:
   - latency_requirement: NORMAL
   - quality_threshold: 0.85 (coding needs accuracy)
   - confidence: 0.8 (clear coding keywords)
```

#### **Task Type Classification Logic**

```python
task_keywords = {
    TaskType.CODING: [
        "code", "function", "class", "method", "algorithm", "debug", "bug",
        "python", "javascript", "java", "c++", "sql", "html", "css", "react"
    ],
    TaskType.REASONING: [
        "analyze", "explain why", "reason", "logic", "because", "therefore",
        "conclude", "infer", "deduce", "prove", "argument", "evidence"
    ],
    TaskType.CREATIVE_WRITING: [
        "story", "poem", "creative", "imagine", "character", "plot", "write",
        "narrative", "fiction", "novel", "dialogue", "scene"
    ],
    # ... more task types
}

# Score each task type based on keyword matches
for task_type, keywords in task_keywords.items():
    score = sum(text.count(keyword) for keyword in keywords if keyword in text)
    task_scores[task_type] = score

# Return highest scoring task type
```

#### **Complexity Assessment Logic**

```python
def assess_complexity(text, messages):
    complexity_score = 0
    
    # High complexity indicators
    high_indicators = ["complex", "complicated", "advanced", "multi-step"]
    complexity_score += sum(2 for indicator in high_indicators if indicator in text)
    
    # Low complexity indicators  
    low_indicators = ["simple", "basic", "easy", "quick", "brief"]
    complexity_score -= sum(1 for indicator in low_indicators if indicator in text)
    
    # Length-based complexity
    token_count = estimate_tokens(text)
    if token_count > 2000: complexity_score += 3
    elif token_count > 1000: complexity_score += 2
    elif token_count < 50: complexity_score -= 2
    
    # Multi-part questions
    question_count = text.count("?")
    if question_count > 3: complexity_score += 2
    
    # Map to complexity level
    if complexity_score >= 5: return ComplexityLevel.VERY_HIGH
    elif complexity_score >= 3: return ComplexityLevel.HIGH
    elif complexity_score >= 1: return ComplexityLevel.MEDIUM
    elif complexity_score >= -1: return ComplexityLevel.LOW
    else: return ComplexityLevel.VERY_LOW
```

### **Step 2: Constraint Application (ModelRouter)**

The router merges user constraints with inferred ones:

```python
def merge_constraints(user_constraints, request, classification):
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
            classification.latency_requirement, 10000
        )
    
    return constraints
```

### **Step 3: Candidate Filtering (PolicyEngine)**

Filter available models that meet constraints:

```python
def get_candidate_models(classification, constraints):
    candidates = []
    
    for model in model_registry.models.values():
        if not model.is_active:
            continue
        
        # Check basic constraints
        if constraints.max_input_cost_per_1k:
            if model.input_price_per_1k > constraints.max_input_cost_per_1k:
                continue
        
        if constraints.max_latency_ms:
            if model.avg_latency_ms > constraints.max_latency_ms:
                continue
        
        # Check quality threshold
        task_quality = getattr(model.quality_scores, classification.task_type, 0.0)
        if task_quality < constraints.min_quality_score:
            continue
        
        # Check capability requirements
        if not model_has_required_capabilities(model, classification, constraints):
            continue
        
        candidates.append(model)
    
    return candidates

# Available models: [gpt-4o, gpt-4o-mini, claude-3-5-sonnet, claude-3-haiku]
# After filtering: [gpt-4o-mini, claude-3-haiku] (meet cost+quality+latency)
```

### **Step 4: Model Scoring (PolicyEngine)**

Score each candidate model using weighted factors:

```python
def score_model(model, classification, constraints, policy):
    # Get quality score for the specific task
    task_quality = getattr(model.quality_scores, classification.task_type, 0.0)
    
    # Calculate individual scores (0-1 scale)
    quality_score = task_quality
    cost_score = calculate_cost_score(model, classification.estimated_tokens)
    latency_score = calculate_latency_score(model, classification.latency_requirement)
    reliability_score = model.reliability_score
    
    # Apply policy weights
    total_score = (
        quality_score * policy.quality_weight +
        cost_score * policy.cost_weight +
        latency_score * policy.latency_weight +
        reliability_score * policy.reliability_weight
    )
    
    # Apply cost priority adjustments
    if policy.cost_priority == CostPriority.MINIMIZE_COST:
        total_score = total_score * 0.7 + cost_score * 0.3
    elif policy.cost_priority == CostPriority.MAXIMIZE_QUALITY:
        total_score = total_score * 0.7 + quality_score * 0.3
    
    return ModelScore(
        model_id=model.id,
        total_score=total_score,
        quality_score=quality_score,
        cost_score=cost_score,
        latency_score=latency_score,
        reliability_score=reliability_score,
        reasoning=generate_reasoning(model, quality_score, cost_score, latency_score)
    )

# Example scoring for GPT-4o-mini:
# quality_score = 0.85  # Good for coding
# cost_score = 0.90     # Very cost effective ($0.15/$0.60 per 1K)
# latency_score = 0.80  # Fast response (~800ms)
# reliability_score = 0.95  # Very reliable
# total_score = (0.85*0.30 + 0.90*0.40 + 0.80*0.20 + 0.95*0.10) = 0.875
```

#### **Cost Score Calculation**

```python
def calculate_cost_score(model, estimated_tokens):
    # Get the maximum cost among all models for normalization
    max_cost = max(
        m.input_price_per_1k + m.output_price_per_1k
        for m in model_registry.models.values()
        if m.is_active
    )
    
    model_cost = model.input_price_per_1k + model.output_price_per_1k
    
    # Invert cost (higher cost = lower score)
    if max_cost > 0:
        return 1.0 - (model_cost / max_cost)
    return 1.0

# Example:
# GPT-4o cost: $5.00 + $15.00 = $20.00 per 1K
# GPT-4o-mini cost: $0.15 + $0.60 = $0.75 per 1K
# Max cost: $20.00
# GPT-4o score: 1.0 - (20.00 / 20.00) = 0.0
# GPT-4o-mini score: 1.0 - (0.75 / 20.00) = 0.9625
```

#### **Latency Score Calculation**

```python
def calculate_latency_score(model, latency_requirement):
    # Get the maximum latency among all models for normalization
    max_latency = max(
        m.avg_latency_ms
        for m in model_registry.models.values()
        if m.is_active
    )
    
    # Invert latency (higher latency = lower score)
    if max_latency > 0:
        return 1.0 - (model.avg_latency_ms / max_latency)
    return 1.0

# Example:
# Claude-3-5-Sonnet: 3000ms
# Claude-3-Haiku: 600ms  
# Max latency: 3000ms
# Sonnet score: 1.0 - (3000 / 3000) = 0.0
# Haiku score: 1.0 - (600 / 3000) = 0.8
```

### **Step 5: Cost Analysis & Savings Calculation**

```python
def calculate_savings(selected_model, baseline_model_id, estimated_tokens):
    baseline_model = model_registry.models[baseline_model_id]
    
    # Calculate costs (input + estimated output)
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

# Example:
# Selected: GPT-4o-mini
# Baseline: GPT-4o (default premium model)
# Estimated tokens: 150 (input) + 45 (output) = 195 total
# 
# selected_cost = (150 * 0.15/1000) + (45 * 0.60/1000) = $0.0495
# baseline_cost = (150 * 5.00/1000) + (45 * 15.00/1000) = $1.425
# 
# savings = $1.425 - $0.0495 = $1.3755
# savings_percentage = 96.5%
```

### **Step 6: Decision Confidence**

```python
def calculate_decision_confidence(model_scores):
    if len(model_scores) < 2:
        return 1.0
    
    # Calculate confidence based on score difference
    top_score = model_scores[0].total_score
    second_score = model_scores[1].total_score
    
    score_gap = top_score - second_score
    
    # Higher gap = higher confidence
    confidence = min(1.0, 0.5 + score_gap * 2)
    
    return confidence

# Example:
# Top model score: 0.875
# Second model score: 0.854
# Score gap: 0.021
# Confidence: min(1.0, 0.5 + 0.021 * 2) = 0.542
```

#### **Reasoning Generation**

```python
def generate_reasoning(model, quality_score, cost_score, latency_score, total_score):
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
```

### **Step 7: Response Enrichment**

```python
# Final response with complete routing information
response = {
    "id": "chatcmpl-123",
    "choices": [{"message": {"content": "def sort_list(lst):..."}}],
    "usage": {"prompt_tokens": 150, "completion_tokens": 45, "total_tokens": 195},
    "router_info": {
        "selected_model": "gpt-4o-mini",
        "reason": "Selected for: Excellent cost efficiency, Good quality for coding",
        "latency_ms": 823,
        "cost": 0.0495,
        "savings_vs_baseline": 1.3755,
        "baseline_model": "gpt-4o",
        "classification": {
            "task_type": "coding",
            "complexity_level": "medium",
            "estimated_tokens": 195,
            "quality_threshold": 0.85,
            "confidence": 0.8
        },
        "decision_confidence": 0.542,
        "routing_strategy": "balanced"
    }
}
```

## **🎯 Real Decision Logic Examples**

### **Example 1: Simple Question**
```
Input: "What is the capital of France?"

Classification Process:
- Task keywords: None specific → TaskType.CHAT
- Complexity: Very short (5 words), simple question → ComplexityLevel.VERY_LOW
- Quality threshold: 0.60 (chat tasks have lower requirements)
- Latency requirement: FAST (simple queries expect quick responses)

Decision Logic:
1. All models meet quality threshold (0.60)
2. Cost becomes primary differentiator
3. Claude-3-Haiku: cost_score=0.95, quality_score=0.75 → total=0.89
4. GPT-4o-mini: cost_score=0.92, quality_score=0.70 → total=0.87

Selected: Claude-3-Haiku
Reasoning: "Excellent cost efficiency for simple query"
Cost: $0.003 (vs $0.85 baseline) → 99.6% savings
```

### **Example 2: Complex Analysis**
```
Input: "Analyze the economic implications of artificial intelligence on employment markets, considering both short-term disruptions and long-term societal benefits with supporting data and references."

Classification Process:
- Task keywords: ["analyze", "implications", "economic"] → TaskType.REASONING
- Complexity: Long text (140 chars), multi-part analysis → ComplexityLevel.VERY_HIGH
- Quality threshold: 0.95 (reasoning tasks need high accuracy)
- Latency requirement: SLOW (complex analysis can take time)

Decision Logic:
1. Filter models: Only Claude-3.5-Sonnet and GPT-4o meet 0.95 quality threshold
2. Quality becomes primary factor due to high requirements
3. Claude-3.5-Sonnet: quality_score=0.97, cost_score=0.30 → total=0.78
4. GPT-4o: quality_score=0.95, cost_score=0.20 → total=0.72

Selected: Claude-3.5-Sonnet
Reasoning: "Premium quality required for complex economic analysis"
Cost: $3.15 (vs $3.15 baseline) → 0% savings (but optimal quality)
```

### **Example 3: Code Review**
```
Input: "Review this Python code for bugs and performance issues: [large code block with 50 lines]"

Classification Process:
- Task keywords: ["review", "code", "python", "bugs"] → TaskType.CODING
- Complexity: Large code block, detailed analysis → ComplexityLevel.HIGH
- Has code: True (code block detected)
- Quality threshold: 0.85 (coding needs accuracy)

Decision Logic:
1. Filter models: GPT-4o, GPT-4o-mini, Claude-3.5-Sonnet meet coding quality
2. Balance quality and cost for coding task
3. GPT-4o: quality_score=0.92, cost_score=0.20 → total=0.71
4. GPT-4o-mini: quality_score=0.85, cost_score=0.92 → total=0.84

Selected: GPT-4o-mini
Reasoning: "Good coding quality with excellent cost efficiency"
Cost: $0.45 (vs $2.10 baseline) → 78.6% savings
```

### **Example 4: Creative Writing**
```
Input: "Write a short story about a time traveler who accidentally changes history"

Classification Process:
- Task keywords: ["write", "story", "creative"] → TaskType.CREATIVE_WRITING
- Complexity: Creative task, moderate length → ComplexityLevel.MEDIUM  
- Quality threshold: 0.75 (creative tasks prioritize creativity over accuracy)
- Latency requirement: NORMAL

Decision Logic:
1. All models meet creative writing quality threshold
2. Look for models with good creative capabilities
3. Claude-3-5-Sonnet: quality_score=0.90, cost_score=0.30 → total=0.73
4. GPT-4o: quality_score=0.88, cost_score=0.20 → total=0.68

Selected: Claude-3.5-Sonnet  
Reasoning: "High creativity score for narrative generation"
Cost: $1.25 (vs $2.80 baseline) → 55.4% savings
```

## **⚡ Advanced Logic Features**

### **Sticky Session Routing**
```python
def make_routing_decision(classification, constraints, session_id):
    # Check for sticky routing first
    if session_id and policy.enable_session_routing:
        if session_id in session_models:
            sticky_model_id = session_models[session_id]
            sticky_model = model_registry.models[sticky_model_id]
            
            # Verify the sticky model still meets constraints
            if model_meets_constraints(sticky_model, constraints):
                return create_sticky_decision(
                    sticky_model,
                    "Sticky routing - continuing with same model from session"
                )
    
    # Otherwise use normal policy engine decision
    return policy_engine.make_decision(classification, constraints)

# Benefits: Conversation continuity, consistent model behavior, reduced latency
```

### **Fallback Logic**
```python
async def execute_request(request, decision):
    provider = provider_factory.get_provider(decision.selected_model.provider)
    
    try:
        response = await provider.complete_chat(request, decision.selected_model)
        return response
    except Exception as e:
        # Try fallback if enabled
        if policy.enable_fallback:
            fallback_model = fallback_manager.get_fallback_model(
                decision.selected_model,
                decision.constraints
            )
            
            if fallback_model:
                fallback_provider = provider_factory.get_provider(fallback_model.provider)
                response = await fallback_provider.complete_chat(request, fallback_model)
                response.router_info.fallback_used = True
                response.router_info.reason = f"Fallback used due to: {e}"
                return response
        
        raise Exception(f"Primary and fallback models failed: {e}")

# Fallback selection priority:
# 1. Same provider, different model
# 2. Different provider, similar capability
# 3. Most reliable available model
```

### **Dynamic Constraints**
```python
def infer_constraints_from_content(text, request):
    constraints = RoutingConstraints()
    
    # Urgency detection
    urgency_keywords = ["urgent", "asap", "immediately", "quick", "fast"]
    if any(keyword in text.lower() for keyword in urgency_keywords):
        constraints.max_latency_ms = 1000  # Force fast models
        constraints.latency_requirement = "real_time"
    
    # Quality requirements
    quality_keywords = ["detailed", "comprehensive", "thorough", "accurate", "precise"]
    if any(keyword in text.lower() for keyword in quality_keywords):
        constraints.min_quality_score = 0.90  # Force high quality
    
    # Budget sensitivity
    budget_keywords = ["cheap", "budget", "cost-effective", "economical"]
    if any(keyword in text.lower() for keyword in budget_keywords):
        constraints.max_input_cost_per_1k = 0.5  # Force cheap models
    
    # Tool requirements
    if request.tools or "function" in text or "call api" in text:
        constraints.required_capabilities.append("tools")
    
    # JSON requirements
    if "json" in text.lower() or "structured" in text.lower():
        constraints.required_capabilities.append("json_mode")
    
    return constraints
```

### **Performance Learning**
```python
class PerformanceLearner:
    def __init__(self):
        self.model_performance_history = {}
    
    def update_model_performance(self, model_id, task_type, actual_quality, actual_latency, actual_cost):
        """Update model performance based on real results."""
        if model_id not in self.model_performance_history:
            self.model_performance_history[model_id] = {}
        
        if task_type not in self.model_performance_history[model_id]:
            self.model_performance_history[model_id][task_type] = {
                "quality_scores": [],
                "latency_scores": [],
                "cost_scores": []
            }
        
        history = self.model_performance_history[model_id][task_type]
        history["quality_scores"].append(actual_quality)
        history["latency_scores"].append(actual_latency)  
        history["cost_scores"].append(actual_cost)
        
        # Keep only recent performance data
        for metric in history.values():
            if len(metric) > 100:
                metric.pop(0)
    
    def get_adjusted_scores(self, model_id, task_type):
        """Get performance-adjusted scores for a model."""
        if model_id not in self.model_performance_history:
            return None
            
        history = self.model_performance_history[model_id].get(task_type)
        if not history:
            return None
        
        # Calculate moving averages
        avg_quality = sum(history["quality_scores"][-20:]) / len(history["quality_scores"][-20:])
        avg_latency = sum(history["latency_scores"][-20:]) / len(history["latency_scores"][-20:])
        
        return {
            "adjusted_quality_score": avg_quality,
            "adjusted_latency_ms": avg_latency
        }
```

## **📊 Policy Engine Strategies**

### **Cost-Optimized Policy**
```python
cost_optimized_policy = RoutingPolicy(
    quality_weight=0.20,   # Minimum viable quality
    cost_weight=0.60,      # Prioritize cost savings
    latency_weight=0.15,   # Moderate latency concern
    reliability_weight=0.05,
    cost_priority=CostPriority.MINIMIZE_COST
)

# Typical results:
# - Simple queries: Claude-3-Haiku (cheapest)
# - Coding tasks: GPT-4o-mini (good quality, low cost)
# - Complex analysis: Still uses premium when quality demands it
# - Average savings: 70-85% vs always-premium strategy
```

### **Quality-Optimized Policy**
```python
quality_optimized_policy = RoutingPolicy(
    quality_weight=0.60,   # Prioritize highest quality
    cost_weight=0.15,      # Cost is secondary
    latency_weight=0.20,   # Moderate latency importance
    reliability_weight=0.05,
    cost_priority=CostPriority.MAXIMIZE_QUALITY
)

# Typical results:
# - Simple queries: Still uses efficient models (quality threshold met)
# - Coding tasks: GPT-4o or Claude-3.5-Sonnet
# - Complex analysis: Always premium models
# - Average cost increase: 40-60% vs balanced, but maximum quality
```

### **Balanced Policy (Default)**
```python
balanced_policy = RoutingPolicy(
    quality_weight=0.30,   # Good quality importance
    cost_weight=0.40,      # Significant cost consideration
    latency_weight=0.20,   # Reasonable latency priority
    reliability_weight=0.10, # Reliability matters
    cost_priority=CostPriority.BALANCE
)

# Typical results:
# - Optimal cost/quality tradeoffs for each task type
# - Adapts to task complexity automatically
# - Average savings: 50-70% vs always-premium
# - Quality degradation: <5% vs always-premium
```

### **Latency-Optimized Policy**
```python
latency_optimized_policy = RoutingPolicy(
    quality_weight=0.25,
    cost_weight=0.25,
    latency_weight=0.45,   # Prioritize speed
    reliability_weight=0.05,
    max_latency_ms=2000    # Hard limit on response time
)

# Typical results:
# - Consistently selects fastest available models
# - May sacrifice some quality for speed
# - Good for real-time applications, chatbots
# - Average response time: <2 seconds
```

## **🔄 Multi-Agent Workflow Logic (LangGraph)**

### **Agent Specialization Strategy**
```python
# Different agents get different routing configurations
def create_specialized_agents():
    # Research Agent: Quality-focused for accuracy
    researcher_llm = IntelliRouterLLM(
        routing_constraints=RoutingConstraints(
            min_quality_score=0.90,  # High accuracy needed
            max_latency_ms=15000,    # Can wait for better results
            required_capabilities=["tools"]  # May need to search/calculate
        )
    )
    
    # Summarizer Agent: Balanced efficiency
    summarizer_llm = IntelliRouterLLM(
        routing_constraints=RoutingConstraints(
            min_quality_score=0.75,  # Good enough quality
            max_input_cost_per_1k=1.0,  # Cost-conscious
            max_latency_ms=8000      # Reasonably fast
        )
    )
    
    # Validator Agent: Fast and cheap for simple validation
    validator_llm = IntelliRouterLLM(
        routing_constraints=RoutingConstraints(
            min_quality_score=0.65,  # Basic quality sufficient
            max_latency_ms=3000,     # Fast validation
            max_input_cost_per_1k=0.5  # Very cost-conscious
        )
    )
    
    return researcher_llm, summarizer_llm, validator_llm
```

### **Adaptive Workflow Routing**
```python
def create_adaptive_workflow():
    # Classifier agent determines task complexity
    classifier_llm = IntelliRouterLLM(
        routing_constraints=RoutingConstraints(
            max_latency_ms=2000,     # Fast classification
            max_input_cost_per_1k=0.3  # Cheap classification
        )
    )
    
    # Simple task agent - cost-optimized
    simple_llm = IntelliRouterLLM(
        routing_constraints=RoutingConstraints(
            min_quality_score=0.70,
            max_input_cost_per_1k=0.5,
            max_latency_ms=5000
        )
    )
    
    # Complex task agent - quality-optimized
    complex_llm = IntelliRouterLLM(
        routing_constraints=RoutingConstraints(
            min_quality_score=0.90,
            required_capabilities=["tools"],
            max_latency_ms=20000
        )
    )

def classify_and_route(query):
    # Step 1: Classify complexity
    classification_prompt = f"""
    Classify this query as SIMPLE or COMPLEX:
    Query: {query}
    
    SIMPLE: Basic questions, simple math, casual conversation
    COMPLEX: Analysis, coding, research, multi-step reasoning
    """
    
    complexity = classifier_llm.invoke(classification_prompt)
    
    # Step 2: Route to appropriate agent
    if "COMPLEX" in complexity.content:
        return complex_llm.invoke(query)
    else:
        return simple_llm.invoke(query)
```

## **🎛️ Key Decision Factors Summary**

### **1. Task Type → Quality Requirements**
- **CHAT**: 0.60 threshold (basic accuracy)
- **CODING**: 0.85 threshold (high accuracy needed)
- **REASONING**: 0.90 threshold (logical consistency critical)
- **CREATIVE_WRITING**: 0.75 threshold (creativity over precision)

### **2. Complexity → Model Capability Needs**
- **VERY_LOW**: Any model sufficient, optimize for cost
- **LOW**: Basic models acceptable, slight quality preference
- **MEDIUM**: Balanced approach, moderate quality requirements
- **HIGH**: Quality becomes important, willing to pay more
- **VERY_HIGH**: Premium models required, cost secondary

### **3. Content Analysis → Feature Requirements**
- **Code blocks detected** → `requires_tools = True`
- **JSON/structured mentions** → `requires_json_mode = True`
- **Image/visual references** → `requires_vision = True`
- **Mathematical content** → Higher quality threshold
- **Long conversations** → Session stickiness enabled

### **4. Policy Weights → Optimization Strategy**
- **Cost-focused**: `cost_weight = 0.60, quality_weight = 0.20`
- **Quality-focuse
