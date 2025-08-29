# 🔗 LangGraph Integration Guide

IntelliRouter includes comprehensive LangGraph integration, enabling intelligent model routing within complex multi-agent workflows with automatic cost optimization.

## 🚀 Quick Start

### Installation

```bash
# LangGraph dependencies are included in requirements.txt
pip install -r requirements.txt
```

### Basic Usage

```python
from intellirouter.integrations.langgraph import IntelliRouterLLM
from langchain_core.messages import HumanMessage

# Create an IntelliRouter-powered LLM for LangGraph
llm = IntelliRouterLLM(intellirouter_url="http://localhost:8000")

# Use it like any LangChain LLM
response = llm.invoke([HumanMessage(content="What is the capital of France?")])
print(response.content)  # "The capital of France is Paris."

# Access routing information
router_info = response.response_metadata.get('router_info', {})
print(f"Model used: {router_info.get('selected_model')}")
print(f"Cost: ${router_info.get('cost', 0):.6f}")
```

## 🏗️ Pre-built Workflow Examples

### 1. Simple Chat Graph

```python
from intellirouter.integrations.langgraph import create_simple_chat_graph
from langchain_core.messages import HumanMessage

# Create a simple chat workflow
graph = create_simple_chat_graph(intellirouter_url="http://localhost:8000")

# Run it
result = await graph.ainvoke({
    "messages": [HumanMessage(content="What is AI?")]
})

print(result['messages'][-1].content)
print(f"Cost: ${result['router_info'].get('cost', 0):.6f}")
```

### 2. Multi-Agent Research Workflow

```python
from intellirouter.integrations.langgraph import create_multi_agent_graph

# Create multi-agent workflow with specialized routing per agent
graph = create_multi_agent_graph(intellirouter_url="http://localhost:8000")

# Run research workflow
result = await graph.ainvoke({
    "query": "What are the benefits and risks of renewable energy?"
})

print("Research:", result['research_result'][:200], "...")
print("Summary:", result['summary'][:200], "...")
print("Validation:", result['validation_result'])

# See costs by agent
print("\n📊 Routing Information by Agent:")
for router_info in result.get('router_infos', []):
    agent = router_info.get('agent', 'unknown')
    model = router_info.get('selected_model', 'unknown')
    cost = router_info.get('cost', 0)
    print(f"  • {agent.title()}: {model} (${cost:.6f})")

total_cost = sum(info.get('cost', 0) for info in result.get('router_infos', []))
print(f"💰 Total workflow cost: ${total_cost:.6f}")
```

**Agent Specialization:**
- **Researcher**: Uses high-quality models (min_quality_score=0.85)
- **Summarizer**: Balances cost/quality (max_input_cost_per_1k=1.0)
- **Validator**: Fast, cheap validation (max_latency_ms=3000)

### 3. Adaptive Complexity-Based Routing

```python
from intellirouter.integrations.langgraph import create_adaptive_graph

# Create adaptive workflow that routes based on task complexity
graph = create_adaptive_graph(intellirouter_url="http://localhost:8000")

# Test with simple query -> uses cost-effective models
result = await graph.ainvoke({"query": "What's 2+2?"})
print(f"Simple query classified as: {result['complexity']}")  # "SIMPLE"
print(f"Response: {result['response']}")

# Test with complex query -> uses high-quality models
result = await graph.ainvoke({
    "query": "Analyze the economic implications of artificial intelligence on employment"
})
print(f"Complex query classified as: {result['complexity']}")  # "COMPLEX"
print(f"Response: {result['response'][:100]}...")

# View model usage by stage
print("📊 Model usage by stage:")
for router_info in result.get('router_infos', []):
    stage = router_info.get('stage', 'unknown')
    model = router_info.get('selected_model', 'unknown')
    cost = router_info.get('cost', 0)
    print(f"  • {stage}: {model} (${cost:.6f})")
```

## ⚙️ Custom Agent Configurations

### Specialized Agents with Different Routing Strategies

```python
from intellirouter.integrations.langgraph import IntelliRouterLLM
from intellirouter.models.routing import RoutingConstraints

# Budget-conscious agent - prioritizes cost
budget_llm = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000",
    routing_constraints=RoutingConstraints(
        max_input_cost_per_1k=0.5,  # Very cost-conscious
        max_latency_ms=5000
    )
)

# Quality-first agent - prioritizes accuracy
quality_llm = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000",
    routing_constraints=RoutingConstraints(
        min_quality_score=0.90,  # High quality required
        required_capabilities=["tools"]
    )
)

# Speed-first agent - prioritizes response time
speed_llm = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000",
    routing_constraints=RoutingConstraints(
        max_latency_ms=2000,  # Fast response required
        min_quality_score=0.65
    )
)

# Test different agents with the same query
test_query = "Explain quantum computing in simple terms"
message = [HumanMessage(content=test_query)]

for name, llm in [("Budget", budget_llm), ("Quality", quality_llm), ("Speed", speed_llm)]:
    result = llm.invoke(message)
    router_info = result.response_metadata.get('router_info', {})
    
    print(f"\n🔧 {name} Agent:")
    print(f"  Model: {router_info.get('selected_model', 'unknown')}")
    print(f"  Cost: ${router_info.get('cost', 0):.6f}")
    print(f"  Latency: {router_info.get('latency_ms', 0)}ms")
    print(f"  Reason: {router_info.get('reason', 'No reason provided')}")
```

### Building Custom Workflows

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, List, Dict, Any
from langchain_core.messages import HumanMessage

# Define workflow state
class ContentWorkflowState(TypedDict):
    topic: str
    research: Optional[str]
    draft: Optional[str]
    final_content: Optional[str]
    router_infos: List[Dict[str, Any]]

# Create specialized agents
researcher_llm = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000",
    routing_constraints=RoutingConstraints(
        min_quality_score=0.90,  # High quality for research
        max_latency_ms=15000
    )
)

writer_llm = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000",
    routing_constraints=RoutingConstraints(
        min_quality_score=0.80,  # Good quality for writing
        max_input_cost_per_1k=2.0
    )
)

editor_llm = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000",
    routing_constraints=RoutingConstraints(
        min_quality_score=0.95,  # Premium quality for editing
        required_capabilities=["tools", "json_mode"]
    )
)

def research_node(state: ContentWorkflowState) -> ContentWorkflowState:
    """Research using high-quality model."""
    result = researcher_llm.invoke([
        HumanMessage(content=f"Research this topic comprehensively: {state['topic']}")
    ])
    
    router_info = result.response_metadata.get('router_info', {})
    router_info['stage'] = 'research'
    
    current_infos = state.get("router_infos", [])
    
    return {
        **state,
        "research": result.content,
        "router_infos": current_infos + [router_info]
    }

def draft_node(state: ContentWorkflowState) -> ContentWorkflowState:
    """Create draft using balanced model."""
    result = writer_llm.invoke([
        HumanMessage(content=f"Write a comprehensive article based on: {state['research']}")
    ])
    
    router_info = result.response_metadata.get('router_info', {})
    router_info['stage'] = 'writing'
    
    current_infos = state.get("router_infos", [])
    
    return {
        **state,
        "draft": result.content,
        "router_infos": current_infos + [router_info]
    }

def edit_node(state: ContentWorkflowState) -> ContentWorkflowState:
    """Final edit using premium model."""
    result = editor_llm.invoke([
        HumanMessage(content=f"Polish and improve this content: {state['draft']}")
    ])
    
    router_info = result.response_metadata.get('router_info', {})
    router_info['stage'] = 'editing'
    
    current_infos = state.get("router_infos", [])
    
    return {
        **state,
        "final_content": result.content,
        "router_infos": current_infos + [router_info]
    }

# Build custom workflow
workflow = StateGraph(ContentWorkflowState)
workflow.add_node("research", research_node)
workflow.add_node("draft", draft_node)
workflow.add_node("edit", edit_node)

workflow.add_edge(START, "research")
workflow.add_edge("research", "draft")
workflow.add_edge("draft", "edit")
workflow.add_edge("edit", END)

content_pipeline = workflow.compile()

# Run the workflow
result = await content_pipeline.ainvoke({
    "topic": "The Future of Renewable Energy"
})

print("Final Content:", result['final_content'][:300], "...")
print("\n💰 Cost Breakdown:")
total_cost = 0
for router_info in result['router_infos']:
    stage = router_info.get('stage', 'unknown')
    model = router_info.get('selected_model', 'unknown')
    cost = router_info.get('cost', 0)
    total_cost += cost
    print(f"  {stage}: {model} (${cost:.6f})")

print(f"📊 Total Cost: ${total_cost:.6f}")
```

## 💡 Advanced Features

### Session-Aware Routing

```python
# Maintain conversation context across workflow steps
session_llm = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000",
    session_id="user-123-session-456"  # Enables sticky routing
)
```

### Cost Comparison Across Strategies

```python
from intellirouter.integrations.langgraph import IntelliRouterLLM

# Create different LLMs for comparison
always_premium = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000",
    routing_constraints=RoutingConstraints(min_quality_score=0.95)
)

auto_route = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000"
    # No constraints - let IntelliRouter choose optimally
)

budget_only = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000",
    routing_constraints=RoutingConstraints(max_input_cost_per_1k=0.3)
)

# Test queries of different complexities
test_queries = [
    "Hello! How are you?",
    "What's the weather like today?", 
    "Write a Python function to sort a list",
    "Explain the theory of relativity with mathematical proofs"
]

strategies = [
    ("Always Premium", always_premium),
    ("Auto Route", auto_route),
    ("Budget Only", budget_only)
]

total_costs = {name: 0 for name, _ in strategies}

for query in test_queries:
    print(f"\n📝 Query: {query}")
    message = [HumanMessage(content=query)]
    
    for strategy_name, llm in strategies:
        try:
            result = llm.invoke(message)
            router_info = result.response_metadata.get('router_info', {})
            
            cost = router_info.get('cost', 0)
            model = router_info.get('selected_model', 'unknown')
            total_costs[strategy_name] += cost
            
            print(f"  {strategy_name}: {model} (${cost:.6f})")
            
        except Exception as e:
            print(f"  {strategy_name}: ❌ {e}")

# Show total costs
print(f"\n💰 Total Costs Comparison:")
for strategy_name, total_cost in total_costs.items():
    print(f"  {strategy_name}: ${total_cost:.6f}")

# Calculate savings
if total_costs["Always Premium"] > 0:
    auto_savings = ((total_costs["Always Premium"] - total_costs["Auto Route"]) / 
                  total_costs["Always Premium"]) * 100
    print(f"\n💵 Auto Route saves {auto_savings:.1f}% vs Always Premium")
```

## 🔧 Configuration Options

### IntelliRouterLLM Parameters

```python
llm = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000",  # IntelliRouter server URL
    api_key="your-api-key",                     # Optional authentication
    routing_constraints=RoutingConstraints(...), # Model selection constraints
    session_id="session-123",                   # Session for sticky routing
    temperature=0.7,                            # Default temperature
    max_tokens=1000                             # Default max tokens
)
```

### Routing Constraints Options

```python
from intellirouter.models.routing import RoutingConstraints

constraints = RoutingConstraints(
    # Cost constraints
    max_input_cost_per_1k=1.0,
    max_output_cost_per_1k=3.0,
    max_total_cost=0.01,
    
    # Performance constraints  
    max_latency_ms=5000,
    min_reliability_score=0.95,
    
    # Quality requirements
    min_quality_score=0.80,
    
    # Capability requirements
    required_capabilities=["tools", "json_mode", "vision"],
    
    # Provider constraints
    excluded_providers=["provider-with-issues"],
    preferred_providers=["openai", "anthropic"]
)
```

## 📊 Monitoring and Analytics

### Real-time Cost Tracking

```python
# Monitor costs in real-time during workflow execution
async def execute_with_monitoring(graph, initial_state):
    result = await graph.ainvoke(initial_state)
    
    # Extract cost information
    total_cost = 0
    model_usage = {}
    
    for router_info in result.get('router_infos', []):
        cost = router_info.get('cost', 0)
        model = router_info.get('selected_model', 'unknown')
        
        total_cost += cost
        model_usage[model] = model_usage.get(model, 0) + 1
    
    print(f"✅ Workflow completed:")
    print(f"  💰 Total cost: ${total_cost:.6f}")
    print(f"  🤖 Models used: {model_usage}")
    
    return result
```

### Performance Optimization

```python
# A/B test different routing strategies
async def compare_strategies(query: str):
    strategies = {
        "cost_optimized": RoutingConstraints(max_input_cost_per_1k=0.5),
        "quality_optimized": RoutingConstraints(min_quality_score=0.90),
        "balanced": RoutingConstraints()  # Default IntelliRouter optimization
    }
    
    results = {}
    
    for name, constraints in strategies.items():
        llm = IntelliRouterLLM(
            intellirouter_url="http://localhost:8000",
            routing_constraints=constraints
        )
        result = llm.invoke([HumanMessage(content=query)])
        router_info = result.response_metadata.get('router_info', {})
        
        results[name] = {
            'cost': router_info.get('cost', 0),
            'model': router_info.get('selected_model'),
            'latency': router_info.get('latency_ms', 0),
            'response_length': len(result.content)
        }
    
    return results

# Example usage
strategy_comparison = await compare_strategies("Explain machine learning")
for strategy, metrics in strategy_comparison.items():
    print(f"{strategy}: {metrics['model']} - ${metrics['cost']:.6f} - {metrics['latency']}ms")
```

## 🚀 Running Examples

### Prerequisites

```bash
# 1. Start IntelliRouter server
uvicorn src.intellirouter.api.app:create_app --factory --host 0.0.0.0 --port 8000 &

# 2. Configure API keys in .env
cp .env.example .env
# Edit .env with your API keys

# 3. Install dependencies
pip install -r requirements.txt
```

### Run Comprehensive Examples

```bash
# Run all LangGraph examples
python examples/langgraph_examples.py
```

**Example Output:**
```
🚀 IntelliRouter + LangGraph Integration Examples
============================================================

🤖 Example 1: Simple Chat with LangGraph + IntelliRouter
🤖 Response: The capital of France is Paris.
📊 Model used: anthropic.claude-3-haiku-20240307-v1:0
💰 Cost: $0.006000
⏱️  Latency: 551ms

🧠 Example 2: Multi-Agent Research Workflow
📊 Routing Information by Agent:
  • Researcher: anthropic.claude-3-haiku-20240307-v1:0 ($0.226250)
  • Summarizer: anthropic.claude-3-5-sonnet-20240620-v1:0 ($3.177000)
  • Validator: anthropic.claude-3-haiku-20240307-v1:0 ($0.046500)
💰 Total workflow cost: $3.449750

🎯 Example 3: Adaptive Complexity-Based Routing
📝 Query 1: What's 2+2?... → 🏷️ Classified as: SIMPLE
📝 Query 2: Analyze AI economic implications... → 🏷️ Classified as: COMPLEX

💰 Example 5: Cost Comparison Across Strategies
Always Premium: $0.323750
Auto Route: $0.363000
Budget Only: $0.418750

🎉 All examples completed!
```

## 📈 Benefits

- **💰 Cost Optimization**: Save 50-80% on complex workflows by using optimal models for each step
- **🎯 Quality Assurance**: Ensure high-quality outputs where needed while using efficient models elsewhere  
- **📈 Scalability**: Handle complex multi-agent workflows with automatic load balancing
- **🔧 Flexibility**: Easy to customize routing strategies per agent or workflow step
- **📊 Analytics**: Comprehensive tracking of costs, performance, and model usage across workflows
- **🛡️ Reliability**: Built-in fallback and error handling for robust production workflows
- **🔗 LangChain Compatible**: Full BaseChatModel implementation works with entire LangChain ecosystem
- **⚡ Async Support**: Both sync and async execution for high-performance applications

## 🎯 Use Cases

### 1. Multi-Agent Research Pipeline
- **Research Agent**: High-quality models for comprehensive research
- **Analysis Agent**: Premium models for deep analysis
- **Summary Agent**: Balanced models for concise summaries
- **Review Agent**: Cost-effective models for final review

### 2. Content Creation Workflow
- **Research**: Quality-focused routing for accurate information
- **Writing**: Creativity-optimized models for engaging content
- **Editing**: Premium models for polishing and refinement
- **Fact-checking**: Fast, reliable models for verification

### 3. Customer Support Automation
- **Classification**: Fast models to categorize inquiries
- **Simple Questions**: Cost-effective models for basic responses
- **Complex Issues**: Premium models for technical support
- **Escalation**: Human handoff with complete context

---

**Ready to build intelligent, cost-optimized AI workflows with LangGraph + IntelliRouter! 🚀**

*Transform your multi-agent workflows with intelligent model routing that saves costs while maintaining quality.*
