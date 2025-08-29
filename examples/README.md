# IntelliRouter Agent Examples

This directory contains practical examples of intelligent agents built with IntelliRouter that demonstrate automatic model selection based on task complexity and requirements.

## 🤖 Available Agent Examples

### 1. Intelligent Agent (`intelligent_agent_example.py`)

A comprehensive agent that automatically classifies tasks and routes them to optimal AI models.

**Features:**
- **Task Classification**: Automatically detects task types (Q&A, coding, analysis, creative writing, etc.)
- **Prompt Optimization**: Optimizes prompts for each task type
- **Cost Analytics**: Tracks costs, model usage, and performance
- **Interactive Mode**: Chat-like interface for testing

**Task Types Supported:**
- Simple Q&A (uses cost-effective models)
- Code Generation (uses coding-optimized models)
- Complex Analysis (uses high-quality models)
- Creative Writing (uses creativity-optimized models)
- Research (balanced quality/cost)
- Summarization (efficient processing)

### 2. LangGraph Integration (`langgraph_examples.py`) ⭐ **NEW!**

Advanced multi-agent workflows using LangGraph with IntelliRouter for intelligent model routing.

**Features:**
- **5 Comprehensive Examples**: From simple chat to complex multi-agent workflows
- **Multi-Agent Research Workflows**: Different agents with specialized routing strategies
- **Adaptive Complexity Routing**: Dynamic model selection based on task analysis
- **Custom Routing Constraints**: Per-agent routing policies (budget, quality, speed)
- **Cost Comparison Analytics**: Compare different routing strategies
- **Real-Time Cost Tracking**: Monitor expenses across complex workflows

**Examples Included:**
1. **Simple Chat Graph**: Basic LangGraph + IntelliRouter integration
2. **Multi-Agent Research**: Researcher → Summarizer → Validator workflow
3. **Adaptive Complexity**: Automatic simple vs complex task routing
4. **Custom Constraints**: Budget-conscious, quality-first, and speed-first agents
5. **Cost Comparison**: Compare "always premium" vs "auto route" vs "budget only"

## 🚀 Quick Start

### Prerequisites

```bash
# 1. Install dependencies
pip install -r ../requirements.txt

# 2. Configure API keys in .env file
cp ../.env.example ../.env
# Edit .env with your OpenAI/Anthropic/AWS Bedrock API keys

# 3. Start IntelliRouter server (in separate terminal)
cd ..
uvicorn src.intellirouter.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### Running the Examples

**Option 1: Basic Intelligent Agent**
```bash
python intelligent_agent_example.py
# Choose option 1 for demo, option 2 for interactive chat
```

**Option 2: LangGraph Integration Examples (Recommended!)**
```bash
python langgraph_examples.py
```

**Expected LangGraph Output:**
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

## 📊 Example Usage

### Basic Agent Usage

```python
from intelligent_agent_example import IntelliRouterAgent

# Create agent
agent = IntelliRouterAgent()

# Process different types of tasks
result = await agent.process_task("Write a Python sorting function")
print(f"Model used: {result.model_used}")
print(f"Cost: ${result.cost:.6f}")
print(f"Response: {result.response}")
```

### LangGraph Integration Usage

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

### Multi-Agent Workflow

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
total_cost = sum(info.get('cost', 0) for info in result.get('router_infos', []))
print(f"💰 Total workflow cost: ${total_cost:.6f}")
```

### Task Classification Examples

The agent automatically classifies these tasks:

```python
# Simple Q&A → Uses cost-effective model
"What is the capital of France?"

# Code Generation → Uses coding-optimized model
"Write a Python function to calculate fibonacci"

# Complex Analysis → Uses high-quality model
"Analyze the economic impacts of renewable energy"

# Creative Writing → Uses creativity-optimized model
"Write a short story about time travel"
```

## 🔧 Customization

### Creating Your Own Agent

```python
from intellirouter.core.router import ModelRouter
from intellirouter.models.routing import RoutingPolicy, CostPriority

class CustomAgent:
    def __init__(self):
        # Customize routing policy
        policy = RoutingPolicy(
            cost_priority=CostPriority.MINIMIZE_COST,  # Focus on cost
            quality_weight=0.2,
            cost_weight=0.7,
            latency_weight=0.1
        )
        
        self.router = ModelRouter(model_registry, policy)
    
    async def process_query(self, query: str):
        # Custom processing logic
        request = ChatCompletionRequest(
            model="auto",
            messages=[ChatMessage(role="user", content=query)]
        )
        return await self.router.route_request(request)
```

### Custom LangGraph Agent with Routing Constraints

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
```

### Custom Task Classification

```python
def classify_custom_task(self, user_input: str) -> TaskType:
    """Add your own classification logic."""
    
    # Example: Detect medical queries
    medical_keywords = ['diagnosis', 'symptoms', 'treatment', 'medical']
    if any(word in user_input.lower() for word in medical_keywords):
        return TaskType.MEDICAL_QUERY
    
    # Fall back to default classification
    return self._classify_task(user_input)
```

## 📈 Performance & Analytics

### Agent Analytics

```python
analytics = agent.get_analytics()
print(f"Total cost: {analytics['total_cost']}")
print(f"Model usage: {analytics['model_usage']}")
print(f"Average latency: {analytics['average_latency_ms']}")
```

### Cost Optimization Tips

1. **Task Classification**: Ensure accurate task classification to use optimal models
2. **Prompt Engineering**: Use task-specific prompts for better results
3. **Routing Policy**: Adjust cost/quality weights based on your needs
4. **Batch Processing**: Process similar tasks together for efficiency
5. **LangGraph Workflows**: Use specialized agents for different workflow steps

## 🎯 Use Cases

### 1. Customer Support Agent
- Route simple queries to fast, cheap models
- Use premium models for complex technical issues
- Track costs per support ticket

### 2. Content Creation Assistant
- Simple editing → Cost-effective models
- Creative writing → Creativity-optimized models
- Technical documentation → Accuracy-focused models

### 3. Code Assistant
- Code review → Quality-focused models
- Simple questions → Cost-effective models
- Complex debugging → Premium models

### 4. Research Assistant (LangGraph Multi-Agent)
- **Research Agent**: Quality-focused routing for comprehensive research
- **Analysis Agent**: Premium models for deep analysis
- **Summary Agent**: Balanced models for concise summaries
- **Review Agent**: Cost-effective models for final review

### 5. Multi-Step Workflows (LangGraph)
- **Classification**: Fast models to categorize tasks
- **Processing**: Appropriate models based on complexity
- **Validation**: Cost-effective models for quality checks
- **Refinement**: Premium models for final polish

## 🔍 Monitoring & Debugging

### Enable Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run agent with detailed logging
result = await agent.process_task("Your query here")
```

### Common Issues

1. **No models available**: Check API keys in `.env`
2. **High costs**: Adjust routing policy cost weights
3. **Slow responses**: Check latency constraints in routing policy
4. **Poor quality**: Increase quality weight in routing policy
5. **LangGraph errors**: Ensure server is running and updated to latest API

## 📚 Advanced Examples

### Multi-Step Workflows (Traditional)

```python
async def research_workflow(self, topic: str):
    """Multi-step research with different models for each step."""
    
    # Step 1: Research (balanced model)
    research = await self.process_task(f"Research {topic}", TaskType.RESEARCH)
    
    # Step 2: Analysis (high-quality model)
    analysis = await self.process_task(f"Analyze: {research.response}", TaskType.COMPLEX_ANALYSIS)
    
    # Step 3: Summary (cost-effective model)
    summary = await self.process_task(f"Summarize: {analysis.response}", TaskType.SUMMARIZATION)
    
    return {
        'research': research,
        'analysis': analysis,
        'summary': summary,
        'total_cost': research.cost + analysis.cost + summary.cost
    }
```

### Multi-Step Workflows (LangGraph)

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, List, Dict, Any

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
    routing_constraints=RoutingConstraints(min_quality_score=0.90)
)

writer_llm = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000",
    routing_constraints=RoutingConstraints(
        min_quality_score=0.80,
        max_input_cost_per_1k=2.0
    )
)

# Build workflow with different routing per step
workflow = StateGraph(ContentWorkflowState)
workflow.add_node("research", research_node)
workflow.add_node("draft", draft_node)
workflow.add_node("edit", edit_node)

workflow.add_edge(START, "research")
workflow.add_edge("research", "draft")
workflow.add_edge("draft", "edit")
workflow.add_edge("edit", END)

content_pipeline = workflow.compile()

# Run with cost tracking
result = await content_pipeline.ainvoke({"topic": "The Future of Renewable Energy"})
```

### Conditional Routing

```python
async def adaptive_process(self, query: str, budget: float):
    """Route based on available budget."""
    
    if budget < 0.001:  # Very low budget
        policy = RoutingPolicy(cost_priority=CostPriority.MINIMIZE_COST)
    elif budget > 0.01:  # High budget
        policy = RoutingPolicy(cost_priority=CostPriority.MAXIMIZE_QUALITY)
    else:  # Balanced
        policy = RoutingPolicy(cost_priority=CostPriority.BALANCE)
    
    # Create router with budget-appropriate policy
    router = ModelRouter(self.model_registry, policy)
    # ... process with custom router
```

## 🤝 Contributing

To add new agent examples:

1. Create new file in `examples/` directory
2. Follow the existing pattern for imports and structure
3. Add documentation to this README
4. Test with different IntelliRouter configurations
5. Consider both traditional and LangGraph approaches

## 📄 License

These examples are part of the IntelliRouter project and follow the same license terms.

---

**Ready to build intelligent agents with optimal AI model routing! 🚀**

*Now featuring comprehensive LangGraph integration for advanced multi-agent workflows with per-agent routing strategies!*
