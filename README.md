# IntelliRouter - Intelligent AI Model Router

An intelligent AI model routing system that automatically selects the optimal model based on complexity, cost, and performance requirements.

## 🌟 Features

- **Intelligent Model Routing**: Automatically routes queries to the most cost-effective model while maintaining quality
- **OpenAI-Compatible API**: Drop-in replacement for existing OpenAI integrations
- **Multi-Provider Support**: OpenAI, Anthropic, Google, AWS Bedrock, Azure OpenAI, and self-hosted models
- **Real-time Analytics**: Detailed metrics on usage, costs, and routing decisions
- **Fallback Management**: Automatic fallback with circuit breaker patterns
- **Streaming Support**: Full streaming support for real-time applications
- **LangGraph Integration**: Native support for multi-agent workflows with per-agent routing strategies
- **MCP Integration**: Ready for Model Context Protocol integration

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd intellirouter-project
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
SECRET_KEY=your-super-secret-key
OPENAI_API_KEY=sk-your-openai-api-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
```

### 3. Run the Server

```bash
# Option 1: Using uvicorn (recommended for development)
uvicorn src.intellirouter.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

# Option 2: Using main.py
python main.py
```

The server will start on `http://localhost:8000`

### 4. Test with Examples

```bash
# Basic functionality test
python example.py

# Intelligent agent example
python examples/intelligent_agent_example.py

# LangGraph integration examples (NEW!)
python examples/langgraph_examples.py
```

## 📚 API Usage

### Chat Completions (OpenAI Compatible)

```python
import openai

# Point to your IntelliRouter instance
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="auto",  # Let IntelliRouter choose the best model
    messages=[
        {"role": "user", "content": "Hello! How can I optimize my AI costs?"}
    ]
)

print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Response with Router Information

IntelliRouter responses include detailed routing information:

```json
{
  "id": "chatcmpl-123",
  "choices": [...],
  "usage": {...},
  "router_info": {
    "selected_model": "anthropic.claude-3-haiku-20240307-v1:0",
    "reason": "Selected for: Excellent cost efficiency, High reliability",
    "latency_ms": 551,
    "cost": 0.006000,
    "savings_vs_baseline": 0.024000,
    "baseline_model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "classification": {
      "task_type": "chat",
      "complexity_level": "low",
      "estimated_tokens": 45
    }
  }
}
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    IntelliRouter System                     │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Application                                        │
│  ├── /v1/chat/completions (OpenAI Compatible)              │
│  ├── /v1/models (List Available Models)                    │
│  └── /health (Health Check)                                │
├─────────────────────────────────────────────────────────────┤
│  Model Router (Core Intelligence)                          │
│  ├── Prompt Classifier                                     │
│  │   ├── Task Type Detection                               │
│  │   ├── Complexity Analysis                               │
│  │   └── Capability Requirements                           │
│  ├── Policy Engine                                         │
│  │   ├── Model Scoring                                     │
│  │   ├── Constraint Satisfaction                           │
│  │   └── Cost Optimization                                 │
│  └── Fallback Manager                                      │
│      ├── Circuit Breaker                                   │
│      └── Health Monitoring                                 │
├─────────────────────────────────────────────────────────────┤
│  LangGraph Integration                            │
│  ├── IntelliRouterLLM (LangChain Compatible)              │
│  ├── Multi-Agent Workflows                                 │
│  ├── Adaptive Complexity Routing                           │
│  └── Custom Constraint Support                             │
├─────────────────────────────────────────────────────────────┤
│  Provider Layer                                             │
│  ├── OpenAI Provider                                       │
│  ├── Anthropic Provider                                    │
│  ├── AWS Bedrock Provider                                  │
│  ├── Azure OpenAI Provider                                 │
│  └── Provider Factory                                      │
└─────────────────────────────────────────────────────────────┘
```

## 🧠 How It Works

1. **Request Analysis**: Incoming requests are analyzed for task type, complexity, and requirements
2. **Model Selection**: The policy engine scores available models based on:
   - Quality scores for the detected task type
   - Cost efficiency
   - Latency requirements
   - Reliability scores
3. **Smart Routing**: The highest-scoring model that meets constraints is selected
4. **Execution**: Request is routed to the selected provider with automatic fallback
5. **Analytics**: Detailed metrics are captured for cost analysis and optimization

## 💡 Routing Intelligence

### Task Type Detection
- **Chat**: General conversation and Q&A
- **Reasoning**: Complex analysis and problem-solving
- **Coding**: Programming and code generation
- **Creative Writing**: Stories, poems, creative content
- **Translation**: Language translation tasks
- **Summarization**: Text summarization and extraction
- **Structured Output**: JSON, XML, formatted data

### Complexity Analysis
- Token count analysis
- Conversation length
- Question complexity
- Technical terminology detection
- Multi-part request detection

### Cost Optimization
- Automatic model selection based on cost/quality tradeoffs
- Baseline cost comparison
- Real-time savings calculation
- Configurable cost priorities (minimize cost, balance, maximize quality)

## 📊 Analytics & Monitoring

Access detailed analytics at `/health`:
- Server health status
- Available models
- Real-time routing performance

## 🔧 Configuration

### Routing Policy

Customize routing behavior:

```python
from intellirouter.models.routing import RoutingPolicy, RoutingConstraints

constraints = RoutingConstraints(
    min_quality_score=0.85,
    max_input_cost_per_1k=1.0,
    max_latency_ms=5000,
    required_capabilities=["tools"]
)
```

### Model Registry

The system automatically detects and configures available models based on your API keys in the `.env` file.

## 🤝 Integration Examples

### LangGraph Integration (Featured!)

IntelliRouter includes comprehensive LangGraph integration for building intelligent multi-agent workflows:

#### Quick Example
```python
from intellirouter.integrations.langgraph import IntelliRouterLLM
from langchain_core.messages import HumanMessage

# Create an IntelliRouter-powered LLM
llm = IntelliRouterLLM(intellirouter_url="http://localhost:8000")

# Use it like any LangChain LLM
response = llm.invoke([HumanMessage(content="What is the capital of France?")])
print(response.content)  # "The capital of France is Paris."

# Access routing information
router_info = response.response_metadata.get('router_info', {})
print(f"Model used: {router_info.get('selected_model')}")
print(f"Cost: ${router_info.get('cost', 0):.6f}")
```

#### Multi-Agent Workflows
```python
from intellirouter.integrations.langgraph import create_multi_agent_graph

# Create a multi-agent research workflow
graph = create_multi_agent_graph(intellirouter_url="http://localhost:8000")

# Run the workflow
result = await graph.ainvoke({
    "query": "What are the benefits and risks of renewable energy?"
})

print(f"Research: {result['research_result']}")
print(f"Summary: {result['summary']}")
print(f"Validation: {result['validation_result']}")
print(f"Total cost: ${sum(info.get('cost', 0) for info in result['router_infos']):.6f}")
```

#### Adaptive Complexity Routing
```python
from intellirouter.integrations.langgraph import create_adaptive_graph

# Create an adaptive workflow that routes based on task complexity
graph = create_adaptive_graph(intellirouter_url="http://localhost:8000")

# Simple task -> uses cost-effective models
result = await graph.ainvoke({"query": "What's 2+2?"})

# Complex task -> uses high-quality models
result = await graph.ainvoke({
    "query": "Analyze the economic implications of artificial intelligence"
})
```

#### Custom Routing Constraints
```python
from intellirouter.integrations.langgraph import IntelliRouterLLM
from intellirouter.models.routing import RoutingConstraints

# Budget-conscious agent
budget_llm = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000",
    routing_constraints=RoutingConstraints(
        max_input_cost_per_1k=0.5,  # Very cost-conscious
        max_latency_ms=5000
    )
)

# Quality-first agent
quality_llm = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000",
    routing_constraints=RoutingConstraints(
        min_quality_score=0.90,  # High quality required
        required_capabilities=["tools"]
    )
)

# Speed-first agent
speed_llm = IntelliRouterLLM(
    intellirouter_url="http://localhost:8000",
    routing_constraints=RoutingConstraints(
        max_latency_ms=2000,  # Fast response required
        min_quality_score=0.65
    )
)
```

#### Running LangGraph Examples

```bash
# Make sure IntelliRouter server is running
uvicorn src.intellirouter.api.app:create_app --factory --host 0.0.0.0 --port 8000 &

# Run comprehensive LangGraph examples
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
```

**Key LangGraph Features:**
- **Multi-Agent Workflows**: Different agents use optimal models for their tasks
- **Adaptive Routing**: Complexity-based model selection within workflows
- **Cost Optimization**: Automatic cost optimization across workflow steps
- **Custom Constraints**: Per-agent routing policies (budget, quality, speed)
- **Analytics Integration**: Track costs and performance across complex workflows
- **LangChain Compatible**: Full BaseChatModel implementation
- **Async Support**: Both sync and async execution

### CrewAI Integration

```python
from crewai import Agent
import openai

# Point CrewAI to IntelliRouter
openai.api_base = "http://localhost:8000/v1"

agent = Agent(
    role="Data Analyst",
    goal="Analyze data efficiently",
    llm="auto"  # Let IntelliRouter choose the best model
)
```

### LangChain Integration

```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    openai_api_base="http://localhost:8000/v1",
    model_name="auto"
)
```

## 🛠️ Development

### Project Structure
```
intellirouter-project/
├── src/intellirouter/
│   ├── api/                    # FastAPI application
│   ├── core/                   # Core routing logic
│   ├── models/                 # Data models
│   ├── providers/              # AI provider implementations
│   ├── integrations/           # Framework integrations
│   │   └── langgraph.py       # LangGraph integration (NEW!)
│   └── config.py              # Configuration
├── examples/
│   ├── intelligent_agent_example.py  # Basic agent example
│   └── langgraph_examples.py         # LangGraph examples (NEW!)
├── main.py                    # Server entry point
├── example.py                 # Basic usage examples
└── requirements.txt           # Dependencies
```

### Running Tests
```bash
# Test basic functionality
python example.py

# Test intelligent agent
python examples/intelligent_agent_example.py

# Test LangGraph integration
python examples/langgraph_examples.py

# Test specific providers
python test_simple.py
python test_bedrock.py
```

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Start development server
uvicorn src.intellirouter.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

## 📈 Performance

- Sub-100ms routing decisions
- Concurrent request handling
- Automatic load balancing
- Circuit breaker pattern for resilience
- Real-time cost optimization

## 🔒 Security

- API key authentication
- Request validation
- Rate limiting (configurable)
- Configurable CORS policies
- Secure provider credential management

This project is provided as-is for educational and development purposes.

## 🤝 Contributing

Contributions are welcome! Please see the contributing guidelines for details.

## 🔗 Links

- **LangGraph Examples**: Run `python examples/langgraph_examples.py` for comprehensive demos
- **Health Check**: Visit `http://localhost:8000/health` when server is running
- **OpenAI Compatible API**: `http://localhost:8000/v1/chat/completions`

---

**IntelliRouter** - Intelligent AI model routing for cost optimization and performance 🚀

*Now with comprehensive LangGraph integration for multi-agent workflows!*
