# 🚀 IntelliRouter Quick Start Guide

This guide shows you exactly how to set up and test IntelliRouter with real examples.

## 📋 Prerequisites

- Python 3.8+
- At least one AI provider API key (OpenAI, Anthropic, or AWS Bedrock recommended)

## 🛠️ Setup (5 minutes)

### Step 1: Install Dependencies

```bash
cd intellirouter-project
pip install -r requirements.txt
```

### Step 2: Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your API keys
# You need at least one of these:
```

**Option A: OpenAI only**
```env
SECRET_KEY=my-secret-key-123
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

**Option B: Anthropic only**  
```env
SECRET_KEY=my-secret-key-123
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key-here
```

**Option C: AWS Bedrock (recommended)**
```env
SECRET_KEY=my-secret-key-123
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
```

**Option D: All providers (best)**
```env
SECRET_KEY=my-secret-key-123
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key-here
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
```

## 🧪 Testing Options

### Option 1: Quick Demo Script (Easiest)

```bash
python example.py
```

**Expected Output:**
```
🚀 IntelliRouter Example
==================================================
📊 Loaded 3 models:
  • GPT-4o Mini (openai) - $0.150/$0.600 per 1K tokens
  • Claude 3 Haiku (anthropic) - $0.250/$1.250 per 1K tokens
  • Claude 3 Haiku (bedrock) - $0.250/$1.250 per 1K tokens

==================================================
💬 Example 1: Simple Chat
✅ Selected Model: anthropic.claude-3-haiku-20240307-v1:0
💡 Reason: Selected for: Excellent cost efficiency, High reliability
💰 Cost: $0.006000
⏱️  Latency: 551ms
📝 Response: Hello! I'd be happy to help you with any question...
```

### Option 2: Start the API Server

**Terminal 1 - Start Server (Fixed Command!):**
```bash
# Option 1: Using uvicorn (recommended for development)
uvicorn src.intellirouter.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

# Option 2: Using main.py (may have issues)
python main.py
```

**Expected Output:**
```
INFO:     Will watch for changes in these directories: ['/Users/user/intellirouter-project']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using WatchFiles
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Terminal 2 - Test with cURL:**
```bash
# Test basic health
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/v1/models

# Simple chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Option 3: Advanced Examples (NEW!)

**Intelligent Agent Example:**
```bash
python examples/intelligent_agent_example.py
```

**LangGraph Integration Examples:**
```bash
python examples/langgraph_examples.py
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
```

### Option 4: Python Client Example

Create `test_client.py`:

```python
import requests
import json

# Test the API
url = "http://localhost:8000/v1/chat/completions"

# Simple chat
response = requests.post(url, json={
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
})

result = response.json()
print("Selected Model:", result['router_info']['selected_model'])
print("Response:", result['choices'][0]['message']['content'])
print("Cost:", f"${result['router_info']['cost']:.6f}")
```

```bash
python test_client.py
```

### Option 5: OpenAI Client Drop-in Replacement

```python
import openai

# Point OpenAI client to IntelliRouter
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # Not required for demo
)

# Use exactly like OpenAI
response = client.chat.completions.create(
    model="auto",  # Let IntelliRouter choose
    messages=[
        {"role": "user", "content": "Write a Python function to calculate fibonacci"}
    ]
)

print(f"Model used: {response.router_info.selected_model}")
print(f"Reason: {response.router_info.reason}")
print(f"Response: {response.choices[0].message.content}")
```

## 🎯 Test Different Scenarios

### 1. Simple Questions (Should use cheap model)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

### 2. Complex Coding Task (Should use better model)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto", 
    "messages": [{"role": "user", "content": "Write a complex algorithm to solve the traveling salesman problem with dynamic programming and explain the time complexity"}]
  }'
```

### 3. Streaming Response
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Tell me a short story"}],
    "stream": true
  }'
```

### 4. Check Health Status
```bash
curl http://localhost:8000/health
```

## 📊 Understanding the Response

IntelliRouter responses include detailed routing information:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant", 
        "content": "The actual AI response here..."
      }
    }
  ],
  "router_info": {
    "selected_model": "anthropic.claude-3-haiku-20240307-v1:0",
    "reason": "Selected for: Excellent cost efficiency, High reliability", 
    "latency_ms": 551,
    "cost": 0.006000,
    "savings_vs_baseline": 0.024000,
    "classification": {
      "task_type": "chat",
      "complexity_level": "low"
    }
  }
}
```

**Key Fields:**
- `selected_model`: Which model was chosen
- `reason`: Why this model was selected
- `cost`: Actual cost of the request
- `savings_vs_baseline`: Money saved vs always using premium model
- `classification`: How IntelliRouter analyzed your prompt

## 🔍 What to Look For

1. **Cost Optimization**: Simple questions should use cheaper models
2. **Quality Routing**: Complex tasks should use more capable models  
3. **Savings**: Check `savings_vs_baseline` to see money saved
4. **Classification**: See how prompts are analyzed for task type and complexity

## 🚨 Troubleshooting

### Error: "Provider openai is not configured"
- Check your `.env` file has `OPENAI_API_KEY=sk-...`
- Make sure the API key is valid

### Error: "No models available that meet constraints" 
- You need at least one provider configured
- Check your API keys are correct

### Server won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process
kill -9 $(lsof -t -i:8000)

# Use working uvicorn command
uvicorn src.intellirouter.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

### Dependencies issues
```bash
# Upgrade pip
pip install --upgrade pip

# Install specific versions
pip install -r requirements.txt --upgrade

# Fix LangGraph issues
pip install langgraph==0.6.6
```

### AWS Bedrock Configuration
```bash
# Configure AWS CLI (alternative to .env)
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key  
# Enter your region (e.g., us-east-1)
# Enter output format (json)
```

## 🎉 Advanced Features

### LangGraph Multi-Agent Workflows

Once the basic setup works, try the advanced LangGraph integration:

```python
from intellirouter.integrations.langgraph import create_multi_agent_graph

# Create workflow with different agents having different routing strategies
graph = create_multi_agent_graph(intellirouter_url="http://localhost:8000")

# Run research workflow
result = await graph.ainvoke({
    "query": "What are the benefits of renewable energy?"
})

print("Research:", result['research_result'][:100], "...")
print("Summary:", result['summary'][:100], "...")
print("Validation:", result['validation_result'])

# See total cost
total_cost = sum(info.get('cost', 0) for info in result['router_infos'])
print(f"Total cost: ${total_cost:.6f}")
```

### Custom Routing Constraints

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

response = budget_llm.invoke("Explain quantum computing")
print(f"Cost: ${response.response_metadata['router_info']['cost']:.6f}")
```

## 📝 Real Integration Examples

### With LangChain
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    model="auto",
    api_key="dummy"
)

response = llm.invoke("Explain quantum computing")
```

### With LangGraph (Advanced)
```python
from intellirouter.integrations.langgraph import IntelliRouterLLM

llm = IntelliRouterLLM(intellirouter_url="http://localhost:8000")
response = llm.invoke("What is the capital of France?")
print(response.content)
```

### With Requests
```python
import requests

def chat_with_intellirouter(message):
    response = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": message}]
    })
    
    data = response.json()
    return {
        "response": data["choices"][0]["message"]["content"],
        "model_used": data["router_info"]["selected_model"],
        "cost": data["router_info"]["cost"]
    }

result = chat_with_intellirouter("What's the weather like?")
print(f"AI: {result['response']}")
print(f"Used: {result['model_used']} (${result['cost']:.6f})")
```

## 🎯 Next Steps

Once you have it working:

1. **✅ Basic Integration**: Replace OpenAI base URL with `http://localhost:8000/v1`
2. **📊 Monitor Costs**: Check `/health` endpoint regularly
3. **🔧 Customize Routing**: Create custom routing constraints
4. **🤖 Try LangGraph**: Run `python examples/langgraph_examples.py` for advanced workflows
5. **📈 Scale Up**: Add more providers and customize policies

## 🏆 Success Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] API keys configured in `.env` file
- [ ] Server starts successfully (using uvicorn command)
- [ ] Health check works (`curl http://localhost:8000/health`)
- [ ] Basic example works (`python example.py`)
- [ ] Intelligent agent works (`python examples/intelligent_agent_example.py`)
- [ ] LangGraph integration works (`python examples/langgraph_examples.py`)

---

**🎯 Ready to save on AI costs while maintaining quality with intelligent routing! 🚀**

*Now with comprehensive LangGraph integration for multi-agent workflows!*
