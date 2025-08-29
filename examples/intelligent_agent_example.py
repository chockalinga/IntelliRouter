"""
IntelliRouter Agent Example
==========================

This example demonstrates how to create an intelligent agent using IntelliRouter
that automatically selects the optimal AI model based on task complexity and requirements.

The agent handles different types of tasks:
- Simple Q&A (uses cost-effective models)
- Code generation (uses coding-optimized models)
- Complex analysis (uses high-quality models)
- Creative writing (uses creativity-optimized models)
"""

import asyncio
import sys
import os
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from intellirouter.models.chat import ChatCompletionRequest, ChatMessage
from intellirouter.core.router import ModelRouter
from intellirouter.api.app import create_sample_model_registry
from intellirouter.models.routing import RoutingPolicy, CostPriority


class TaskType(Enum):
    """Types of tasks the agent can handle."""
    SIMPLE_QA = "simple_qa"
    CODE_GENERATION = "code_generation"
    COMPLEX_ANALYSIS = "complex_analysis"
    CREATIVE_WRITING = "creative_writing"
    RESEARCH = "research"
    SUMMARIZATION = "summarization"


@dataclass
class TaskResult:
    """Result from agent task execution."""
    task_type: TaskType
    response: str
    model_used: str
    cost: float
    latency_ms: int
    reasoning: str


class IntelliRouterAgent:
    """
    An intelligent agent that uses IntelliRouter for optimal model selection.
    
    The agent automatically classifies tasks and routes them to the most
    appropriate AI model based on cost, quality, and performance requirements.
    """
    
    def __init__(self):
        """Initialize the agent with IntelliRouter."""
        # Create model registry and router
        self.model_registry = create_sample_model_registry()
        self.policy = RoutingPolicy(
            cost_priority=CostPriority.BALANCE,  # Balance cost and quality
            quality_weight=0.4,
            cost_weight=0.3,
            latency_weight=0.2
        )
        self.router = ModelRouter(self.model_registry, self.policy)
        
        # Task history for learning
        self.task_history: List[TaskResult] = []
        
    async def process_task(self, user_input: str, task_type: TaskType = None) -> TaskResult:
        """
        Process a user task with intelligent model routing.
        
        Args:
            user_input: The user's task/question
            task_type: Optional manual task type (auto-detected if None)
            
        Returns:
            TaskResult with response and routing information
        """
        # Auto-detect task type if not provided
        if task_type is None:
            task_type = self._classify_task(user_input)
        
        # Create optimized prompt for the task type
        optimized_prompt = self._optimize_prompt_for_task(user_input, task_type)
        
        # Create chat request
        request = ChatCompletionRequest(
            model="auto",  # Let IntelliRouter choose
            messages=[ChatMessage(role="user", content=optimized_prompt)]
        )
        
        # Route and execute
        try:
            response = await self.router.route_request(request)
            
            # Create result
            result = TaskResult(
                task_type=task_type,
                response=response.choices[0].message.content,
                model_used=response.router_info.selected_model,
                cost=response.router_info.cost,
                latency_ms=response.router_info.latency_ms,
                reasoning=response.router_info.reason
            )
            
            # Add to history for analytics
            self.task_history.append(result)
            
            return result
            
        except Exception as e:
            # Fallback result in case of error
            return TaskResult(
                task_type=task_type,
                response=f"Sorry, I encountered an error: {e}",
                model_used="error",
                cost=0.0,
                latency_ms=0,
                reasoning="Error occurred during processing"
            )
    
    def _classify_task(self, user_input: str) -> TaskType:
        """
        Classify the task type based on user input.
        
        This is a simple heuristic-based classifier. In production,
        you might use a more sophisticated classification model.
        """
        user_input_lower = user_input.lower()
        
        # Code-related keywords
        code_keywords = ['python', 'code', 'function', 'algorithm', 'programming', 
                        'script', 'debug', 'implement', 'class', 'method']
        if any(keyword in user_input_lower for keyword in code_keywords):
            return TaskType.CODE_GENERATION
        
        # Analysis keywords
        analysis_keywords = ['analyze', 'analysis', 'compare', 'evaluate', 'assess',
                           'examine', 'investigate', 'study', 'research', 'implications']
        if any(keyword in user_input_lower for keyword in analysis_keywords):
            return TaskType.COMPLEX_ANALYSIS
        
        # Creative keywords
        creative_keywords = ['story', 'poem', 'creative', 'write', 'imagine',
                           'fiction', 'character', 'plot', 'narrative']
        if any(keyword in user_input_lower for keyword in creative_keywords):
            return TaskType.CREATIVE_WRITING
        
        # Research keywords
        research_keywords = ['research', 'find information', 'what is', 'explain',
                           'tell me about', 'history of', 'facts about']
        if any(keyword in user_input_lower for keyword in research_keywords):
            return TaskType.RESEARCH
        
        # Summarization keywords
        summary_keywords = ['summarize', 'summary', 'brief', 'outline', 'key points']
        if any(keyword in user_input_lower for keyword in summary_keywords):
            return TaskType.SUMMARIZATION
        
        # Default to simple Q&A for short, simple questions
        if len(user_input.split()) < 10:
            return TaskType.SIMPLE_QA
        
        return TaskType.COMPLEX_ANALYSIS  # Default for longer, complex inputs
    
    def _optimize_prompt_for_task(self, user_input: str, task_type: TaskType) -> str:
        """
        Optimize the prompt based on the task type to get better results.
        """
        task_prompts = {
            TaskType.SIMPLE_QA: f"{user_input}",
            
            TaskType.CODE_GENERATION: f"""
As an expert programmer, please help with this coding task:
{user_input}

Please provide:
1. Clean, well-commented code
2. Brief explanation of the approach
3. Any important considerations
""",
            
            TaskType.COMPLEX_ANALYSIS: f"""
Please provide a comprehensive analysis of the following:
{user_input}

Structure your response with:
1. Key findings
2. Supporting evidence/reasoning
3. Implications and conclusions
4. Potential limitations or considerations
""",
            
            TaskType.CREATIVE_WRITING: f"""
Creative writing request:
{user_input}

Please create engaging, original content that:
- Is well-structured and flows naturally
- Uses vivid descriptions and engaging language
- Captures the requested tone and style
""",
            
            TaskType.RESEARCH: f"""
Research query:
{user_input}

Please provide:
1. Accurate, up-to-date information
2. Multiple perspectives where relevant
3. Key sources or references
4. Clear, organized presentation
""",
            
            TaskType.SUMMARIZATION: f"""
Please provide a concise summary of:
{user_input}

Focus on:
1. Main points and key information
2. Clear, organized structure
3. Appropriate level of detail
"""
        }
        
        return task_prompts.get(task_type, user_input)
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics about agent performance."""
        if not self.task_history:
            return {"message": "No tasks processed yet"}
        
        # Calculate statistics
        total_tasks = len(self.task_history)
        total_cost = sum(task.cost for task in self.task_history)
        avg_cost = total_cost / total_tasks
        avg_latency = sum(task.latency_ms for task in self.task_history) / total_tasks
        
        # Model usage distribution
        model_usage = {}
        for task in self.task_history:
            model = task.model_used
            model_usage[model] = model_usage.get(model, 0) + 1
        
        # Task type distribution
        task_type_usage = {}
        for task in self.task_history:
            task_type = task.task_type.value
            task_type_usage[task_type] = task_type_usage.get(task_type, 0) + 1
        
        return {
            "total_tasks": total_tasks,
            "total_cost": f"${total_cost:.6f}",
            "average_cost_per_task": f"${avg_cost:.6f}",
            "average_latency_ms": f"{avg_latency:.0f}ms",
            "model_usage": model_usage,
            "task_type_distribution": task_type_usage,
            "latest_tasks": [
                {
                    "task_type": task.task_type.value,
                    "model_used": task.model_used,
                    "cost": f"${task.cost:.6f}",
                    "latency_ms": f"{task.latency_ms}ms"
                }
                for task in self.task_history[-5:]  # Last 5 tasks
            ]
        }


async def demo_agent():
    """Demonstrate the IntelliRouter Agent with various tasks."""
    
    print("🤖 IntelliRouter Agent Demo")
    print("=" * 50)
    print("Demonstrating intelligent model routing for different task types\n")
    
    # Create agent
    agent = IntelliRouterAgent()
    
    # Demo tasks of different types
    demo_tasks = [
        ("What is the capital of France?", TaskType.SIMPLE_QA),
        
        ("Write a Python function to calculate the fibonacci sequence", TaskType.CODE_GENERATION),
        
        ("Analyze the environmental and economic impacts of renewable energy adoption in developing countries", TaskType.COMPLEX_ANALYSIS),
        
        ("Write a short story about a robot who discovers emotions", TaskType.CREATIVE_WRITING),
        
        ("Research the history and key features of quantum computing", TaskType.RESEARCH),
    ]
    
    # Process each task
    for i, (task_input, expected_type) in enumerate(demo_tasks, 1):
        print(f"\n📝 Task {i}: {expected_type.value.replace('_', ' ').title()}")
        print(f"Input: {task_input[:60]}...")
        
        try:
            result = await agent.process_task(task_input, expected_type)
            
            print(f"✅ Model Selected: {result.model_used}")
            print(f"💡 Reasoning: {result.reasoning}")
            print(f"💰 Cost: ${result.cost:.6f}")
            print(f"⏱️  Latency: {result.latency_ms}ms")
            print(f"🔍 Response Preview: {result.response[:100]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)
    
    # Show analytics
    print(f"\n📊 Agent Analytics:")
    analytics = agent.get_analytics()
    
    print(f"Total Tasks: {analytics['total_tasks']}")
    print(f"Total Cost: {analytics['total_cost']}")
    print(f"Average Cost per Task: {analytics['average_cost_per_task']}")
    print(f"Average Latency: {analytics['average_latency_ms']}")
    
    print(f"\n🤖 Model Usage:")
    for model, count in analytics['model_usage'].items():
        print(f"  • {model}: {count} tasks")
    
    print(f"\n📋 Task Type Distribution:")
    for task_type, count in analytics['task_type_distribution'].items():
        print(f"  • {task_type.replace('_', ' ').title()}: {count} tasks")


async def interactive_agent():
    """Run an interactive agent session."""
    print("\n🤖 Interactive IntelliRouter Agent")
    print("=" * 40)
    print("Type your tasks and I'll route them intelligently!")
    print("Commands: 'quit' to exit, 'stats' for analytics")
    print("-" * 40)
    
    agent = IntelliRouterAgent()
    
    while True:
        try:
            user_input = input("\n💭 Your task: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() in ['stats', 'analytics']:
                analytics = agent.get_analytics()
                print(f"\n📊 Analytics: {analytics}")
                continue
            
            if not user_input:
                print("Please enter a task!")
                continue
            
            print("🔄 Processing...")
            result = await agent.process_task(user_input)
            
            print(f"\n🤖 Agent Response:")
            print(f"Task Type: {result.task_type.value.replace('_', ' ').title()}")
            print(f"Model Used: {result.model_used}")
            print(f"Cost: ${result.cost:.6f}")
            print(f"Latency: {result.latency_ms}ms")
            print(f"\n📝 Response:\n{result.response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Main function to run agent examples."""
    print("🚀 IntelliRouter Agent Examples")
    print("=" * 50)
    
    choice = input("""
Choose an option:
1. Run demo with sample tasks
2. Interactive agent session
3. Both

Enter choice (1/2/3): """).strip()
    
    if choice in ['1', '3']:
        await demo_agent()
    
    if choice in ['2', '3']:
        await interactive_agent()
    
    if choice not in ['1', '2', '3']:
        print("Running demo by default...")
        await demo_agent()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Agent terminated by user")
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        print("\nMake sure IntelliRouter dependencies are installed:")
        print("pip install -r requirements.txt")
