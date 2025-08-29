"""LangGraph + IntelliRouter integration examples."""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from intellirouter.integrations.langgraph import (
    IntelliRouterLLM,
    create_simple_chat_graph,
    create_multi_agent_graph,
    create_adaptive_graph
)
from intellirouter.models.routing import RoutingConstraints
from langchain_core.messages import HumanMessage


async def example_1_simple_chat():
    """Example 1: Simple chat using IntelliRouter with LangGraph."""
    print("🤖 Example 1: Simple Chat with LangGraph + IntelliRouter")
    print("=" * 60)
    
    try:
        # Create a simple chat graph
        graph = create_simple_chat_graph(
            intellirouter_url="http://localhost:8000"
        )
        
        # Test query
        initial_state = {
            "messages": [HumanMessage(content="What is the capital of France?")]
        }
        
        # Run the graph
        result = await graph.ainvoke(initial_state)
        
        print(f"🤖 Response: {result['messages'][-1].content}")
        if result.get('router_info'):
            router_info = result['router_info']
            print(f"📊 Model used: {router_info.get('selected_model', 'unknown')}")
            print(f"💰 Cost: ${router_info.get('cost', 0):.6f}")
            print(f"⏱️  Latency: {router_info.get('latency_ms', 0)}ms")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure IntelliRouter server is running: python main.py")


async def example_2_multi_agent():
    """Example 2: Multi-agent workflow with different routing strategies."""
    print("\n🧠 Example 2: Multi-Agent Research Workflow")
    print("=" * 60)
    
    try:
        # Create multi-agent graph
        graph = create_multi_agent_graph(
            intellirouter_url="http://localhost:8000"
        )
        
        # Research query
        initial_state = {
            "query": "What are the benefits and risks of renewable energy?"
        }
        
        # Run the multi-agent workflow
        result = await graph.ainvoke(initial_state)
        
        print(f"🔍 Research Result: {result['research_result'][:200]}...")
        print(f"📝 Summary: {result['summary'][:200]}...")
        print(f"✅ Validation: {result['validation_result']}")
        
        print("\n📊 Routing Information by Agent:")
        for router_info in result.get('router_infos', []):
            agent = router_info.get('agent', 'unknown')
            model = router_info.get('selected_model', 'unknown')
            cost = router_info.get('cost', 0)
            print(f"  • {agent.title()}: {model} (${cost:.6f})")
        
        # Calculate total cost
        total_cost = sum(info.get('cost', 0) for info in result.get('router_infos', []))
        print(f"💰 Total workflow cost: ${total_cost:.6f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_3_adaptive_routing():
    """Example 3: Adaptive routing based on task complexity."""
    print("\n🎯 Example 3: Adaptive Complexity-Based Routing")
    print("=" * 60)
    
    try:
        # Create adaptive graph
        graph = create_adaptive_graph(
            intellirouter_url="http://localhost:8000"
        )
        
        # Test different complexity queries
        test_queries = [
            "What's 2+2?",  # Simple
            "Analyze the economic implications of artificial intelligence on employment markets, considering both short-term disruptions and long-term societal benefits."  # Complex
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query[:50]}...")
            
            initial_state = {"query": query}
            result = await graph.ainvoke(initial_state)
            
            print(f"🏷️  Classified as: {result['complexity']}")
            print(f"🤖 Response: {result['response'][:150]}...")
            
            print("📊 Model usage by stage:")
            for router_info in result.get('router_infos', []):
                stage = router_info.get('stage', 'unknown')
                model = router_info.get('selected_model', 'unknown')
                cost = router_info.get('cost', 0)
                print(f"  • {stage}: {model} (${cost:.6f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_4_custom_constraints():
    """Example 4: Custom routing constraints for specific use cases."""
    print("\n⚙️  Example 4: Custom Routing Constraints")
    print("=" * 60)
    
    try:
        # Create LLMs with different constraints
        
        # Budget-conscious LLM
        budget_llm = IntelliRouterLLM(
            intellirouter_url="http://localhost:8000",
            routing_constraints=RoutingConstraints(
                max_input_cost_per_1k=0.5,  # Very cost-conscious
                max_latency_ms=5000
            )
        )
        
        # Quality-first LLM
        quality_llm = IntelliRouterLLM(
            intellirouter_url="http://localhost:8000",
            routing_constraints=RoutingConstraints(
                min_quality_score=0.90,  # High quality required
                required_capabilities=["tools"]
            )
        )
        
        # Speed-first LLM
        speed_llm = IntelliRouterLLM(
            intellirouter_url="http://localhost:8000",
            routing_constraints=RoutingConstraints(
                max_latency_ms=2000,  # Fast response required
                min_quality_score=0.65
            )
        )
        
        test_query = "Explain quantum computing in simple terms"
        message = [HumanMessage(content=test_query)]
        
        # Test each LLM
        llms = [
            ("Budget-Conscious", budget_llm),
            ("Quality-First", quality_llm),
            ("Speed-First", speed_llm)
        ]
        
        for name, llm in llms:
            print(f"\n🔧 {name} LLM:")
            try:
                result = llm.invoke(message)
                router_info = getattr(result, 'response_metadata', {}).get('router_info', {})
                
                print(f"  Model: {router_info.get('selected_model', 'unknown')}")
                print(f"  Cost: ${router_info.get('cost', 0):.6f}")
                print(f"  Latency: {router_info.get('latency_ms', 0)}ms")
                print(f"  Reason: {router_info.get('reason', 'No reason provided')}")
                
            except Exception as llm_error:
                print(f"  ❌ Error: {llm_error}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_5_cost_comparison():
    """Example 5: Cost comparison between different routing strategies."""
    print("\n💰 Example 5: Cost Comparison Across Strategies")
    print("=" * 60)
    
    try:
        # Create different LLMs for comparison
        always_premium = IntelliRouterLLM(
            intellirouter_url="http://localhost:8000",
            routing_constraints=RoutingConstraints(
                min_quality_score=0.95  # Force premium models
            )
        )
        
        auto_route = IntelliRouterLLM(
            intellirouter_url="http://localhost:8000"
            # No constraints - let IntelliRouter choose optimally
        )
        
        budget_only = IntelliRouterLLM(
            intellirouter_url="http://localhost:8000",
            routing_constraints=RoutingConstraints(
                max_input_cost_per_1k=0.3  # Force cheap models
            )
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
                    router_info = getattr(result, 'response_metadata', {}).get('router_info', {})
                    
                    cost = router_info.get('cost', 0)
                    model = router_info.get('selected_model', 'unknown')
                    total_costs[strategy_name] += cost
                    
                    print(f"  {strategy_name}: {model} (${cost:.6f})")
                    
                except Exception as strategy_error:
                    print(f"  {strategy_name}: ❌ {strategy_error}")
        
        # Show total costs
        print(f"\n💰 Total Costs Comparison:")
        for strategy_name, total_cost in total_costs.items():
            print(f"  {strategy_name}: ${total_cost:.6f}")
        
        # Calculate savings
        if total_costs["Always Premium"] > 0:
            auto_savings = ((total_costs["Always Premium"] - total_costs["Auto Route"]) / 
                          total_costs["Always Premium"]) * 100
            print(f"\n💵 Auto Route saves {auto_savings:.1f}% vs Always Premium")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run all LangGraph examples."""
    print("🚀 IntelliRouter + LangGraph Integration Examples")
    print("=" * 60)
    print("📋 Prerequisites:")
    print("1. IntelliRouter server running: python main.py")
    print("2. At least one AI provider configured (OpenAI/Anthropic)")
    print("3. LangGraph dependencies installed")
    print()
    
    # Run examples
    await example_1_simple_chat()
    await example_2_multi_agent()
    await example_3_adaptive_routing()
    await example_4_custom_constraints()
    await example_5_cost_comparison()
    
    print("\n🎉 All examples completed!")
    print("\n💡 Key Benefits of LangGraph + IntelliRouter:")
    print("• Intelligent model selection at each workflow step")
    print("• Different routing strategies for different agents")
    print("• Automatic cost optimization across complex workflows")
    print("• Adaptive routing based on task complexity")
    print("• Detailed cost and performance analytics")


if __name__ == "__main__":
    asyncio.run(main())
