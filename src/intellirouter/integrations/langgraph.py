"""LangGraph integration for IntelliRouter."""

import asyncio
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator, TypedDict
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult, Generation
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langgraph.graph import StateGraph, START, END
import requests
import json

from ..models.chat import ChatCompletionRequest, ChatMessage as IntelliRouterMessage
from ..models.routing import RoutingConstraints, RoutingPolicy


class IntelliRouterLLM(BaseChatModel):
    """LangChain-compatible LLM that uses IntelliRouter for intelligent model selection."""
    
    intellirouter_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    routing_constraints: Optional[RoutingConstraints] = None
    session_id: Optional[str] = None
    
    def __init__(
        self,
        intellirouter_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        routing_constraints: Optional[RoutingConstraints] = None,
        session_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize IntelliRouter LLM.
        
        Args:
            intellirouter_url: URL of IntelliRouter server
            api_key: Optional API key for authentication
            routing_constraints: Constraints for model routing
            session_id: Session ID for conversation continuity
        """
        super().__init__(**kwargs)
        self.intellirouter_url = intellirouter_url.rstrip("/")
        self.api_key = api_key
        self.routing_constraints = routing_constraints
        self.session_id = session_id
    
    @property
    def _llm_type(self) -> str:
        """Return identifier of llm type."""
        return "intellirouter"
    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to IntelliRouter format."""
        converted = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                role = "user"  # Default fallback
            
            converted.append({
                "role": role,
                "content": message.content
            })
        
        return converted
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response using IntelliRouter."""
        
        # Convert messages
        intellirouter_messages = self._convert_messages(messages)
        
        # Prepare request
        request_data = {
            "model": "auto",
            "messages": intellirouter_messages,
            "stream": False
        }
        
        # Add optional parameters
        if stop:
            request_data["stop"] = stop
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
                request_data[key] = value
        
        # Add session ID if provided
        if self.session_id:
            request_data["session_id"] = self.session_id
        
        # Make request to IntelliRouter
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(
                f"{self.intellirouter_url}/v1/chat/completions",
                json=request_data,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response content
            content = result["choices"][0]["message"]["content"]
            
            # Create AIMessage with router info in response_metadata
            ai_message = AIMessage(
                content=content,
                response_metadata={
                    "router_info": result.get("router_info", {}),
                    "usage": result.get("usage", {}),
                    "model": result.get("model", "unknown")
                }
            )
            
            # Create ChatGeneration with the AIMessage that has response_metadata
            generation = ChatGeneration(
                message=ai_message,
                generation_info={
                    "router_info": result.get("router_info", {}),
                    "usage": result.get("usage", {}),
                    "model": result.get("model", "unknown")
                }
            )
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            raise Exception(f"IntelliRouter request failed: {str(e)}")
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate chat response using IntelliRouter."""
        
        # For now, run sync version in thread pool
        # In production, you'd use aiohttp for async requests
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self._generate(messages, stop, None, **kwargs)
        )


# Define state classes using TypedDict for LangGraph
class ChatState(TypedDict):
    messages: List[BaseMessage]
    router_info: Optional[Dict[str, Any]]


class MultiAgentState(TypedDict):
    query: str
    research_result: Optional[str]
    summary: Optional[str]
    validation_result: Optional[str]
    router_infos: List[Dict[str, Any]]


class AdaptiveState(TypedDict):
    query: str
    complexity: Optional[str]
    response: Optional[str]
    router_infos: List[Dict[str, Any]]


def create_simple_chat_graph(
    intellirouter_url: str = "http://localhost:8000",
    api_key: Optional[str] = None
):
    """Create a simple LangGraph chat workflow using IntelliRouter.
    
    Args:
        intellirouter_url: URL of IntelliRouter server
        api_key: Optional API key
        
    Returns:
        Compiled LangGraph
    """
    
    # Initialize IntelliRouter LLM
    llm = IntelliRouterLLM(
        intellirouter_url=intellirouter_url,
        api_key=api_key
    )
    
    def chat_node(state: ChatState) -> ChatState:
        """Chat node that uses IntelliRouter."""
        result = llm.invoke(state["messages"])
        
        # Add AI response to messages
        new_messages = state["messages"] + [result]
        
        # Extract router info from response_metadata
        router_info = getattr(result, 'response_metadata', {}).get('router_info')
        
        return {
            "messages": new_messages,
            "router_info": router_info
        }
    
    # Create graph
    workflow = StateGraph(ChatState)
    workflow.add_node("chat", chat_node)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    
    return workflow.compile()


def create_multi_agent_graph(
    intellirouter_url: str = "http://localhost:8000",
    api_key: Optional[str] = None
):
    """Create a multi-agent workflow where different agents can use different routing strategies.
    
    Args:
        intellirouter_url: URL of IntelliRouter server
        api_key: Optional API key
        
    Returns:
        Compiled LangGraph
    """
    
    # Create different LLMs with different constraints for different agent types
    
    # Researcher: Prefers quality over cost
    researcher_llm = IntelliRouterLLM(
        intellirouter_url=intellirouter_url,
        api_key=api_key,
        routing_constraints=RoutingConstraints(
            min_quality_score=0.85,  # High quality requirement
            max_latency_ms=10000
        )
    )
    
    # Summarizer: Balances cost and quality
    summarizer_llm = IntelliRouterLLM(
        intellirouter_url=intellirouter_url,
        api_key=api_key,
        routing_constraints=RoutingConstraints(
            min_quality_score=0.70,
            max_input_cost_per_1k=1.0  # Cost constraint
        )
    )
    
    # Validator: Fast and cheap for simple validation
    validator_llm = IntelliRouterLLM(
        intellirouter_url=intellirouter_url,
        api_key=api_key,
        routing_constraints=RoutingConstraints(
            min_quality_score=0.60,
            max_latency_ms=3000,  # Fast response
            max_input_cost_per_1k=0.5  # Very cost-conscious
        )
    )
    
    def research_node(state: MultiAgentState) -> MultiAgentState:
        """Research node using high-quality models."""
        messages = [HumanMessage(content=f"Research this topic in detail: {state['query']}")]
        result = researcher_llm.invoke(messages)
        
        # Extract router info from response metadata
        router_info = getattr(result, 'response_metadata', {}).get('router_info', {})
        if router_info:
            router_info['agent'] = 'researcher'
        
        # Ensure router_infos exists in state
        current_infos = state.get("router_infos", [])
        
        return {
            **state,
            "research_result": result.content,
            "router_infos": current_infos + [router_info] if router_info else current_infos
        }
    
    def summarize_node(state: MultiAgentState) -> MultiAgentState:
        """Summarize node using balanced cost/quality."""
        messages = [
            HumanMessage(content=f"Summarize this research concisely: {state['research_result']}")
        ]
        result = summarizer_llm.invoke(messages)
        
        # Extract router info from response metadata
        router_info = getattr(result, 'response_metadata', {}).get('router_info', {})
        if router_info:
            router_info['agent'] = 'summarizer'
        
        # Ensure router_infos exists in state
        current_infos = state.get("router_infos", [])
        
        return {
            **state,
            "summary": result.content,
            "router_infos": current_infos + [router_info] if router_info else current_infos
        }
    
    def validate_node(state: MultiAgentState) -> MultiAgentState:
        """Validation node using fast, cheap models."""
        messages = [
            HumanMessage(content=f"Is this summary accurate and complete? Answer with just YES or NO: {state['summary']}")
        ]
        result = validator_llm.invoke(messages)
        
        # Extract router info from response metadata
        router_info = getattr(result, 'response_metadata', {}).get('router_info', {})
        if router_info:
            router_info['agent'] = 'validator'
        
        # Ensure router_infos exists in state
        current_infos = state.get("router_infos", [])
        
        return {
            **state,
            "validation_result": result.content.strip(),
            "router_infos": current_infos + [router_info] if router_info else current_infos
        }
    
    # Create graph
    workflow = StateGraph(MultiAgentState)
    workflow.add_node("research", research_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("validate", validate_node)
    
    # Define flow
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "summarize")
    workflow.add_edge("summarize", "validate")
    workflow.add_edge("validate", END)
    
    return workflow.compile()


def create_adaptive_graph(
    intellirouter_url: str = "http://localhost:8000",
    api_key: Optional[str] = None
):
    """Create an adaptive workflow that changes routing strategy based on task complexity.
    
    Args:
        intellirouter_url: URL of IntelliRouter server
        api_key: Optional API key
        
    Returns:
        Compiled LangGraph
    """
    
    # Base LLM for classification
    classifier_llm = IntelliRouterLLM(
        intellirouter_url=intellirouter_url,
        api_key=api_key,
        routing_constraints=RoutingConstraints(
            max_latency_ms=2000,  # Fast classification
            max_input_cost_per_1k=0.3
        )
    )
    
    # High-performance LLM for complex tasks
    complex_llm = IntelliRouterLLM(
        intellirouter_url=intellirouter_url,
        api_key=api_key,
        routing_constraints=RoutingConstraints(
            min_quality_score=0.90,  # Premium quality
            required_capabilities=["tools"]  # Advanced capabilities
        )
    )
    
    # Efficient LLM for simple tasks
    simple_llm = IntelliRouterLLM(
        intellirouter_url=intellirouter_url,
        api_key=api_key,
        routing_constraints=RoutingConstraints(
            min_quality_score=0.70,
            max_input_cost_per_1k=0.5,  # Cost-effective
            max_latency_ms=3000
        )
    )
    
    def classify_node(state: AdaptiveState) -> AdaptiveState:
        """Classify task complexity."""
        messages = [
            HumanMessage(content=f"""
            Classify this query's complexity as either "SIMPLE" or "COMPLEX":
            
            Query: {state['query']}
            
            SIMPLE tasks include: basic questions, simple math, casual conversation
            COMPLEX tasks include: detailed analysis, coding, research, multi-step reasoning
            
            Respond with just one word: SIMPLE or COMPLEX
            """)
        ]
        
        result = classifier_llm.invoke(messages)
        complexity = result.content.strip().upper()
        
        # Extract router info from response metadata
        router_info = getattr(result, 'response_metadata', {}).get('router_info', {})
        if router_info:
            router_info['stage'] = 'classification'
        
        # Ensure router_infos exists in state
        current_infos = state.get("router_infos", [])
        
        return {
            **state,
            "complexity": complexity,
            "router_infos": current_infos + [router_info] if router_info else current_infos
        }
    
    def simple_task_node(state: AdaptiveState) -> AdaptiveState:
        """Handle simple tasks with cost-effective models."""
        messages = [HumanMessage(content=state["query"])]
        result = simple_llm.invoke(messages)
        
        # Extract router info from response metadata
        router_info = getattr(result, 'response_metadata', {}).get('router_info', {})
        if router_info:
            router_info['stage'] = 'simple_execution'
        
        # Ensure router_infos exists in state
        current_infos = state.get("router_infos", [])
        
        return {
            **state,
            "response": result.content,
            "router_infos": current_infos + [router_info] if router_info else current_infos
        }
    
    def complex_task_node(state: AdaptiveState) -> AdaptiveState:
        """Handle complex tasks with high-performance models."""
        messages = [HumanMessage(content=state["query"])]
        result = complex_llm.invoke(messages)
        
        # Extract router info from response metadata
        router_info = getattr(result, 'response_metadata', {}).get('router_info', {})
        if router_info:
            router_info['stage'] = 'complex_execution'
        
        # Ensure router_infos exists in state
        current_infos = state.get("router_infos", [])
        
        return {
            **state,
            "response": result.content,
            "router_infos": current_infos + [router_info] if router_info else current_infos
        }
    
    def route_by_complexity(state: AdaptiveState) -> str:
        """Route based on classified complexity."""
        if state["complexity"] == "COMPLEX":
            return "complex_task"
        else:
            return "simple_task"
    
    # Create graph
    workflow = StateGraph(AdaptiveState)
    workflow.add_node("classify", classify_node)
    workflow.add_node("simple_task", simple_task_node)
    workflow.add_node("complex_task", complex_task_node)
    
    # Define conditional routing
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        route_by_complexity,
        {
            "simple_task": "simple_task",
            "complex_task": "complex_task"
        }
    )
    workflow.add_edge("simple_task", END)
    workflow.add_edge("complex_task", END)
    
    return workflow.compile()


# Alias for backward compatibility
create_dynaroute_graph = create_simple_chat_graph
