"""FastAPI application factory for IntelliRouter."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ..core.router import ModelRouter
from ..models.providers import ModelRegistry
from ..models.routing import RoutingPolicy
from ..config import settings
from .routes import create_router


# Global router instance
model_router = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global model_router
    
    # Initialize model registry with sample models
    model_registry = create_sample_model_registry()
    
    # Create routing policy
    policy = RoutingPolicy()
    
    # Initialize model router
    model_router = ModelRouter(model_registry, policy)
    
    # Store in app state
    app.state.model_router = model_router
    
    yield
    
    # Cleanup if needed
    pass


def create_sample_model_registry() -> ModelRegistry:
    """Create a sample model registry for demo purposes."""
    from ..models.providers import (
        ModelRegistry, ModelMetadata, ProviderType, 
        QualityScores, ModelCapabilities
    )
    
    registry = ModelRegistry()
    
    # Add sample OpenAI models (direct API)
    if settings.openai_api_key:
        gpt_4o_mini = ModelMetadata(
            id="gpt-4o-mini",
            name="GPT-4o Mini",
            provider=ProviderType.OPENAI,
            input_price_per_1k=0.15,
            output_price_per_1k=0.60,
            context_window=128000,
            avg_latency_ms=800,
            reliability_score=0.99,
            quality_scores=QualityScores(
                reasoning=0.85,
                coding=0.90,
                chat=0.90,
                rag=0.85,
                structured_output=0.88,
                creative_writing=0.82,
                translation=0.85,
                summarization=0.87
            ),
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_vision=False
            )
        )
        registry.models[gpt_4o_mini.id] = gpt_4o_mini
        
        gpt_4o = ModelMetadata(
            id="gpt-4o",
            name="GPT-4o",
            provider=ProviderType.OPENAI,
            input_price_per_1k=5.0,
            output_price_per_1k=15.0,
            context_window=128000,
            avg_latency_ms=1200,
            reliability_score=0.995,
            quality_scores=QualityScores(
                reasoning=0.95,
                coding=0.92,
                chat=0.94,
                rag=0.90,
                structured_output=0.93,
                creative_writing=0.90,
                translation=0.88,
                summarization=0.90
            ),
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_vision=True
            )
        )
        registry.models[gpt_4o.id] = gpt_4o
    
    # Add Azure OpenAI models (enterprise)
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        azure_gpt_4o = ModelMetadata(
            id="azure-gpt-4o",
            name="Azure GPT-4o (Enterprise)",
            provider=ProviderType.AZURE_OPENAI,
            input_price_per_1k=5.0,
            output_price_per_1k=15.0,
            context_window=128000,
            avg_latency_ms=1000,  # Slightly better latency through Azure
            reliability_score=0.999,  # Higher reliability through enterprise SLA
            quality_scores=QualityScores(
                reasoning=0.95,
                coding=0.92,
                chat=0.94,
                rag=0.90,
                structured_output=0.93,
                creative_writing=0.90,
                translation=0.88,
                summarization=0.90
            ),
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_vision=True
            ),
            regions=["East US", "West Europe", "Japan East"],
            compliance_certifications=["SOC2", "ISO27001", "HIPAA"],
            tags=["enterprise", "azure", "high-availability"]
        )
        registry.models[azure_gpt_4o.id] = azure_gpt_4o
        
        azure_gpt_4o_mini = ModelMetadata(
            id="azure-gpt-4o-mini",
            name="Azure GPT-4o Mini (Enterprise)",
            provider=ProviderType.AZURE_OPENAI,
            input_price_per_1k=0.15,
            output_price_per_1k=0.60,
            context_window=128000,
            avg_latency_ms=700,  # Better latency through Azure
            reliability_score=0.999,
            quality_scores=QualityScores(
                reasoning=0.85,
                coding=0.90,
                chat=0.90,
                rag=0.85,
                structured_output=0.88,
                creative_writing=0.82,
                translation=0.85,
                summarization=0.87
            ),
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_vision=False
            ),
            regions=["East US", "West Europe", "Japan East"],
            compliance_certifications=["SOC2", "ISO27001", "HIPAA"],
            tags=["enterprise", "azure", "cost-effective"]
        )
        registry.models[azure_gpt_4o_mini.id] = azure_gpt_4o_mini
    
    # Add sample Anthropic models (direct API)
    if settings.anthropic_api_key:
        # Claude 3.5 Sonnet (Latest and most advanced)
        claude_35_sonnet = ModelMetadata(
            id="claude-3-5-sonnet-20241022",
            name="Claude 3.5 Sonnet",
            provider=ProviderType.ANTHROPIC,
            input_price_per_1k=3.0,
            output_price_per_1k=15.0,
            context_window=200000,
            avg_latency_ms=1000,
            reliability_score=0.99,
            quality_scores=QualityScores(
                reasoning=0.95,
                coding=0.93,
                chat=0.94,
                rag=0.92,
                structured_output=0.90,
                creative_writing=0.92,
                translation=0.90,
                summarization=0.91
            ),
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                supports_json_mode=False,
                supports_vision=True
            )
        )
        registry.models[claude_35_sonnet.id] = claude_35_sonnet
        
        # Claude 3 Opus (Most capable Claude 3)
        claude_opus = ModelMetadata(
            id="claude-3-opus-20240229",
            name="Claude 3 Opus",
            provider=ProviderType.ANTHROPIC,
            input_price_per_1k=15.0,
            output_price_per_1k=75.0,
            context_window=200000,
            avg_latency_ms=1500,
            reliability_score=0.995,
            quality_scores=QualityScores(
                reasoning=0.98,
                coding=0.94,
                chat=0.95,
                rag=0.93,
                structured_output=0.92,
                creative_writing=0.95,
                translation=0.92,
                summarization=0.93
            ),
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                supports_json_mode=False,
                supports_vision=True
            )
        )
        registry.models[claude_opus.id] = claude_opus
        
        # Claude 3 Haiku (Fast and cost-effective)
        claude_haiku = ModelMetadata(
            id="claude-3-haiku-20240307",
            name="Claude 3 Haiku",
            provider=ProviderType.ANTHROPIC,
            input_price_per_1k=0.25,
            output_price_per_1k=1.25,
            context_window=200000,
            avg_latency_ms=600,
            reliability_score=0.98,
            quality_scores=QualityScores(
                reasoning=0.82,
                coding=0.85,
                chat=0.88,
                rag=0.83,
                structured_output=0.80,
                creative_writing=0.85,
                translation=0.87,
                summarization=0.85
            ),
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                supports_json_mode=False,
                supports_vision=False
            )
        )
        registry.models[claude_haiku.id] = claude_haiku
    
    # Add AWS Bedrock Anthropic models (enterprise) - Using correct Bedrock model IDs
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        # Bedrock Claude 3.5 Sonnet - Correct model ID
        bedrock_claude_35_sonnet = ModelMetadata(
            id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            name="Bedrock Claude 3.5 Sonnet (Enterprise)",
            provider=ProviderType.AWS_BEDROCK,
            input_price_per_1k=3.0,
            output_price_per_1k=15.0,
            context_window=200000,
            avg_latency_ms=900,  # Better latency through AWS
            reliability_score=0.999,  # Higher reliability through enterprise SLA
            quality_scores=QualityScores(
                reasoning=0.95,
                coding=0.93,
                chat=0.94,
                rag=0.92,
                structured_output=0.90,
                creative_writing=0.92,
                translation=0.90,
                summarization=0.91
            ),
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                supports_json_mode=False,
                supports_vision=True
            ),
            regions=["us-east-1", "us-west-2", "eu-west-1"],
            compliance_certifications=["SOC2", "ISO27001", "HIPAA", "FedRAMP"],
            tags=["enterprise", "aws", "bedrock", "default"]
        )
        registry.models[bedrock_claude_35_sonnet.id] = bedrock_claude_35_sonnet
        
        # Bedrock Claude 3 Haiku - Correct model ID
        bedrock_claude_haiku = ModelMetadata(
            id="anthropic.claude-3-haiku-20240307-v1:0",
            name="Bedrock Claude 3 Haiku (Enterprise)",
            provider=ProviderType.AWS_BEDROCK,
            input_price_per_1k=0.25,
            output_price_per_1k=1.25,
            context_window=200000,
            avg_latency_ms=500,  # Best latency through AWS
            reliability_score=0.999,
            quality_scores=QualityScores(
                reasoning=0.82,
                coding=0.85,
                chat=0.88,
                rag=0.83,
                structured_output=0.80,
                creative_writing=0.85,
                translation=0.87,
                summarization=0.85
            ),
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                supports_json_mode=False,
                supports_vision=False
            ),
            regions=["us-east-1", "us-west-2", "eu-west-1"],
            compliance_certifications=["SOC2", "ISO27001", "HIPAA", "FedRAMP"],
            tags=["enterprise", "aws", "bedrock", "cost-effective"]
        )
        registry.models[bedrock_claude_haiku.id] = bedrock_claude_haiku
    
    return registry


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="IntelliRouter",
        description="Intelligent AI model routing for cost optimization",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    api_router = create_router()
    app.include_router(api_router, prefix="/v1")
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "IntelliRouter",
            "description": "Intelligent AI model routing for cost optimization",
            "version": "0.1.0",
            "docs": "/docs"
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": "2025-01-08T12:00:00Z"}
    
    return app
