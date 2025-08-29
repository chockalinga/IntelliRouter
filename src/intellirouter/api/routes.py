"""API routes for IntelliRouter."""

from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
from typing import Optional
import json

from ..models.chat import ChatCompletionRequest, ChatCompletionResponse
from ..models.routing import RoutingConstraints


def create_router() -> APIRouter:
    """Create API router."""
    
    router = APIRouter()
    
    @router.post("/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        request_data: ChatCompletionRequest,
        http_request: Request,
        authorization: Optional[str] = Header(None)
    ):
        """OpenAI-compatible chat completions endpoint."""
        
        # Get model router from app state
        model_router = http_request.app.state.model_router
        
        try:
            if request_data.stream:
                # Streaming response
                async def stream_generator():
                    async for chunk in model_router.route_streaming_request(request_data):
                        yield chunk
                
                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming response
                response = await model_router.route_request(request_data)
                return response
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/models")
    async def list_models(http_request: Request):
        """List available models."""
        model_router = http_request.app.state.model_router
        
        models = []
        for model_id, model in model_router.model_registry.models.items():
            models.append({
                "id": model_id,
                "object": "model",
                "created": 1677610602,
                "owned_by": model.provider
            })
        
        return {
            "object": "list",
            "data": models
        }
    
    @router.get("/stats")
    async def get_stats(http_request: Request):
        """Get router statistics."""
        model_router = http_request.app.state.model_router
        return model_router.get_stats()
    
    return router

# Export router for backward compatibility
router = create_router()
