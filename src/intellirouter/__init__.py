"""DynaRoute: Intelligent AI model routing for cost optimization."""

__version__ = "0.1.0"
__author__ = "DynaRoute Team"
__description__ = "Intelligent AI model routing for cost optimization and performance"

from .core.router import ModelRouter
from .core.classifier import PromptClassifier
from .models.chat import ChatCompletionRequest, ChatCompletionResponse
from .models.providers import ModelRegistry, ProviderType
from .config import settings

__all__ = [
    "ModelRouter",
    "PromptClassifier", 
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ModelRegistry",
    "ProviderType",
    "settings",
]
