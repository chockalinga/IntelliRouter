"""FastAPI application for DynaRoute."""

from .app import create_app
from .routes import router

__all__ = ["create_app", "router"]
