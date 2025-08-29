"""Main entry point for IntelliRouter server."""

import uvicorn
from src.intellirouter.api.app import create_app
from src.intellirouter.config import settings

if __name__ == "__main__":
    app = create_app()
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
