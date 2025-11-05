"""Factory that constructs the FastAPI application instance."""
from __future__ import annotations

from fastapi import FastAPI

from ..api.routes import api_router
from .config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.api_version,
        description=settings.description,
        debug=settings.debug,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
    )
    app.include_router(api_router)
    return app
