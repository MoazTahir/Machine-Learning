"""Application configuration utilities."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import BaseModel


class Settings(BaseModel):
    """Global application settings."""

    app_name: str = "Machine Learning Playbook API"
    api_version: str = "1.0.0"
    description: str = (
        "Unified inference layer for the educational machine learning repository."
    )
    debug: bool = True
    docs_url: str | None = "/docs"
    redoc_url: str | None = "/redoc"

    class Config:
        arbitrary_types_allowed = True


@lru_cache(maxsize=1)
def get_settings(**overrides: Any) -> Settings:
    """Fetch cached settings instance with optional overrides for testing."""
    if overrides:
        return Settings(**overrides)
    return Settings()
