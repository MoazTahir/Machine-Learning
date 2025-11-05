"""Pydantic models for health check responses."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    registered_models: List[str]
