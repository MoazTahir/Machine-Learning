"""Application routing configuration."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..models.health import HealthResponse
from ..services.registry import RegistryEntry, get_registry

api_router = APIRouter()


@api_router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    registry = get_registry()
    return HealthResponse(status="ok", registered_models=list(registry.keys()))


def _register_model_routes(router: APIRouter) -> None:
    registry = get_registry()
    for entry in registry.values():
        entry.register_route(router)


@api_router.post("/models/{slug}/invoke", tags=["runtime"])
async def invoke_model(slug: str, payload: dict[str, Any]) -> Any:
    registry = get_registry()
    entry: RegistryEntry | None = registry.get(slug)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown model slug: {slug}")

    request_model = entry.request_model()
    response_model = entry.response_model()
    service = entry.build_service()

    validated_payload = request_model(**payload)
    result = service.predict(validated_payload)
    if isinstance(result, response_model):
        return result
    return response_model(**result)


_register_model_routes(api_router)
