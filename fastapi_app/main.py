"""CLI entry-point for running the FastAPI application."""
from __future__ import annotations

import uvicorn

from .core.app import create_app

app = create_app()


def run() -> None:
    """Launch a development server using uvicorn."""
    uvicorn.run(
        "fastapi_app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()
