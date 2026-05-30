"""
platform_api/api/app.py — FastAPI application factory.

This is the main entry point for the Supreme ModelTX business API.
It mounts all sub-routers and provides OpenAPI documentation.

Usage:
    uvicorn supreme_modeltx.platform_api.api.app:create_app --factory --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from supreme_modeltx.platform_api.api import engine as engine_module
from supreme_modeltx.platform_api.api.routers import (
    audit,
    auth,
    chat,
    embeddings,
    health,
    keys,
    models,
    projects,
    responses,
    usage,
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Supreme ModelTX Platform API",
        description=(
            "API-first sovereign LLM platform. "
            "Provides model access, project management, auth, and usage metering."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(health.router, prefix="/health", tags=["Health"])
    application.include_router(auth.router, prefix="/v1/auth", tags=["Auth"])
    application.include_router(models.router, prefix="/v1/models", tags=["Models"])
    application.include_router(chat.router, prefix="/v1/chat", tags=["Chat"])
    application.include_router(responses.router, prefix="/v1/responses", tags=["Responses"])
    application.include_router(embeddings.router, prefix="/v1/embeddings", tags=["Embeddings"])
    application.include_router(projects.router, prefix="/v1/projects", tags=["Projects"])
    application.include_router(usage.router, prefix="/v1/usage", tags=["Usage"])
    application.include_router(keys.router, prefix="/v1/keys", tags=["Keys"])
    application.include_router(audit.router, prefix="/v1/audit", tags=["Audit"])

    @application.on_event("startup")
    async def _startup() -> None:
        logger.info("Supreme ModelTX Platform API starting up.")
        engine_module.initialize_engine()

    return application


app = create_app()


def main() -> None:
    """Run the API server (used as console_scripts entrypoint)."""
    import uvicorn
    uvicorn.run(
        "supreme_modeltx.platform_api.api.app:create_app",
        host="0.0.0.0",
        port=9000,
        reload=False,
        factory=True,
    )


if __name__ == "__main__":
    main()
