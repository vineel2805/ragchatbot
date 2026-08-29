from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.rag import router as rag_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Technical documentation RAG assistant API",
    version=settings.app_version,
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")