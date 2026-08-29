from fastapi import FastAPI

from app.api.routes.health import router as health_router

app = FastAPI(
    title="DevDocs API",
    description="Technical documentation RAG assistant API",
    version="0.1.0",
)

app.include_router(health_router, prefix="/api/v1")