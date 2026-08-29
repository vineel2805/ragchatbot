from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DevDocs API"
    app_version: str = "0.1.0"
    environment: str = "development"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "devdocs_chunks"

    openrouter_api_key: str | None = None
    openrouter_model: str = "google/gemma-3-27b-it:free"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()