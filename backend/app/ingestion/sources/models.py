from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class DiscoveryMode(StrEnum):
    SITEMAP = "sitemap"
    CRAWL_SAME_PREFIX = "crawl_same_prefix"
    HYBRID = "hybrid"


class ExtractionConfig(BaseModel):
    """Selector packs used later for HTML extraction. Unused until parsing exists."""

    content_selectors: list[str]
    strip_selectors: list[str]
    heading_selectors: list[str] = Field(
        default_factory=lambda: ["h1", "h2", "h3", "h4", "h5", "h6"]
    )
    code_selectors: list[str] = Field(
        default_factory=lambda: ["pre", "pre > code", "[class*='language-']"]
    )


class SourceDefinition(BaseModel):
    """One allowlisted documentation source. Add a new source by registering another instance."""

    source_id: str
    display_name: str
    origin_url: str
    allowed_hosts: list[str]
    allowed_path_prefixes: list[str]
    seed_urls: list[str]
    extraction: ExtractionConfig
    discovery: DiscoveryMode = DiscoveryMode.HYBRID
    sitemap_urls: list[str] = Field(default_factory=list)
    allowed_exact_paths: list[str] = Field(default_factory=list)
    library_allowlist: list[str] = Field(default_factory=list)
    rate_limit_rps: float = 1.0
    user_agent: str = "DevDocsRAG/0.1 (+ingestion; official-docs-only)"
    respect_robots_txt: bool = True
    keep_query_strings: bool = False
    notes: str = ""

    @field_validator("allowed_hosts")
    @classmethod
    def _hosts_lower(cls, hosts: list[str]) -> list[str]:
        if not hosts:
            raise ValueError("allowed_hosts must not be empty")
        return [host.lower() for host in hosts]

    @field_validator("allowed_path_prefixes", "allowed_exact_paths")
    @classmethod
    def _paths_start_with_slash(cls, paths: list[str]) -> list[str]:
        normalized: list[str] = []
        for path in paths:
            if not path.startswith("/"):
                raise ValueError(f"path must start with '/': {path}")
            if path != "/" and path.endswith("/"):
                path = path.rstrip("/")
            normalized.append(path)
        return normalized
