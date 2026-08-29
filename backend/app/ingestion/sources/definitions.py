from app.ingestion.sources.models import DiscoveryMode, ExtractionConfig, SourceDefinition

_DEFAULT_STRIP = [
    "nav",
    "header",
    "footer",
    "aside",
    "[role='navigation']",
    "[role='banner']",
    "[role='contentinfo']",
    ".cookie",
    "#on-this-page",
]

FASTAPI = SourceDefinition(
    source_id="fastapi",
    display_name="FastAPI",
    origin_url="https://fastapi.tiangolo.com/",
    allowed_hosts=["fastapi.tiangolo.com"],
    allowed_path_prefixes=["/tutorial", "/how-to", "/advanced", "/reference"],
    seed_urls=[
        "https://fastapi.tiangolo.com/tutorial/",
        "https://fastapi.tiangolo.com/how-to/",
        "https://fastapi.tiangolo.com/advanced/",
        "https://fastapi.tiangolo.com/reference/",
    ],
    discovery=DiscoveryMode.HYBRID,
    sitemap_urls=["https://fastapi.tiangolo.com/sitemap.xml"],
    extraction=ExtractionConfig(
        content_selectors=["article.md-content", ".md-content__inner", "article", "main"],
        strip_selectors=[*_DEFAULT_STRIP, ".md-header", ".md-sidebar", ".md-footer", ".headerlink"],
    ),
    notes="Tutorial + How-To + Advanced + Reference only.",
)

PYTHON = SourceDefinition(
    source_id="python",
    display_name="Python",
    origin_url="https://docs.python.org/3/",
    allowed_hosts=["docs.python.org"],
    allowed_path_prefixes=["/3/tutorial", "/3/reference"],
    library_allowlist=[
        "functions",
        "stdtypes",
        "exceptions",
        "typing",
        "dataclasses",
        "abc",
        "enum",
        "contextlib",
        "functools",
        "itertools",
        "collections",
        "collections.abc",
        "pathlib",
        "os",
        "os.path",
        "sys",
        "json",
        "re",
        "datetime",
        "logging",
        "argparse",
        "unittest",
        "asyncio",
        "concurrent.futures",
        "http",
        "urllib",
        "urllib.request",
        "urllib.parse",
    ],
    seed_urls=[
        "https://docs.python.org/3/tutorial/",
        "https://docs.python.org/3/reference/",
        "https://docs.python.org/3/library/",
        "https://docs.python.org/3/library/index.html",
    ],
    discovery=DiscoveryMode.HYBRID,
    sitemap_urls=["https://docs.python.org/3/sitemap.xml"],
    extraction=ExtractionConfig(
        content_selectors=["div.body", "div.document", "main"],
        strip_selectors=[*_DEFAULT_STRIP, ".sphinxsidebar", ".related", ".headerlink"],
    ),
    notes="Tutorial + language reference + selected stdlib modules (not the full library tree).",
)

REACT = SourceDefinition(
    source_id="react",
    display_name="React",
    origin_url="https://react.dev/",
    allowed_hosts=["react.dev"],
    allowed_path_prefixes=["/learn", "/reference"],
    seed_urls=[
        "https://react.dev/learn",
        "https://react.dev/reference",
    ],
    discovery=DiscoveryMode.HYBRID,
    sitemap_urls=["https://react.dev/sitemap.xml"],
    extraction=ExtractionConfig(
        content_selectors=["article", "main .mdx-content", "main"],
        strip_selectors=[*_DEFAULT_STRIP, "[data-site-nav]", ".toc"],
    ),
    notes="Learn + Reference only.",
)

DOCKER = SourceDefinition(
    source_id="docker",
    display_name="Docker",
    origin_url="https://docs.docker.com/",
    allowed_hosts=["docs.docker.com"],
    allowed_path_prefixes=["/get-started", "/guides", "/reference"],
    seed_urls=[
        "https://docs.docker.com/get-started/",
        "https://docs.docker.com/guides/",
        "https://docs.docker.com/reference/",
    ],
    discovery=DiscoveryMode.SITEMAP,
    sitemap_urls=["https://docs.docker.com/sitemap.xml"],
    extraction=ExtractionConfig(
        content_selectors=["article", "main article", "main"],
        strip_selectors=[*_DEFAULT_STRIP, ".sidebar", ".navbar"],
    ),
    notes="Get Started + Guides + Reference only.",
)

QDRANT = SourceDefinition(
    source_id="qdrant",
    display_name="Qdrant",
    origin_url="https://qdrant.tech/documentation/",
    allowed_hosts=["qdrant.tech"],
    allowed_path_prefixes=[
        "/documentation/overview",
        "/documentation/concepts",
        "/documentation/guides",
        "/documentation/tutorials",
        "/documentation/beginner-tutorials",
    ],
    allowed_exact_paths=["/documentation"],
    seed_urls=[
        "https://qdrant.tech/documentation/",
        "https://qdrant.tech/documentation/overview/",
        "https://qdrant.tech/documentation/concepts/",
        "https://qdrant.tech/documentation/guides/",
        "https://qdrant.tech/documentation/tutorials/",
        "https://qdrant.tech/documentation/beginner-tutorials/",
    ],
    discovery=DiscoveryMode.CRAWL_SAME_PREFIX,
    sitemap_urls=["https://qdrant.tech/sitemap.xml"],
    extraction=ExtractionConfig(
        content_selectors=["article", "main .theme-doc-markdown", "main"],
        strip_selectors=[*_DEFAULT_STRIP, ".theme-doc-sidebar-container", ".pagination-nav"],
    ),
    notes="User manual trees + tutorials. Landing /documentation is exact-path only, not a prefix.",
)

ALL_SOURCES: tuple[SourceDefinition, ...] = (FASTAPI, PYTHON, REACT, DOCKER, QDRANT)
