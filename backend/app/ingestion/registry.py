from collections.abc import Iterator

from app.ingestion.sources.definitions import ALL_SOURCES
from app.ingestion.sources.models import SourceDefinition

_REGISTRY: dict[str, SourceDefinition] = {source.source_id: source for source in ALL_SOURCES}


def get_source(source_id: str) -> SourceDefinition:
    try:
        return _REGISTRY[source_id]
    except KeyError as exc:
        raise KeyError(f"unknown source_id: {source_id}") from exc


def list_source_ids() -> list[str]:
    return list(_REGISTRY.keys())


def iter_sources() -> Iterator[SourceDefinition]:
    return iter(_REGISTRY.values())
