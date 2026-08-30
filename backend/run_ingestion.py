#!/usr/bin/env python
"""Run documentation ingestion for a single source.

Usage:
    python run_ingestion.py <source_id>

Available sources:
    fastapi, python, react, docker, qdrant

Example:
    python run_ingestion.py fastapi

Requirements:
    - Qdrant running at QDRANT_URL (default: http://localhost:6333)
    - .env file configured (QDRANT_URL, QDRANT_COLLECTION)
    - First run downloads BGE model (~133MB)

Output:
    - SQLite catalog: data/catalog/url_catalog.sqlite
    - Qdrant collection: devdocs_chunks (or QDRANT_COLLECTION)
"""
from __future__ import annotations

import sys
from pathlib import Path

from app.ingestion import (
    IngestionCatalog,
    get_source,
    ingest_source,
    list_source_ids,
    make_qdrant_indexer,
)


def main(source_id: str) -> int:
    """Run ingestion for the specified source_id.

    Returns:
        0 on success, 1 on error.
    """
    # Validate source_id
    available = list_source_ids()
    if source_id not in available:
        print(f"Error: Unknown source_id '{source_id}'", file=sys.stderr)
        print(f"Available sources: {', '.join(available)}", file=sys.stderr)
        return 1

    # Initialize catalog (creates SQLite DB and schema if not exists)
    catalog_path = Path("data/catalog/url_catalog.sqlite")
    print(f"Initializing catalog at {catalog_path}")
    catalog = IngestionCatalog(catalog_path)

    try:
        # Get source definition from registry
        source = get_source(source_id)
        print(f"Loaded source: {source.display_name} ({source.source_id})")
        print(f"Origin: {source.origin_url}")

        # Create Qdrant indexer (reads QDRANT_URL from settings/env)
        print("Connecting to Qdrant and initializing indexer...")
        indexer = make_qdrant_indexer(catalog)

        # Ensure collection exists (creates if needed)
        indexer.ensure_collection()
        print(f"Qdrant collection ready: {indexer._collection}")

        # Run ingestion
        print(f"\nStarting ingestion for {source_id}...")
        print("This may take several minutes depending on the source size.")
        print("Progress: discovering URLs, fetching, extracting, chunking, embedding, indexing...")
        print()

        result = ingest_source(source, catalog, indexer=indexer)

        # Print results
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"Source:              {result.source_id}")
        print(f"Run ID:              {result.run.id}")
        print(f"Status:              {result.run.status.value}")
        print(f"Started:             {result.run.started_at}")
        print(f"Finished:            {result.run.finished_at}")
        print()
        print(f"URLs registered:     {result.urls_registered}")
        print(f"URLs fetched:        {result.urls_fetched}")
        print(f"URL failures:        {result.url_failures}")
        print(f"Sitemap failures:    {result.sitemaps_failed}")
        print()
        print(f"Chunks indexed:      {result.chunks_indexed}")
        print(f"Points deactivated:  {result.points_deactivated}")

        if result.errors:
            print()
            print(f"Errors encountered:  {len(result.errors)}")
            for i, error in enumerate(result.errors, 1):
                print(f"  {i}. {error}")

        print("=" * 60)
        print()
        print(f"Catalog: {catalog_path}")
        print(f"Collection: {indexer._collection}")
        print()

        return 0 if result.run.status.value == "succeeded" else 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Catalog state preserved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nError during ingestion: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        catalog.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        print("\nUsage: python run_ingestion.py <source_id>")
        print(f"Available sources: {', '.join(list_source_ids())}")
        sys.exit(1)

    sys.exit(main(sys.argv[1]))
