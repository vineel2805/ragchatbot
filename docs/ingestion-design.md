# DevDocs RAG — Documentation Ingestion Design

Status: design only. Not implemented.

Approved corpus, catalog, cache, chunking, IDs, payload, and evaluation baseline are locked below. This document does not change the existing FastAPI app, create Qdrant collections, or implement ingestion.

## 1. Goals and constraints

**Goals**

- Ingest only the **locked v1 corpus** in §2 (five official sources, scoped trees — not entire sites).
- Produce retrieval-ready chunks with stable identities and the approved Qdrant payload.
- Embed locally with `BAAI/bge-small-en-v1.5` (384-dimensional vectors).
- Upsert into one collection: `devdocs_chunks` (Qdrant 1.19.0 at `http://localhost:6333`).
- Track fetch state in **SQLite**. Optionally cache raw HTML under `data/raw/` (never git, never Qdrant).
- Support adding another documentation source via the registry without rewriting the pipeline.
- Freeze a retrieval **evaluation dataset** for future Recall@k and MRR.

**Constraints**

- No paid crawl/search APIs.
- No OpenRouter usage in this pipeline.
- **No Playwright, Selenium, or other browser automation in v1.** If a page has no extractable HTML body, mark `failed_js_required` and skip it.
- **robots.txt is mandatory.** If robots disallows a path, do not fetch it; log and continue.
- Exact **host + path-prefix** allowlisting only (no registrable-domain expansion).
- Collection creation, embedding runtime, and ingestion code remain out of scope until implementation.
- The FastAPI service remains a query API; ingestion is an offline/batch process.

## 2. Locked corpus scope and allowlisting

Allowlisting is **HTTPS + exact host + path prefix**. A URL is eligible only if it matches **at least one** prefix for its `source_id` and is not denylisted.

Redirects are followed only if the **final** URL still matches the same source’s allowlist; otherwise record `rejected_redirect` and skip.

Reject examples (all sources):

- Host suffix tricks (`https://fastapi.tiangolo.com.evil.example/`)
- Wrong host or subdomain not listed
- `http://` (reject or require HTTPS equivalent still on the allowlist)
- Paths outside the locked prefixes (blogs, community, marketing, other language/version trees)

### 2.1 FastAPI (`source_id`: `fastapi`)

| | |
|--|--|
| Origin | `https://fastapi.tiangolo.com/` |
| Host | `fastapi.tiangolo.com` |
| **In scope** | Tutorial + Reference + relevant How-To / Advanced |

**Allowlisted prefixes**

- `https://fastapi.tiangolo.com/tutorial/`
- `https://fastapi.tiangolo.com/how-to/`
- `https://fastapi.tiangolo.com/advanced/`
- `https://fastapi.tiangolo.com/reference/`

**Out of scope (v1):** site home marketing, `deployment/` except where linked from the prefixes above and still matching a prefix, `about/`, `newsletter/`, `img/`, language-prefixed trees if they appear as separate path roots.

### 2.2 Python (`source_id`: `python`)

| | |
|--|--|
| Origin | `https://docs.python.org/3/` |
| Host | `docs.python.org` |
| **In scope** | Tutorial + Language Reference + **selected** Standard Library |

**Allowlisted prefixes**

- `https://docs.python.org/3/tutorial/`
- `https://docs.python.org/3/reference/`
- Selected library pages under `https://docs.python.org/3/library/` listed in the source registry (not the entire library tree).

**v1 selected Standard Library (registry list; extend later without pipeline changes)**

Index plus these modules (HTML pages and their `*.html` siblings such as `asyncio-task.html` when the prefix/module stem is listed):

`functions` (builtins), `stdtypes`, `exceptions`, `typing`, `dataclasses`, `abc`, `enum`, `contextlib`, `functools`, `itertools`, `collections`, `collections.abc`, `pathlib`, `os`, `os.path`, `sys`, `json`, `re`, `datetime`, `logging`, `argparse`, `unittest`, `asyncio` (including `asyncio-*` pages), `concurrent.futures`, `http`, `urllib`, `urllib.request`, `urllib.parse`.

**Out of scope (v1):** `docs.python.org/2/`, PEPs, HOWTOs, extending/embedding, Distutils/setuptools, `whatsnew/`, `c-api/`, `faq/`, `using/`, `distutils/`, `installing/`, and unlisted `library/` modules.

### 2.3 React (`source_id`: `react`)

| | |
|--|--|
| Origin | `https://react.dev/` |
| Host | `react.dev` |
| **In scope** | Learn + Reference |

**Allowlisted prefixes**

- `https://react.dev/learn`
- `https://react.dev/reference`

(Path matching: prefix match so `/learn/thinking-in-react` and `/reference/react/useState` are included.)

**Out of scope (v1):** `/blog`, `/community`, `/versions`, `/warnings`, conference/marketing pages.

### 2.4 Docker (`source_id`: `docker`)

| | |
|--|--|
| Origin | `https://docs.docker.com/` |
| Host | `docs.docker.com` |
| **In scope** | Get Started + Guides + Reference |

**Allowlisted prefixes**

- `https://docs.docker.com/get-started/`
- `https://docs.docker.com/guides/`
- `https://docs.docker.com/reference/`

**Out of scope (v1):** `/manuals/` unless a URL still canonicalizes into one of the prefixes above, Docker Scout/product marketing, billing, community forums, `desktop/` release notes spam if not under those prefixes.

### 2.5 Qdrant (`source_id`: `qdrant`)

| | |
|--|--|
| Origin | `https://qdrant.tech/documentation/` |
| Host | `qdrant.tech` |
| **In scope** | Documentation / User Manual + relevant Tutorials |

**Allowlisted prefixes**

- `https://qdrant.tech/documentation/overview/`
- `https://qdrant.tech/documentation/concepts/`
- `https://qdrant.tech/documentation/guides/`
- `https://qdrant.tech/documentation/tutorials/`
- `https://qdrant.tech/documentation/beginner-tutorials/` (if present)

Also allow the documentation landing page `https://qdrant.tech/documentation/` (exact path `/documentation` or `/documentation/`) as a seed only; do not treat that as a blanket crawl of every child path.

**Out of scope (v1):** `https://qdrant.tech/` outside `/documentation/`, Cloud console marketing, `/documentation/cloud/` except pages that are also under `guides/`/`concepts/` if duplicated, legal, blog, Discord.

### 2.6 Fetch eligibility checklist

1. `source_id` is in the versioned registry.
2. Scheme is `https`.
3. Host equals the registered host exactly.
4. Path starts with a **locked** prefix for that source (Python library: prefix **or** listed module stem).
5. `robots.txt` for that host allows the path for our User-Agent (or `*` if no specific rule). If disallowed → do not fetch.
6. Not denylisted (binaries, auth, search, edit, locale-spam, `utm` landing duplicates).

No open-web crawl.

## 3. Source registry and source metadata

Each source is a config record (intended to live in-repo, e.g. YAML/JSON under `backend/app/ingestion/sources/` at implementation time). Pipeline code reads the registry; it does not hard-code crawl graphs.

```text
source_id
display_name
origin_url
allowed_hosts[]              # exact hosts
allowed_path_prefixes[]      # locked v1 prefixes
library_allowlist[]          # python only: module stems
seed_urls[]
discovery                    # sitemap | crawl_same_prefix | hybrid
sitemap_urls[]               # still allowlist + robots checked
content_selectors
strip_selectors
heading_selectors
code_selectors
rate_limit_rps
user_agent                   # identifiable DevDocs bot string
respect_robots_txt           # true (required)
notes
```

**Discovery:** sitemap preferred when it exists, then filter to allowlisted prefixes. Otherwise prefix-limited crawl from seeds. Sitemap URLs outside the locked prefixes are ignored.

**Source-level metadata** on every document/chunk: `source_id`, plus origin implied by URL. Language: English only (`en`); skip locale path variants unless they match the locked prefixes (they generally will not).

## 4. URL catalog (SQLite) and optional HTML cache

### 4.1 SQLite catalog

**Store:** local SQLite database (proposed path `data/catalog/url_catalog.sqlite`). Not Qdrant. Not committed if it contains run state; schema/migrations live in git.

One row per **normalized** URL.

| Column | Purpose |
|--------|---------|
| `source_id` | Allowlist owner |
| `url` | Normalized canonical URL |
| `document_id` | See §5 |
| `discovered_from` | Parent URL or sitemap |
| `http_status` | Last fetch status |
| `etag` / `last_modified` | Conditional GET |
| `content_type` | Must be HTML (`text/html`) |
| `bytes` | Response size |
| `content_sha256` | Hash of raw response body |
| `extracted_sha256` | Hash of extracted markdown/text |
| `fetch_status` | `pending`, `fetched`, `skipped`, `failed`, `rejected`, `robots_disallowed` |
| `error` | Error class + message (no secrets) |
| `fetched_at` | UTC |
| `ingestion_run_id` | Last run that touched this URL |
| `is_in_corpus` | Still present in latest discovery |
| `duplicate_of` | Other `document_id` if exact extracted dup |

**Normalization**

- Lowercase host; strip fragment; remove default ports; collapse duplicate slashes.
- Canonical trailing slash: prefer no trailing slash except origin `/`.
- **Drop all query strings** unless a source registry flag says otherwise (v1: none need queries).

**Fetch rules**

- `GET` / `HEAD` only.
- Honor `Retry-After`, 429, `rate_limit_rps`.
- Response size cap: 5 MiB.
- Skip non-HTML.
- **Do not execute JavaScript.** Empty shell → `failed_js_required`.

### 4.2 Optional raw HTML cache (`data/raw/`)

- Layout example: `data/raw/{source_id}/{url_sha256}.html` plus a sidecar or catalog pointer from SQLite.
- **Optional:** a run may skip cache and always fetch, or write-through after a successful GET.
- **Never commit** `data/raw/` (gitignore). Treat as local replay input only.
- **Never store raw HTML in Qdrant** (or in payload `text`).
- Cache is not identity: `document_id` remains `source_id` + canonical URL.
- May retain HTML for URLs later marked inactive (disk hygiene is ops, not retrieval).

## 5. Document identity

A **document** is one HTML page after extraction.

```text
document_id = SHA256(utf8(source_id + "\n" + canonical_url))
```

Hex-encoded lowercase. Identity is not the page title and not the raw HTML hash.

**Title:** first `h1` in main content, else `<title>` with site suffix stripped.

**headings / breadcrumb:** see §9 and §13.

## 6. HTML extraction and boilerplate removal

Pipeline: `html → main DOM → markdown-like structured text`.

1. Parse HTML (HTTP / `<meta charset>`).
2. Apply `strip_selectors` (nav, header, footer, aside, cookie, search, duplicate TOC, edit widgets, language switchers).
3. Select **one** main node via `content_selectors`. If none match → `extract_failed`; **do not** index `<body>`.
4. Walk the main node in document order.

**Selector packs** (tune at implementation; smoke-test before indexing):

- FastAPI: MkDocs Material markdown container.
- Python: Sphinx `div.body` / `div.document`.
- React: docs/learn/reference content column.
- Docker: docs article; no global nav.
- Qdrant: documentation article.

Selector packs are versioned in git next to the registry.

## 7. Preservation of headings and code blocks

Target: **structured Markdown**.

**Headings:** `h1`–`h6` → ATX (`#` … `######`). Keep text verbatim (trim whitespace). Record ordered heading texts for payload `headings` and `breadcrumb`.

**Code:** `pre` / `pre > code` → fenced blocks with language from `class` when present. Preserve inner text exactly (HTML unescape only). Inline `code` → backticks. **Never split a fence across chunks.** Drop copy-button / line-number chrome if it is extra DOM text.

**Other:** lists/tables as Markdown when practical; citation URL is the page URL; images → alt text only (no binary download).

## 8. Document preprocessing

- Unicode NFC; HTML unescape.
- Collapse blank lines to max two; strip trailing line whitespace; **do not** reflow code fences.
- Drop documents under 200 characters of extracted body.
- Do not paraphrase or call an LLM.

Output: `extracted_text`, `extracted_sha256`, `title`, heading outline.

## 9. Structure-aware chunking (locked baseline)

Chunking is **heading-first, then size-capped**. Never token-blind `split()`.

**Locked parameters**

| Parameter | Value |
|-----------|--------|
| `target_tokens` | `400` |
| `overlap_tokens` | `50` |
| `chunker_version` | `"heading-v1"` |

These three are frozen together as **heading-v1**. Changing token sizes or split rules requires a **new** `chunker_version` string and a document replace (see §14).

Token counts use the **Sentence Transformers tokenizer** for `BAAI/bge-small-en-v1.5` (model max 512 tokens; 400 leaves room for breadcrumb).

**Algorithm**

1. Parse Markdown into sections by heading.
2. If a section is ≤ `target_tokens`, emit one chunk. Prepend **breadcrumb** (see payload) as a single line, then body.
3. If larger, split on paragraphs/list items **outside** fences. A fence that exceeds `target_tokens` is its own chunk; truncate only as last resort (`truncated` may be tracked in SQLite, not required in Qdrant payload).
4. Still too large → sentence split.
5. Overlap: prepend the last `overlap_tokens` of the previous chunk **in the same document**. `chunk_index` counts primary chunks only.

**Stored/embedded `text`:** breadcrumb line + overlap (if any) + primary content. Stored `text` and embedded string **must be identical**.

**Logical chunk fields (catalog / memory; Qdrant payload is §13 only)**

- `chunk_id`, `document_id`, `source_id`, `canonical_url`
- `title`, `headings`, `breadcrumb`
- `chunk_index` (0-based)
- `content_hash` (SHA-256 of primary content only: no overlap, no breadcrumb line)
- `extracted_sha256`, `chunker_version`

## 10. Duplicate detection

1. **URL alias:** same `document_id` after normalization → one document.
2. **Document exact dup:** same `source_id` + `extracted_sha256`, different URL → keep canonical/shorter; set `duplicate_of`; do not embed duplicates.
3. **Chunk exact dup:** same `content_hash` in one run → keep first.
4. **Near-dup:** out of scope for v1.

If `extracted_sha256` and `chunker_version` are unchanged vs last indexed document → **skip re-embed**.

## 11. Deterministic chunk IDs

```text
chunk_id = SHA256(utf8(source_id + canonical_url + chunker_version + chunk_index))
```

- Concatenate in that order with **no extra separators** beyond the values themselves (`chunk_index` decimal, no leading zeros, e.g. `0`, `12`).
- `chunker_version` is the literal `heading-v1` for this baseline.
- Output: lowercase hex (64 characters). This value is payload `chunk_id`.

Do not include timestamps, random salts, or embedding bytes.

**Qdrant point ID:** Qdrant 1.19 accepts UUID or unsigned integer, not a 64-char hex string. Use a **deterministic UUID** derived from the same digest: interpret the **first 16 bytes** of the SHA-256 as a UUID (set RFC 4122 version/variant bits in a documented way so the mapping is stable). Payload still stores the full hex `chunk_id`. Upserts key off this UUID.

If a heading insert shifts `chunk_index`, later IDs change. Replace the whole `document_id` (deactivate old points, upsert new) when extraction or `chunker_version` changes.

## 12. Embedding generation

- Model: `BAAI/bge-small-en-v1.5` (Sentence Transformers 6.x).
- Dimension: **384**; reject other lengths.
- Local only; L2-normalize; Qdrant **Cosine**.
- Embed the same string as payload `text`.
- Query instruction (retrieval later): BGE query prefix on **queries only**, not on indexed passages. Freeze in config at implementation.
- No fine-tuning. `embedding_model` payload value: `BAAI/bge-small-en-v1.5`.

## 13. Qdrant payload design

**One collection:** `devdocs_chunks` (Settings `qdrant_collection`).

| Setting | Value |
|---------|--------|
| Vector size | 384 |
| Distance | Cosine |
| Point ID | UUID derived from `chunk_id` SHA-256 (§11) |

**Payload fields (locked)**

| Field | Type | Indexed | Description |
|-------|------|---------|-------------|
| `source_id` | keyword | yes | `fastapi` \| `python` \| `react` \| `docker` \| `qdrant` |
| `document_id` | keyword | yes | SHA-256 hex of source + URL |
| `chunk_id` | keyword | yes | SHA-256 hex as specified in §11 |
| `url` | keyword | yes | Canonical citation URL |
| `title` | keyword or text | optional | Page title |
| `headings` | keyword[] | optional | Heading texts from page root to this section (ordered) |
| `breadcrumb` | keyword or text | optional | Display trail, e.g. `FastAPI > Tutorial > Path Parameters` |
| `text` | text | no | Embedded string (breadcrumb + overlap + body) |
| `content_hash` | keyword | no | SHA-256 of primary chunk body |
| `chunker_version` | keyword | yes | `heading-v1` |
| `embedding_model` | keyword | no | `BAAI/bge-small-en-v1.5` |
| `ingestion_run_id` | keyword | yes | UUID of the ingest run |
| `created_at` | keyword or datetime | no | UTC ISO-8601; preserve on overwrite of the same point ID when payload is first written, else set on insert |
| `is_active` | bool | yes | Retrieval must filter `is_active = true` |

Do not store API keys, cookies, raw HTML, HTTP headers, or `data/raw/` paths in Qdrant.

No additional payload keys in v1.

## 14. Re-index and update strategy

**Incremental**

1. Discover URLs (allowlist + robots) → upsert SQLite catalog.
2. Conditional GET when validators exist; optional read from `data/raw/` if hash still valid.
3. Raw `content_sha256` unchanged → skip.
4. Extract; if `extracted_sha256` + `chunker_version` unchanged → skip.
5. Else **replace document**:
   - Set `is_active = false` on existing points with that `document_id` (filter update), or delete them.
   - Upsert new points with `is_active = true`.
6. URLs missing from discovery for **2 consecutive successful source crawls**: set catalog `is_in_corpus = false` and Qdrant `is_active = false`. Do not deactivate on a single failed crawl.
7. 404/410: mark gone; set `is_active = false` for that `document_id`.

**Full rebuild:** new `ingestion_run_id`; deactivate or drop points with old `chunker_version` / `embedding_model`; required if model, vector size, distance, or chunker changes.

**Concurrency:** one writer on `devdocs_chunks` in v1. FastAPI does not mutate Qdrant during ingest.

## 15. Failure handling

| Failure | Behavior |
|---------|----------|
| DNS / 5xx / timeout | Retry 3× with backoff, then `failed`; continue |
| 404 / 410 | `gone`; deactivate indexed points |
| 429 | Back off |
| Allowlist miss / bad redirect | `rejected` |
| robots.txt disallow | `robots_disallowed`; never fetch |
| Non-HTML / oversize | `skipped` |
| Empty JS shell | `failed_js_required`; no browser fallback in v1 |
| Extract miss | `extract_failed` |
| Embed/Qdrant error | Fail batch with SQLite checkpoint; no half-active documents |
| Interrupt | Persist SQLite; Qdrant only completed documents |

Exit non-zero if any source fails on **>10%** of discovered in-scope HTML URLs.

No unofficial mirrors.

## 16. Logging

JSON lines to stdout. No secrets, no `.env`, no raw HTML at info.

Fields: `ts`, `level`, `ingestion_run_id`, `stage`, `source_id`, `url`, `status`, `duration_ms`, `error_type`.

Stages: `discover`, `robots`, `fetch`, `cache`, `extract`, `preprocess`, `chunk`, `embed`, `upsert`, `deactivate`.

End counters: discovered, fetched, robots_disallowed, skipped, failed, js_required, documents_embedded, chunks_upserted, deactivated, dups.

## 17. Reproducibility

Run manifest (file, not only Qdrant):

- Git commit, registry hash, `ingestion_run_id`
- `chunker_version` = `heading-v1`, `target_tokens` = 400, `overlap_tokens` = 50
- `embedding_model`, ST version, 384 dims, Cosine, normalize=true
- Collection name `devdocs_chunks`
- Seed/sitemap URLs actually used
- UTC timestamp

Same HTML bytes + same registry + same chunker/embedder → same `chunk_id`s. Prefer CPU for golden embedding tests.

Golden tests (later): fixture HTML → Markdown + chunk boundaries. No live scrape in CI by default.

## 18. Adding another documentation source

1. New `source_id` + registry row (hosts, **locked prefixes**, seeds, selectors, rate limit).
2. Allowlist review (HTTPS, official docs, robots).
3. Smoke extract 5–10 pages.
4. Ingest filtered by `source_id`.
5. Add eval questions for that source (§19). Payload schema unchanged.

Custom extractor / query-string rules only if the site is unusual. **Still no Playwright/Selenium in v1.**

## 19. Evaluation dataset design

Purpose: measure **retrieval** (not generation) against the locked corpus. Used later for **Recall@k** and **MRR**. Not implemented in this change; design only.

### 19.1 Artifact

Proposed path: `eval/retrieval_v1.jsonl` (committed). One JSON object per line. Frozen independently of Qdrant (gold is questions + expected pages/concepts, not point IDs).

Do not put secrets or raw HTML in the eval file.

### 19.2 Record schema

| Field | Required | Description |
|-------|----------|-------------|
| `id` | yes | Stable string, e.g. `eval-fastapi-001` |
| `question` | yes | Natural-language user query |
| `expected_source_id` | yes | One of the five slugs (primary source) |
| `expected_urls` | yes | 1–3 canonical URL prefixes or exact URLs that should rank |
| `expected_concepts` | yes | Short strings the top chunks should mention (APIs, headings) |
| `acceptable_source_ids` | no | If a second official source could fairly answer |
| `question_type` | yes | `factual` \| `how_to` \| `api_lookup` \| `compare` \| `debug` |
| `difficulty` | yes | `easy` \| `medium` \| `hard` |
| `notes` | no | Why this item exists; ambiguous cases |

**Matching rules (for future scoring code)**

- **URL hit:** retrieved chunk `url` equals an expected URL after normalization, **or** expected URL is a prefix of chunk `url` (so a gold `.../learn/thinking-in-react` matches that page’s chunks). Fragments ignored.
- **Source hit:** `source_id` equals `expected_source_id` (report separately from URL hit).
- **Concept hit (diagnostic only):** `text` or `headings` contains at least one `expected_concepts` token (case-insensitive). Does not replace URL gold for Recall@k.

v1 gold is **page-level** (any chunk from a gold URL counts). Chunk-level gold can be added later via optional `expected_heading` without breaking `id`s.

### 19.3 Metrics (future)

For each question, rank active chunks (`is_active = true`) with the same embedder and query prefix policy.

- **Recall@k** for `k ∈ {1, 5, 10}`: 1 if any of the top-k chunks is a URL hit, else 0. Macro-average over questions; also report **per `source_id`**.
- **MRR:** `1/rank` of the first URL hit, else 0. Macro-average.
- Optional later: nDCG@k if graded labels are added (`relevant` / `partial`).

Retrieval filter for eval: `is_active = true`. Do not require `source_id` filter unless the question set is source-sliced.

### 19.4 Coverage and size

Target **≥ 50** questions in v1, **≥ 8 per source**, mix of `question_type` and `difficulty`. Include at least:

- One **negative-control** style item per source that is still in-corpus but easy to confuse (e.g. React `useEffect` vs `useLayoutEffect`) so MRR is meaningful.
- No questions whose only answer is **out of corpus** (Python HOWTO, React blog, Docker manuals-only pages). If asked, expected result is “not in corpus” and those items live in a separate `eval/out_of_scope_v1.jsonl` (not scored in Recall@k).

### 19.5 Representative questions (illustrative gold — refine URLs at implementation against live canonical paths)

These lock **intent and source**; exact paths must be verified once against the allowlist before scoring.

**FastAPI**

| id | question | expected_source_id | expected_urls / concepts |
|----|----------|--------------------|---------------------------|
| `eval-fastapi-001` | How do I declare a path parameter in FastAPI? | `fastapi` | Tutorial path-params page; `Path`, `{item_id}` |
| `eval-fastapi-002` | How does FastAPI validate a Pydantic request body? | `fastapi` | Tutorial body / Pydantic; `BaseModel` |
| `eval-fastapi-003` | How do I add a dependency with `Depends`? | `fastapi` | Tutorial dependencies; `Depends` |
| `eval-fastapi-004` | What is the FastAPI `APIRouter` used for? | `fastapi` | Bigger apps / `APIRouter` reference |
| `eval-fastapi-005` | How do I return an HTTP 404 from a FastAPI path operation? | `fastapi` | Handling errors / `HTTPException` |
| `eval-fastapi-006` | How do I define a response model? | `fastapi` | Response model tutorial/reference; `response_model` |
| `eval-fastapi-007` | How do I use FastAPI security with OAuth2 password flow? | `fastapi` | Security tutorial; `OAuth2PasswordBearer` |
| `eval-fastapi-008` | How do I upload a file in FastAPI? | `fastapi` | Request files; `UploadFile` |
| `eval-fastapi-009` | What does FastAPI `Query` do for validation? | `fastapi` | Query parameters / `Query` |
| `eval-fastapi-010` | How do I run background tasks in FastAPI? | `fastapi` | Background tasks / how-to or tutorial; `BackgroundTasks` |

**Python**

| id | question | expected_source_id | expected_urls / concepts |
|----|----------|--------------------|---------------------------|
| `eval-python-001` | How do `for` loops and the iterator protocol work in Python? | `python` | Tutorial iteration / reference datamodel; `iter`, `__next__` |
| `eval-python-002` | What is the difference between `*args` and `**kwargs`? | `python` | Tutorial defining functions; unpacking |
| `eval-python-003` | How do I use `pathlib.Path` to join paths? | `python` | `library/pathlib.html`; `Path`, `/` operator |
| `eval-python-004` | How does `asyncio.gather` run coroutines concurrently? | `python` | asyncio docs; `gather` |
| `eval-python-005` | How do I parse JSON in the standard library? | `python` | `library/json.html`; `json.loads` |
| `eval-python-006` | What are Python type hints for optional values? | `python` | `library/typing.html`; `Optional`, `\| None` |
| `eval-python-007` | How do I match a regex group in `re`? | `python` | `library/re.html`; `Match`, groups |
| `eval-python-008` | What does the language reference say about LEGB / name binding? | `python` | Language reference execution model / naming |
| `eval-python-009` | How do I create a `dataclass` with a default factory? | `python` | `library/dataclasses.html`; `field(default_factory=...)` |
| `eval-python-010` | How do I log an exception with the `logging` module? | `python` | `library/logging.html`; `exception` / `exc_info` |

**React**

| id | question | expected_source_id | expected_urls / concepts |
|----|----------|--------------------|---------------------------|
| `eval-react-001` | How do I update state with `useState`? | `react` | Learn state / reference `useState` |
| `eval-react-002` | When should I use `useEffect`? | `react` | Learn synchronizing / reference `useEffect` |
| `eval-react-003` | How does React render lists and `key`s? | `react` | Learn rendering lists; `key` |
| `eval-react-004` | How do I pass data with context? | `react` | Learn passing data / `useContext` |
| `eval-react-005` | What is the difference between `useRef` and state? | `react` | Learn refs / reference `useRef` |
| `eval-react-006` | How do I write a custom Hook? | `react` | Learn reusing logic; `use` prefix |
| `eval-react-007` | How does `useMemo` work? | `react` | Reference `useMemo` |
| `eval-react-008` | How do forms work in React? | `react` | Learn forms / `<form>` |
| `eval-react-009` | What is React Server Components at a high level? | `react` | Learn server components (if under `/learn` or `/reference`) |
| `eval-react-010` | How do I lift state up? | `react` | Learn sharing state |

**Docker**

| id | question | expected_source_id | expected_urls / concepts |
|----|----------|--------------------|---------------------------|
| `eval-docker-001` | How do I write a simple Dockerfile? | `docker` | Get Started / guides; `FROM`, `COPY`, `CMD` |
| `eval-docker-002` | What does `docker build` do? | `docker` | Get Started or reference `docker build` |
| `eval-docker-003` | How do I map a container port to the host? | `docker` | Guides or reference `-p` / `ports` |
| `eval-docker-004` | How do I use a Docker volume to persist data? | `docker` | Guides volumes; `docker volume` |
| `eval-docker-005` | What is the difference between `CMD` and `ENTRYPOINT`? | `docker` | Dockerfile reference |
| `eval-docker-006` | How do I pass an environment variable into a container? | `docker` | Guides or `docker run -e` reference |
| `eval-docker-007` | How does a multi-stage Docker build work? | `docker` | Guides multi-stage; `AS` |
| `eval-docker-008` | What is `docker compose` used for? | `docker` | Guides Compose; `compose.yaml` |
| `eval-docker-009` | How do I inspect running containers? | `docker` | Reference `docker ps` / `docker inspect` |
| `eval-docker-010` | How do I tag and push an image? | `docker` | Get Started publish / `docker tag` `docker push` |

**Qdrant**

| id | question | expected_source_id | expected_urls / concepts |
|----|----------|--------------------|---------------------------|
| `eval-qdrant-001` | How do I create a collection in Qdrant? | `qdrant` | Concepts collections / guides; vectors, distance |
| `eval-qdrant-002` | What payload indexes exist in Qdrant? | `qdrant` | Concepts payload / indexing |
| `eval-qdrant-003` | How do I run a similarity search with a filter? | `qdrant` | Guides filtering / search |
| `eval-qdrant-004` | What is a Qdrant point ID? | `qdrant` | Concepts points; UUID vs integer |
| `eval-qdrant-005` | How do I upsert points? | `qdrant` | Guides upsert / points |
| `eval-qdrant-006` | How does Cosine distance work in Qdrant? | `qdrant` | Concepts distance / vectors |
| `eval-qdrant-007` | How do I delete points by filter? | `qdrant` | Guides delete; filter |
| `eval-qdrant-008` | What is a tutorial for getting started with Qdrant locally? | `qdrant` | Tutorials / beginner; Docker or local |
| `eval-qdrant-009` | How do I store extra JSON on a point? | `qdrant` | Concepts payload |
| `eval-qdrant-010` | How do I choose vector size for a collection? | `qdrant` | Concepts collections; vector size |

Fill remaining slots to reach ≥50 with more `how_to` / `debug` items (e.g. FastAPI CORS, Python `asyncio.Lock`, React key warnings, Docker `.dockerignore`, Qdrant quantization) **only if** those pages stay inside locked prefixes.

### 19.6 Process

1. After first successful ingest, resolve each `expected_urls` entry to a live canonical URL still in SQLite `is_in_corpus`.
2. Drop or rewrite items whose gold pages were never fetched (`robots_disallowed` / `failed_js_required`).
3. Freeze `eval/retrieval_v1.jsonl`; bump to `retrieval_v2` if gold URLs change with docs sites.
4. Eval harness (later) reads JSONL, embeds questions, queries Qdrant, writes a score report. No OpenRouter required for Recall@k/MRR.

## 20. Proposed pipeline stages (implementation order, later)

```text
registry → robots + discover (allowlist) → fetch → optional data/raw/ cache
  → SQLite catalog → extract → preprocess → chunk (heading-v1)
  → embed (bge-small 384) → upsert devdocs_chunks
```

Batch CLI under `backend/` is the intended entrypoint. Not an HTTP ingest API in v1.

## 21. Assumptions

1. Official **HTML** documentation only; not GitHub trees, PDFs, or community Q&A.
2. English only; locked prefixes in §2 define v1 completeness.
3. robots.txt is never overridden.
4. One collection `devdocs_chunks`; sources distinguished by `source_id`.
5. No browser automation in v1; JS-only pages are skipped.
6. SQLite is the URL catalog; `data/raw/` is optional and uncommitted.
7. `chunk_id` formula uses **direct concatenation** (no delimiters).
8. Retrieval always filters `is_active = true`.
9. OpenRouter is unrelated to ingestion and to Recall@k/MRR.
10. Docker `/manuals/` and Python unlisted library modules are out of corpus even if useful.

## 22. Remaining items before implementation

Decided in this revision: corpus prefixes, SQLite, `data/raw/`, no Playwright/Selenium, robots + exact allowlist, chunking baseline, SHA-256 `chunk_id`, payload fields, single collection, eval dataset design.

Still to approve at implementation kickoff:

1. HTTP/HTML libraries (`httpx`, `beautifulsoup4`/`lxml`, optional `markdownify`) — new dependencies.
2. Create-if-missing vs explicit ops step for `devdocs_chunks` and payload indexes.
3. CLI module path vs separate worker (must stay off the live request path).
4. What to do if robots.txt blocks a locked prefix (skip source vs shrink corpus vs vendor dump) — skip remains the default.
5. CPU vs GPU for embeddings (CPU preferred for reproducible eval).
6. Physical delete of `is_active = false` points vs retain for audit.
7. Exact Python `asyncio-*` page stems and Docker/React canonical paths when sitemaps are first fetched (registry edit, not architecture change).
8. Qdrant UUID bit-setting details for point IDs derived from SHA-256.
)
