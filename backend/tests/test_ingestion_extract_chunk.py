from __future__ import annotations

import hashlib
import unittest

from app.ingestion.chunker import OVERLAP_TOKENS, TARGET_TOKENS, chunk_document
from app.ingestion.extract import extract_document
from app.ingestion.ids import CHUNKER_VERSION, make_chunk_id, make_document_id
from app.ingestion.normalize import normalize_extracted_text
from app.ingestion.registry import get_source, iter_sources, list_source_ids
from app.ingestion.tokenize import count_tokens, overlap_text

FILL = (
    "This documentation paragraph is padding so the extracted page exceeds the "
    "minimum character threshold before chunking. It must remain in the main "
    "content node and must not be confused with navigation chrome. "
)

URLS = {
    "fastapi": "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "python": "https://docs.python.org/3/tutorial/index.html",
    "react": "https://react.dev/learn/thinking-in-react",
    "docker": "https://docs.docker.com/get-started/introduction/",
    "qdrant": "https://qdrant.tech/documentation/concepts/points/",
}

WRAPPERS = {
    "fastapi": ("article", {"class": "md-content"}),
    "python": ("div", {"class": "body"}),
    "react": ("article", {}),
    "docker": ("article", {}),
    "qdrant": ("article", {}),
}


def _wrap(source_id: str, inner: str) -> str:
    tag, attrs = WRAPPERS[source_id]
    attr = "".join(f' {key}="{value}"' for key, value in attrs.items())
    return f"""<!doctype html>
<html>
<head><title>Demo Page - {source_id}</title></head>
<body>
<nav id="site-nav">NAVIGATION_CHROME should never be indexed</nav>
<header>HEADER_CHROME</header>
<aside>SIDEBAR_CHROME</aside>
<footer>FOOTER_CHROME</footer>
<script>alert("xss")</script>
<style>body {{ display:none }}</style>
<form action="/search"><input name="q" /></form>
<{tag}{attr}>
{inner}
</{tag}>
</body>
</html>
"""


def _extract(source_id: str, inner: str, url: str | None = None):
    source = get_source(source_id)
    return extract_document(source, _wrap(source_id, inner), url or URLS[source_id])


class SourceRegistryExtractTests(unittest.TestCase):
    def test_all_five_source_definitions(self) -> None:
        self.assertEqual(list_source_ids(), ["fastapi", "python", "react", "docker", "qdrant"])
        inner = f"<h1>Title</h1><p>{FILL}</p><pre><code class='language-python'>print(1)</code></pre>"
        for source in iter_sources():
            self.assertTrue(source.extraction.content_selectors)
            self.assertTrue(source.extraction.strip_selectors)
            result = _extract(source.source_id, inner)
            self.assertTrue(result.ok, msg=f"{source.source_id}: {result.reason}")
            self.assertNotIn("NAVIGATION_CHROME", result.extracted_text)
            self.assertIn("Title", result.extracted_text)
            chunks = chunk_document(source, result)
            self.assertGreaterEqual(len(chunks), 1)
            self.assertEqual(chunks[0].chunker_version, CHUNKER_VERSION)


class ExtractionTests(unittest.TestCase):
    def test_main_content_selection(self) -> None:
        inner = f"<h1>Path Parameters</h1><p>{FILL}</p>"
        html = _wrap("fastapi", inner)
        html = html.replace(
            "</article>",
            "</article><div class='other'><p>OUTSIDE_MAIN should be ignored</p></div>",
        )
        result = extract_document(get_source("fastapi"), html, URLS["fastapi"])
        self.assertTrue(result.ok)
        self.assertIn("Path Parameters", result.extracted_text)
        self.assertNotIn("OUTSIDE_MAIN", result.extracted_text)

    def test_navigation_footer_script_removal(self) -> None:
        inner = (
            f"<h1>Install</h1><p>{FILL}</p>"
            "<pre><code class='language-python'>print(1)</code>"
            "<button class='copy'>Copy</button></pre>"
        )
        result = _extract("fastapi", inner)
        self.assertTrue(result.ok)
        combined = result.extracted_text
        for chrome in (
            "NAVIGATION_CHROME",
            "HEADER_CHROME",
            "FOOTER_CHROME",
            "SIDEBAR_CHROME",
            "alert(",
            "display:none",
            "<form",
            "Copy",
        ):
            self.assertNotIn(chrome, combined)

    def test_heading_preservation(self) -> None:
        inner = f"<h1>Root</h1><h2>Child</h2><h3>Grandchild</h3><p>{FILL}</p>"
        result = _extract("python", inner)
        self.assertIn("# Root", result.extracted_text)
        self.assertIn("## Child", result.extracted_text)
        self.assertIn("### Grandchild", result.extracted_text)
        self.assertEqual(result.heading_outline, ["Root", "Child", "Grandchild"])

    def test_code_block_preservation(self) -> None:
        code = "def greet(name):\n    return f'hi {name}'\n"
        inner = (
            f"<h1>Example</h1><p>{FILL}</p>"
            f"<pre><code class='language-python'>{code}</code></pre>"
        )
        result = _extract("react", inner)
        self.assertIn("```python", result.extracted_text)
        self.assertIn("def greet(name):", result.extracted_text)
        self.assertIn("    return f'hi {name}'", result.extracted_text)

    def test_malformed_html(self) -> None:
        inner = f"<h1>Broken</h1><p>{FILL}<b>still parsed"
        result = _extract("docker", inner)
        self.assertTrue(result.ok)
        self.assertIn("Broken", result.extracted_text)
        self.assertIn("still parsed", result.extracted_text)

    def test_empty_extraction(self) -> None:
        result = _extract("qdrant", "<p>   </p>")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "empty")

    def test_no_main_node_does_not_use_body(self) -> None:
        html = f"""<html><body><nav>NAV</nav><div class="not-main"><h1>Nope</h1><p>{FILL}</p></div></body></html>"""
        result = extract_document(get_source("fastapi"), html, URLS["fastapi"])
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "extract_failed")
        self.assertEqual(result.extracted_text, "")

    def test_deterministic_normalization(self) -> None:
        inner = f"<h1>Unicode  café </h1><p>{FILL}</p>\n\n\n\n<p>Again.   </p>"
        first = _extract("fastapi", inner)
        second = _extract("fastapi", inner)
        self.assertEqual(first.extracted_text, second.extracted_text)
        self.assertEqual(first.extracted_sha256, second.extracted_sha256)
        nfc = normalize_extracted_text(first.extracted_text)
        self.assertEqual(nfc, first.extracted_text)
        self.assertNotIn("\n\n\n", first.extracted_text)


class IdentityTests(unittest.TestCase):
    def test_deterministic_document_ids(self) -> None:
        url = "https://fastapi.tiangolo.com/tutorial/first-steps/"
        source = get_source("fastapi")
        inner = f"<h1>IDs</h1><p>{FILL}</p>"
        first = extract_document(source, _wrap("fastapi", inner), url)
        second = extract_document(source, _wrap("fastapi", inner), url)
        canonical = "https://fastapi.tiangolo.com/tutorial/first-steps"
        expected = hashlib.sha256(f"fastapi\n{canonical}".encode("utf-8")).hexdigest()
        self.assertEqual(first.document_id, expected)
        self.assertEqual(first.document_id, second.document_id)
        self.assertEqual(make_document_id("fastapi", canonical), expected)

    def test_deterministic_chunk_ids(self) -> None:
        canonical = "https://fastapi.tiangolo.com/tutorial/first-steps"
        expected = hashlib.sha256(f"fastapi{canonical}heading-v10".encode("utf-8")).hexdigest()
        self.assertEqual(make_chunk_id("fastapi", canonical, 0), expected)
        inner = f"<h1>One</h1><p>{FILL}</p><h2>Two</h2><p>{FILL}</p>"
        extracted = _extract("fastapi", inner)
        chunks = chunk_document(get_source("fastapi"), extracted)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].chunk_id, make_chunk_id("fastapi", extracted.canonical_url, 0))
        self.assertEqual(chunks[1].chunk_id, make_chunk_id("fastapi", extracted.canonical_url, 1))
        again = chunk_document(get_source("fastapi"), extracted)
        self.assertEqual([c.chunk_id for c in chunks], [c.chunk_id for c in again])


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        count_tokens("warmup")

    def test_heading_first_chunking(self) -> None:
        inner = f"<h1>Guide</h1><h2>Alpha</h2><p>{FILL}</p><h2>Beta</h2><p>{FILL}</p>"
        extracted = _extract("fastapi", inner)
        chunks = chunk_document(get_source("fastapi"), extracted)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any("Alpha" in c.primary_text for c in chunks))
        self.assertTrue(any("Beta" in c.primary_text for c in chunks))
        alpha = next(c for c in chunks if "Alpha" in c.headings or "Alpha" in c.primary_text)
        self.assertNotIn("## Beta", alpha.primary_text)

    def test_400_token_target(self) -> None:
        words = " ".join(f"tok{i}" for i in range(700))
        inner = f"<h1>Long</h1><p>{words}</p>"
        extracted = _extract("fastapi", inner)
        self.assertTrue(extracted.ok)
        chunks = chunk_document(get_source("fastapi"), extracted)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            if chunk.primary_text.lstrip().startswith("```"):
                continue
            self.assertLessEqual(count_tokens(chunk.primary_text), TARGET_TOKENS)

    def test_50_token_overlap(self) -> None:
        words = " ".join(f"ovl{i}" for i in range(700))
        inner = f"<h1>Overlap</h1><p>{words}</p>"
        extracted = _extract("fastapi", inner)
        chunks = chunk_document(get_source("fastapi"), extracted)
        self.assertGreater(len(chunks), 1)
        expected = overlap_text(chunks[0].primary_text, OVERLAP_TOKENS)
        self.assertTrue(expected)
        self.assertTrue(chunks[1].text.startswith(f"{chunks[1].breadcrumb}\n{expected}"))
        self.assertGreaterEqual(count_tokens(expected), OVERLAP_TOKENS - 8)
        self.assertLessEqual(count_tokens(chunks[1].primary_text), TARGET_TOKENS)

    def test_no_code_block_splitting(self) -> None:
        lines = "\n".join(f"value_{index} = {index}" for index in range(120))
        inner = f"<h1>Code</h1><p>{FILL}</p><pre><code class='language-python'>{lines}</code></pre>"
        extracted = _extract("fastapi", inner)
        chunks = chunk_document(get_source("fastapi"), extracted)
        fence_chunks = [c for c in chunks if "```python" in c.primary_text]
        self.assertTrue(fence_chunks)
        for chunk in fence_chunks:
            self.assertIn("value_0 = 0", chunk.primary_text)
            self.assertIn("value_119 = 119", chunk.primary_text)
            self.assertEqual(chunk.primary_text.count("```"), 2)

    def test_breadcrumb_preservation(self) -> None:
        inner = f"<h1>Tutorial</h1><h2>Path Params</h2><p>{FILL}</p>"
        extracted = _extract("fastapi", inner)
        chunks = chunk_document(get_source("fastapi"), extracted)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertTrue(chunk.text.startswith(chunk.breadcrumb))
            self.assertIn("FastAPI", chunk.breadcrumb)
            self.assertIn(chunk.breadcrumb, chunk.text)

    def test_chunker_version(self) -> None:
        inner = f"<h1>Version</h1><p>{FILL}</p>"
        extracted = _extract("fastapi", inner)
        chunks = chunk_document(get_source("fastapi"), extracted)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertEqual(chunk.chunker_version, "heading-v1")
            self.assertEqual(chunk.chunker_version, CHUNKER_VERSION)


if __name__ == "__main__":
    unittest.main()
