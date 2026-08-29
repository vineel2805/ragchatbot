from __future__ import annotations

import unittest

from app.ingestion.registry import get_source, list_source_ids
from app.ingestion.url_security import (
    REASON_CREDENTIALS,
    REASON_HTTPS_REQUIRED,
    REASON_INVALID_HOST,
    REASON_INVALID_PATH,
    REASON_OK,
    REASON_UNSUPPORTED_SCHEME,
    canonicalize_url,
    validate_redirect,
    validate_url,
)


class RegistryTests(unittest.TestCase):
    def test_five_approved_sources(self) -> None:
        self.assertEqual(
            list_source_ids(),
            ["fastapi", "python", "react", "docker", "qdrant"],
        )

    def test_seeds_are_allowlisted(self) -> None:
        for source_id in list_source_ids():
            source = get_source(source_id)
            for seed in source.seed_urls:
                result = validate_url(source, seed)
                self.assertTrue(
                    result.allowed,
                    f"{source_id} seed rejected: {seed} ({result.reason})",
                )


class UrlAllowlistTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fastapi = get_source("fastapi")
        self.python = get_source("python")
        self.react = get_source("react")
        self.qdrant = get_source("qdrant")

    def test_valid_source_url(self) -> None:
        result = validate_url(
            self.fastapi,
            "https://fastapi.tiangolo.com/tutorial/first-steps/",
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, REASON_OK)
        self.assertEqual(
            result.canonical_url,
            "https://fastapi.tiangolo.com/tutorial/first-steps",
        )

    def test_invalid_host(self) -> None:
        result = validate_url(
            self.fastapi,
            "https://fastapi.tiangolo.com.evil.example/tutorial/",
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_INVALID_HOST)

    def test_wrong_subdomain_rejected(self) -> None:
        result = validate_url(self.fastapi, "https://docs.fastapi.tiangolo.com/tutorial/")
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_INVALID_HOST)

    def test_invalid_path(self) -> None:
        result = validate_url(self.fastapi, "https://fastapi.tiangolo.com/about/")
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_INVALID_PATH)

    def test_prefix_does_not_match_sibling_path(self) -> None:
        result = validate_url(self.react, "https://react.dev/learning")
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_INVALID_PATH)

    def test_http_url_rejected(self) -> None:
        result = validate_url(self.fastapi, "http://fastapi.tiangolo.com/tutorial/")
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_HTTPS_REQUIRED)

    def test_unsupported_scheme_rejected(self) -> None:
        result = validate_url(self.fastapi, "ftp://fastapi.tiangolo.com/tutorial/")
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_UNSUPPORTED_SCHEME)

    def test_url_with_credentials_rejected(self) -> None:
        result = validate_url(
            self.fastapi,
            "https://user:secret@fastapi.tiangolo.com/tutorial/",
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_CREDENTIALS)

    def test_username_only_credentials_rejected(self) -> None:
        result = validate_url(
            self.fastapi,
            "https://user@fastapi.tiangolo.com/tutorial/",
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_CREDENTIALS)

    def test_python_selected_library_allowed(self) -> None:
        result = validate_url(
            self.python,
            "https://docs.python.org/3/library/asyncio-task.html",
        )
        self.assertTrue(result.allowed)

    def test_python_unlisted_library_rejected(self) -> None:
        result = validate_url(
            self.python,
            "https://docs.python.org/3/library/pickle.html",
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_INVALID_PATH)

    def test_qdrant_landing_allowed_but_not_cloud_prefix(self) -> None:
        landing = validate_url(self.qdrant, "https://qdrant.tech/documentation/")
        cloud = validate_url(self.qdrant, "https://qdrant.tech/documentation/cloud/")
        self.assertTrue(landing.allowed)
        self.assertFalse(cloud.allowed)
        self.assertEqual(cloud.reason, REASON_INVALID_PATH)


class CanonicalizationTests(unittest.TestCase):
    def test_url_canonicalization(self) -> None:
        canonical = canonicalize_url(
            "https://FastAPI.Tiangolo.com:443/tutorial//first-steps/?utm_source=x#intro"
        )
        self.assertEqual(
            canonical,
            "https://fastapi.tiangolo.com/tutorial/first-steps",
        )

    def test_query_and_fragment_dropped(self) -> None:
        canonical = canonicalize_url(
            "https://react.dev/learn/thinking-in-react?foo=1&bar=2#hooks"
        )
        self.assertEqual(canonical, "https://react.dev/learn/thinking-in-react")
        self.assertNotIn("?", canonical)
        self.assertNotIn("#", canonical)

    def test_trailing_slash_stripped_except_origin(self) -> None:
        self.assertEqual(
            canonicalize_url("https://docs.docker.com/get-started/"),
            "https://docs.docker.com/get-started",
        )
        self.assertEqual(
            canonicalize_url("https://docs.docker.com/"),
            "https://docs.docker.com/",
        )

    def test_validate_url_uses_canonical_form(self) -> None:
        source = get_source("fastapi")
        result = validate_url(
            source,
            "https://FASTAPI.TIANGOLO.COM/tutorial/first-steps/?q=1#x",
        )
        self.assertTrue(result.allowed)
        self.assertEqual(
            result.canonical_url,
            "https://fastapi.tiangolo.com/tutorial/first-steps",
        )


class RedirectValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fastapi = get_source("fastapi")

    def test_redirect_to_allowed_url(self) -> None:
        result = validate_redirect(
            self.fastapi,
            "https://fastapi.tiangolo.com/tutorial/",
            "https://fastapi.tiangolo.com/tutorial/first-steps/",
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, REASON_OK)
        self.assertEqual(
            result.canonical_url,
            "https://fastapi.tiangolo.com/tutorial/first-steps",
        )

    def test_redirect_relative_location_allowed(self) -> None:
        result = validate_redirect(
            self.fastapi,
            "https://fastapi.tiangolo.com/tutorial/",
            "../advanced/path-operation-advanced-configuration/",
        )
        self.assertTrue(result.allowed)
        self.assertEqual(
            result.canonical_url,
            "https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration",
        )

    def test_redirect_to_disallowed_url(self) -> None:
        result = validate_redirect(
            self.fastapi,
            "https://fastapi.tiangolo.com/tutorial/",
            "https://evil.example/tutorial/",
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_INVALID_HOST)

    def test_redirect_to_http_rejected(self) -> None:
        result = validate_redirect(
            self.fastapi,
            "https://fastapi.tiangolo.com/tutorial/",
            "http://fastapi.tiangolo.com/tutorial/first-steps/",
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_HTTPS_REQUIRED)

    def test_redirect_outside_source_scope(self) -> None:
        result = validate_redirect(
            self.fastapi,
            "https://fastapi.tiangolo.com/tutorial/",
            "https://fastapi.tiangolo.com/deployment/",
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_INVALID_PATH)


if __name__ == "__main__":
    unittest.main()
