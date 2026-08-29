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
    validate_sitemap_url,
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



class SitemapUrlValidationTests(unittest.TestCase):
    """Verify validate_sitemap_url stays within the registered sitemap scope.

    Critical invariants:
    - Registered sitemap URL is always accepted (allow_child ignored).
    - allow_child=False rejects any URL not in the registered list.
    - allow_child=True permits .xml / sitemap-named paths, but ONLY on a
      host that is in source.allowed_hosts.  A different host is always
      rejected, even when allow_child=True and the path ends with .xml.
    """

    def setUp(self) -> None:
        self.fastapi = get_source("fastapi")
        # fastapi sitemap: https://fastapi.tiangolo.com/sitemap.xml
        # fastapi allowed_hosts: ["fastapi.tiangolo.com"]

    # --- registered URL acceptance ------------------------------------------

    def test_registered_sitemap_url_allowed(self) -> None:
        result = validate_sitemap_url(
            self.fastapi,
            "https://fastapi.tiangolo.com/sitemap.xml",
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, REASON_OK)

    def test_registered_sitemap_url_allows_false_child_flag(self) -> None:
        """Even with allow_child=False the registered URL must pass."""
        result = validate_sitemap_url(
            self.fastapi,
            "https://fastapi.tiangolo.com/sitemap.xml",
            allow_child=False,
        )
        self.assertTrue(result.allowed)

    # --- allow_child=False guards -------------------------------------------

    def test_unregistered_xml_on_same_host_rejected_without_child_flag(self) -> None:
        """An arbitrary .xml on the correct host is rejected when allow_child=False."""
        result = validate_sitemap_url(
            self.fastapi,
            "https://fastapi.tiangolo.com/other-data.xml",
            allow_child=False,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_INVALID_PATH)

    def test_unregistered_non_xml_on_same_host_rejected_without_child_flag(self) -> None:
        result = validate_sitemap_url(
            self.fastapi,
            "https://fastapi.tiangolo.com/tutorial/",
            allow_child=False,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_INVALID_PATH)

    # --- allow_child=True: same-host, acceptable paths ----------------------

    def test_xml_child_on_same_host_allowed_with_child_flag(self) -> None:
        """A .xml path on the registered host is permitted as a sitemap child."""
        result = validate_sitemap_url(
            self.fastapi,
            "https://fastapi.tiangolo.com/sitemap-tutorial.xml",
            allow_child=True,
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, REASON_OK)

    def test_sitemap_named_path_on_same_host_allowed_with_child_flag(self) -> None:
        """A URL with 'sitemap' in the path is accepted as a child sitemap."""
        result = validate_sitemap_url(
            self.fastapi,
            "https://fastapi.tiangolo.com/sitemaps/en.xml",
            allow_child=True,
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, REASON_OK)

    # --- host-pinning: wrong host always rejected ----------------------------

    def test_xml_on_wrong_host_rejected_even_with_child_flag(self) -> None:
        """Host must be in source.allowed_hosts — .xml extension is not enough."""
        result = validate_sitemap_url(
            self.fastapi,
            "https://evil.example/sitemap.xml",
            allow_child=True,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_INVALID_HOST)

    def test_sitemap_subdomain_not_in_allowlist_rejected(self) -> None:
        """A subdomain of the allowed host that is not listed is rejected."""
        result = validate_sitemap_url(
            self.fastapi,
            "https://cdn.fastapi.tiangolo.com/sitemap.xml",
            allow_child=True,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_INVALID_HOST)

    # --- scheme / credential guards are inherited ---------------------------

    def test_http_sitemap_rejected(self) -> None:
        result = validate_sitemap_url(
            self.fastapi,
            "http://fastapi.tiangolo.com/sitemap.xml",
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_HTTPS_REQUIRED)

    def test_sitemap_url_with_credentials_rejected(self) -> None:
        result = validate_sitemap_url(
            self.fastapi,
            "https://user:pw@fastapi.tiangolo.com/sitemap.xml",
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, REASON_CREDENTIALS)


if __name__ == "__main__":
    unittest.main()
