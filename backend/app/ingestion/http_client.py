from __future__ import annotations

from collections.abc import Mapping

import httpx

from app.ingestion.fetch_models import HttpExchange
from app.ingestion.sanitize import sanitize_error_message

DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 20.0
DEFAULT_WRITE_TIMEOUT = 10.0
DEFAULT_POOL_TIMEOUT = 5.0

HTML_ACCEPT = "text/html,application/xhtml+xml;q=0.9"
ROBOTS_ACCEPT = "text/plain,*/*;q=0.1"
XML_ACCEPT = "application/xml,text/xml,application/xhtml+xml;q=0.8,*/*;q=0.1"


def default_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=DEFAULT_CONNECT_TIMEOUT,
        read=DEFAULT_READ_TIMEOUT,
        write=DEFAULT_WRITE_TIMEOUT,
        pool=DEFAULT_POOL_TIMEOUT,
    )


class HttpClient:
    """Reusable HTTPS client. Redirects are not followed; callers validate targets."""

    def __init__(
        self,
        *,
        user_agent: str,
        timeout: httpx.Timeout | None = None,
        transport: httpx.BaseTransport | None = None,
        accept: str = HTML_ACCEPT,
    ) -> None:
        self.user_agent = user_agent
        self._client = httpx.Client(
            headers={
                "User-Agent": user_agent,
                "Accept": accept,
            },
            timeout=timeout or default_timeout(),
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get(
        self,
        url: str,
        *,
        max_bytes: int,
        accept: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> HttpExchange:
        headers: dict[str, str] = {}
        if accept is not None:
            headers["Accept"] = accept
        if extra_headers:
            for key, value in extra_headers.items():
                if key.lower() in {"authorization", "cookie", "proxy-authorization"}:
                    continue
                headers[key] = value

        try:
            with self._client.stream("GET", url, headers=headers or None) as response:
                response_headers = {
                    key.lower(): value
                    for key, value in response.headers.items()
                    if key.lower() not in {"set-cookie", "authorization"}
                }
                status = response.status_code
                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        if int(content_length) > max_bytes:
                            return HttpExchange(
                                status_code=status,
                                headers=response_headers,
                                content=None,
                                size_exceeded=True,
                            )
                    except ValueError:
                        pass

                if status in {301, 302, 303, 307, 308}:
                    return HttpExchange(
                        status_code=status,
                        headers=response_headers,
                        content=b"",
                    )

                buf = bytearray()
                for chunk in response.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        return HttpExchange(
                            status_code=status,
                            headers=response_headers,
                            content=None,
                            size_exceeded=True,
                        )
                return HttpExchange(
                    status_code=status,
                    headers=response_headers,
                    content=bytes(buf),
                )
        except httpx.TimeoutException as exc:
            return HttpExchange(
                status_code=None,
                headers={},
                content=None,
                error_type="timeout",
                error_message=sanitize_error_message(exc),
            )
        except httpx.NetworkError as exc:
            return HttpExchange(
                status_code=None,
                headers={},
                content=None,
                error_type="connection_failure",
                error_message=sanitize_error_message(exc),
            )
        except httpx.HTTPError as exc:
            return HttpExchange(
                status_code=None,
                headers={},
                content=None,
                error_type="http_error",
                error_message=sanitize_error_message(exc),
            )
