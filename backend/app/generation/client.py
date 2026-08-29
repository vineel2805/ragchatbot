from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Default request timeout for the provider (seconds).
DEFAULT_TIMEOUT = 30.0

# OpenRouter chat completions endpoint.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_COMPLETIONS_PATH = "/chat/completions"


# ---------------------------------------------------------------------------
# Protocol — injectable for tests
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface for a chat-completion LLM provider.

    Implementations MUST:
    - Accept a *system* and a *user* message.
    - Return the model's reply as a plain string.
    - Never surface secrets (API keys) in raised exceptions.
    - Raise :class:`~app.generation.errors.ProviderError` (or a subclass)
      on any provider failure.
    """

    def complete(self, system: str, user: str) -> str:
        """Return the model completion text for the given messages."""
        ...


# ---------------------------------------------------------------------------
# OpenRouter implementation (all heavy imports are lazy)
# ---------------------------------------------------------------------------


class OpenRouterClient:
    """Chat completion client backed by OpenRouter.

    All HTTP work is done with ``httpx`` (already a project dependency).
    The ``httpx`` import is deferred to :meth:`complete` so that importing
    this module is cheap and safe in test environments without the package.

    Security notes
    --------------
    - The API key is read from config and stored as a private attribute.
    - The key is NEVER logged or included in exception messages.
    - A redacted placeholder ``"sk-or-…[redacted]"`` is used in debug output.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = OPENROUTER_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not api_key:
            # Raise immediately — not worth attempting any request.
            raise ValueError("OpenRouter API key must not be empty.")
        self.__api_key = api_key   # double-underscore: name-mangled, harder to log
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def complete(self, system: str, user: str) -> str:
        """Send a chat completion request and return the reply text.

        Raises
        ------
        app.generation.errors.RateLimitError
            On HTTP 429.
        app.generation.errors.TimeoutError
            On request timeout.
        app.generation.errors.ProviderError
            On any other HTTP error or connection failure.
        app.generation.errors.MalformedResponseError
            When the response JSON cannot be parsed or has no content.
        """
        # Lazy imports — not pulled in at module load time.
        import httpx

        from app.generation.errors import (
            MalformedResponseError,
            ProviderError,
            RateLimitError,
            TimeoutError as GenTimeoutError,
        )

        url = self._base_url + OPENROUTER_COMPLETIONS_PATH
        headers = {
            "Authorization": f"Bearer {self.__api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/devdocs-rag",
            "X-Title": "DevDocs RAG",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        logger.debug("OpenRouter request model=%s", self._model)

        try:
            response = httpx.post(
                url, json=payload, headers=headers, timeout=self._timeout
            )
        except httpx.TimeoutException as exc:
            raise GenTimeoutError(
                f"Request to OpenRouter timed out after {self._timeout}s."
            ) from exc
        except httpx.RequestError as exc:
            # Connection error — exc message may contain the URL but not the key.
            raise ProviderError(f"Network error contacting provider: {type(exc).__name__}") from exc

        if response.status_code == 429:
            raise RateLimitError("OpenRouter rate limit exceeded (HTTP 429).")

        if response.status_code >= 400:
            raise ProviderError(
                f"Provider returned HTTP {response.status_code}."
            )

        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise MalformedResponseError(
                f"Could not parse provider response: {exc}"
            ) from exc

        if not text or not text.strip():
            raise MalformedResponseError("Provider returned an empty completion.")

        logger.debug("OpenRouter response received length=%d", len(text))
        return text


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_openrouter_client(
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> OpenRouterClient:
    """Wire up an :class:`OpenRouterClient` from config/env.

    ``api_key`` is read from ``Settings.openrouter_api_key`` when not
    explicitly supplied.  Raises :class:`ValueError` if no key is available.
    """
    from app.core.config import get_settings

    settings = get_settings()
    key = api_key or settings.openrouter_api_key
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY is not configured. "
            "Set it in .env or as an environment variable."
        )
    resolved_model = model or settings.openrouter_model
    resolved_base = base_url or OPENROUTER_BASE_URL
    return OpenRouterClient(
        api_key=key, model=resolved_model, base_url=resolved_base, timeout=timeout
    )
