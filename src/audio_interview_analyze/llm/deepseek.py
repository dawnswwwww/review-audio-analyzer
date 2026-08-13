"""DeepSeek HTTP client with retry / backoff.

DeepSeek exposes an OpenAI-compatible chat-completions endpoint, so we use
plain httpx rather than the OpenAI SDK. Retry policy: up to ``max_retries``
attempts with exponential backoff on 5xx, 429, and transport errors. 4xx
other than 429 are treated as terminal errors with a clear message.

The client holds a single shared ``httpx.Client`` (thread-safe, keep-alive
connection pool) so concurrent pipeline stages can issue requests in
parallel without paying a TCP+TLS handshake per call.

The default total timeout is generous (600s): requests are non-streaming,
so the server only starts responding after the full completion is
generated. Large outputs (``max_tokens`` up to 8000) routinely take
several minutes — a short timeout here causes false ReadTimeout failures
followed by exponential-backoff retry storms, which historically turned a
full pipeline run into a multi-hour job.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import httpx


class DeepSeekError(Exception):
    """Base class for DeepSeek client errors."""


class DeepSeekAuthError(DeepSeekError):
    """401 / 403 — bad API key."""


class DeepSeekRateLimitError(DeepSeekError):
    """429 — rate limit, even after retries."""


class DeepSeekClient:
    """Thin wrapper around the DeepSeek chat-completions API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        timeout: float = 600.0,
        connect_timeout: float = 10.0,
        max_retries: int = 8,
        backoff_seconds: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 90.0, 120.0),
        max_connections: int = 32,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        # Use a connect timeout shorter than the total timeout so we
        # don't hang forever on DNS / TCP-level failures.
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=min(connect_timeout, timeout),
                read=timeout,
                write=timeout,
                pool=timeout,
            ),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
        )

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._client.close()

    def chat(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> str:
        """Call DeepSeek chat-completions and return the assistant content string.

        Raises:
            DeepSeekAuthError: on 401 / 403.
            DeepSeekRateLimitError: on 429 after exhausting retries.
            DeepSeekError: on other non-2xx responses.
        """
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format is not None:
            body["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(url, headers=headers, json=body)
            except httpx.HTTPError as e:
                last_error = e
                if attempt + 1 < self.max_retries:
                    sleep_for = self.backoff_seconds[min(attempt, len(self.backoff_seconds) - 1)]
                    print(
                        f"[DeepSeek] {type(e).__name__} on attempt "
                        f"{attempt + 1}/{self.max_retries}, retrying in {sleep_for}s",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_for)
                    continue
                raise DeepSeekError(
                    f"HTTP transport error after {self.max_retries} attempts: {e}"
                ) from e

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise DeepSeekError("DeepSeek returned no choices")
                content = choices[0]["message"].get("content") or ""
                if not content.strip():
                    # Reasoning models occasionally spend the whole output
                    # budget on reasoning_content and leave content empty.
                    # That yields an unparseable response, so retry instead
                    # of returning "" to the caller.
                    last_error = DeepSeekError(
                        "DeepSeek returned empty content (reasoning only?)"
                    )
                    if attempt + 1 < self.max_retries:
                        sleep_for = self.backoff_seconds[
                            min(attempt, len(self.backoff_seconds) - 1)
                        ]
                        print(
                            f"[DeepSeek] empty content on attempt "
                            f"{attempt + 1}/{self.max_retries}, retrying in {sleep_for}s",
                            file=sys.stderr,
                        )
                        time.sleep(sleep_for)
                        continue
                    raise last_error
                return content

            if resp.status_code in (401, 403):
                raise DeepSeekAuthError(
                    "DeepSeek authentication failed. Check DEEPSEEK_API_KEY."
                )
            if resp.status_code == 429:
                last_error = DeepSeekRateLimitError("DeepSeek rate limit hit.")
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_seconds[attempt])
                    continue
                raise last_error
            if 500 <= resp.status_code < 600:
                last_error = DeepSeekError(
                    f"DeepSeek server error: {resp.status_code} {resp.text[:200]}"
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_seconds[attempt])
                    continue
                raise last_error

            raise DeepSeekError(
                f"DeepSeek returned {resp.status_code}: {resp.text[:200]}"
            )

        # Should not be reached, but be defensive.
        if last_error is not None:
            raise last_error
        raise DeepSeekError("DeepSeek client exited retry loop unexpectedly")
