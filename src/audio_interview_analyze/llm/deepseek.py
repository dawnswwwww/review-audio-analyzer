"""DeepSeek HTTP client with retry / backoff.

DeepSeek exposes an OpenAI-compatible chat-completions endpoint, so we use
plain httpx rather than the OpenAI SDK. Retry policy: 3 attempts with
exponential backoff (2s, 4s, 8s) on 5xx and 429. 4xx other than 429 are
treated as terminal errors with a clear message.
"""

from __future__ import annotations

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
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_seconds: tuple[float, ...] = (2.0, 4.0, 8.0),
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

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
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, headers=headers, json=body)
            except httpx.HTTPError as e:
                last_error = e
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_seconds[attempt])
                    continue
                raise DeepSeekError(f"HTTP transport error after {self.max_retries} attempts: {e}") from e

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise DeepSeekError("DeepSeek returned no choices")
                return choices[0]["message"]["content"]

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
