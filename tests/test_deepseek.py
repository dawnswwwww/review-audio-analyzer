"""Tests for the DeepSeek HTTP client."""

import httpx
import pytest
import respx

from audio_interview_analyze.llm.deepseek import (
    DeepSeekClient,
    DeepSeekError,
    DeepSeekRateLimitError,
)


@pytest.fixture
def client() -> DeepSeekClient:
    return DeepSeekClient(api_key="test-key", base_url="https://api.deepseek.com")


@respx.mock
def test_chat_sends_correct_request(client):
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            },
        )
    )
    result = client.chat(
        system="sys",
        user="user",
        model="deepseek-chat",
        temperature=0.3,
        max_tokens=100,
        response_format=None,
    )
    assert result == "hello"
    assert route.called
    sent = route.calls.last.request
    import json as _json

    body = _json.loads(sent.content)
    assert body["model"] == "deepseek-chat"
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 100
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]


@respx.mock
def test_chat_includes_response_format_when_set(client):
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    )
    client.chat(
        system="s",
        user="u",
        model="deepseek-chat",
        temperature=0.3,
        max_tokens=100,
        response_format={"type": "json_object"},
    )
    import json as _json

    body = _json.loads(route.calls.last.request.content)
    assert body["response_format"] == {"type": "json_object"}


@respx.mock
def test_chat_retries_on_5xx(client):
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(503, json={"error": "unavailable"}),
            httpx.Response(503, json={"error": "unavailable"}),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    result = client.chat(
        system="s",
        user="u",
        model="deepseek-chat",
        temperature=0.3,
        max_tokens=10,
        response_format=None,
    )
    assert result == "ok"
    assert route.call_count == 3


@respx.mock
def test_chat_raises_on_401(client):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    with pytest.raises(DeepSeekError, match="authentication"):
        client.chat(
            system="s",
            user="u",
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=10,
            response_format=None,
        )


@respx.mock
def test_chat_raises_on_429_after_retries(client):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": "rate limit"})
    )
    with pytest.raises(DeepSeekRateLimitError):
        client.chat(
            system="s",
            user="u",
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=10,
            response_format=None,
        )
