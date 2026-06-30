"""Tests for the transcript-cleaning LLM call."""

import json

import httpx
import pytest
import respx

from audio_interview_analyze.analysis.clean_transcript import clean_transcript
from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.report.model import Transcript, Utterance


@pytest.fixture
def client() -> DeepSeekClient:
    return DeepSeekClient(api_key="k", base_url="https://api.deepseek.com")


@pytest.fixture
def sample_transcript() -> Transcript:
    return Transcript(
        utterances=[
            Utterance(speaker="interviewer", text="讲讲 BUD 框架", start=0.0, end=1.0),
            Utterance(speaker="candidate", text="BUD 是一个渐进式框架", start=1.5, end=3.0),
            Utterance(speaker="interviewer", text="怎么实现灵火", start=4.0, end=5.0),
        ],
        duration=5.0,
    )


def _make_correction_response(entries: list[dict]) -> str:
    """Build a JSON-encoded LLM response with the given corrected entries."""
    return json.dumps(entries, ensure_ascii=False)


@respx.mock
def test_clean_transcript_corrections_applied(client, sample_transcript):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": _make_correction_response(
                                [
                                    {
                                        "index": 0,
                                        "speaker": "interviewer",
                                        "start": 0.0,
                                        "end": 1.0,
                                        "text": "讲讲 Vue 框架",
                                        "corrections": ["BUD → Vue: 上下文是前端框架，Vue 是合理推断"],
                                    },
                                    {
                                        "index": 1,
                                        "speaker": "candidate",
                                        "start": 1.5,
                                        "end": 3.0,
                                        "text": "Vue 是一个渐进式框架",
                                        "corrections": ["BUD → Vue: 同上"],
                                    },
                                    {
                                        "index": 2,
                                        "speaker": "interviewer",
                                        "start": 4.0,
                                        "end": 5.0,
                                        "text": "怎么实现流火",
                                        "corrections": ["灵火 → 流火: 上下文是流处理相关问题"],
                                    },
                                ]
                            )
                        }
                    }
                ]
            },
        )
    )
    cleaned = clean_transcript(client, sample_transcript)
    assert cleaned.utterances[0].text == "讲讲 Vue 框架"
    assert cleaned.utterances[1].text == "Vue 是一个渐进式框架"
    assert cleaned.utterances[2].text == "怎么实现流火"


@respx.mock
def test_clean_transcript_preserves_speaker_and_timing(client, sample_transcript):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": _make_correction_response(
                                [
                                    {
                                        "index": 0,
                                        "speaker": "interviewer",
                                        "start": 0.0,
                                        "end": 1.0,
                                        "text": "讲讲 Vue 框架",
                                        "corrections": ["BUD → Vue"],
                                    },
                                    {
                                        "index": 1,
                                        "speaker": "candidate",
                                        "start": 1.5,
                                        "end": 3.0,
                                        "text": "Vue 是一个渐进式框架",
                                        "corrections": [],
                                    },
                                    {
                                        "index": 2,
                                        "speaker": "interviewer",
                                        "start": 4.0,
                                        "end": 5.0,
                                        "text": "怎么实现流火",
                                        "corrections": ["灵火 → 流火"],
                                    },
                                ]
                            )
                        }
                    }
                ]
            },
        )
    )
    cleaned = clean_transcript(client, sample_transcript)
    for orig, new in zip(sample_transcript.utterances, cleaned.utterances, strict=True):
        assert new.speaker == orig.speaker
        assert new.start == orig.start
        assert new.end == orig.end
    assert cleaned.duration == sample_transcript.duration


@respx.mock
def test_clean_transcript_no_changes_passes_through(client, sample_transcript):
    """When LLM has no corrections, text should be unchanged."""
    entries = [
        {
            "index": i,
            "speaker": u.speaker,
            "start": u.start,
            "end": u.end,
            "text": u.text,
            "corrections": [],
        }
        for i, u in enumerate(sample_transcript.utterances)
    ]
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": _make_correction_response(entries)}}]},
        )
    )
    cleaned = clean_transcript(client, sample_transcript)
    for orig, new in zip(sample_transcript.utterances, cleaned.utterances, strict=True):
        assert new.text == orig.text


@respx.mock
def test_clean_transcript_retries_on_invalid_json(client, sample_transcript):
    """First response invalid JSON → retry succeeds → use the second response."""
    corrected_entries = [
        {
            "index": i,
            "speaker": u.speaker,
            "start": u.start,
            "end": u.end,
            "text": "corrected " + u.text,
            "corrections": ["x → y"],
        }
        for i, u in enumerate(sample_transcript.utterances)
    ]
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": _make_correction_response(corrected_entries)}}
                    ]
                },
            ),
        ]
    )
    cleaned = clean_transcript(client, sample_transcript)
    assert cleaned.utterances[0].text == "corrected 讲讲 BUD 框架"
    assert route.call_count == 2


@respx.mock
def test_clean_transcript_tolerates_wrapped_response(client, sample_transcript):
    """LLM may wrap the array in a dict under common keys."""
    wrapped = {
        "utterances": [
            {
                "index": i,
                "speaker": u.speaker,
                "start": u.start,
                "end": u.end,
                "text": "fixed " + u.text,
                "corrections": [],
            }
            for i, u in enumerate(sample_transcript.utterances)
        ]
    }
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(wrapped, ensure_ascii=False)}}]},
        )
    )
    cleaned = clean_transcript(client, sample_transcript)
    assert cleaned.utterances[0].text == "fixed 讲讲 BUD 框架"


@respx.mock
def test_clean_transcript_uses_response_format_json_object(client, sample_transcript):
    """The LLM call should request JSON-object response format."""
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": _make_correction_response(
                                [
                                    {
                                        "index": i,
                                        "speaker": u.speaker,
                                        "start": u.start,
                                        "end": u.end,
                                        "text": u.text,
                                        "corrections": [],
                                    }
                                    for i, u in enumerate(sample_transcript.utterances)
                                ]
                            )
                        }
                    }
                ]
            },
        )
    )
    clean_transcript(client, sample_transcript)
    body = json.loads(route.calls.last.request.content)
    assert body["response_format"] == {"type": "json_object"}


def test_clean_transcript_empty_passthrough(client):
    """Empty transcript returns the same empty transcript without calling LLM."""
    t = Transcript(utterances=[], duration=0.0)
    cleaned = clean_transcript(client, t)
    assert cleaned.utterances == []
    assert cleaned.duration == 0.0


@respx.mock
def test_clean_transcript_length_mismatch_raises(client, sample_transcript):
    """If the LLM returns a different number of entries, raise clearly."""
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": _make_correction_response(
                                [
                                    {
                                        "index": 0,
                                        "speaker": "interviewer",
                                        "start": 0.0,
                                        "end": 1.0,
                                        "text": "讲讲 Vue 框架",
                                        "corrections": [],
                                    }
                                ]
                            )
                        }
                    }
                ]
            },
        )
    )
    with pytest.raises(ValueError, match="Length mismatch"):
        clean_transcript(client, sample_transcript)
