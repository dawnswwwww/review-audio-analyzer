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


def _make_pass_through_response(utterances):
    """Build a pass-through response (no corrections) for a chunk."""
    entries = [
        {
            "index": i,
            "speaker": u.speaker,
            "start": u.start,
            "end": u.end,
            "text": u.text,
            "corrections": [],
        }
        for i, u in enumerate(utterances)
    ]
    return _make_correction_response(entries)


@respx.mock
def test_clean_transcript_chunks_large_transcript(client):
    """A 60-utterance transcript with default chunk_size=30 should make
    2 LLM calls; each gets a subset; the cleaned transcript has all 60
    utterances in order."""
    utterances = [
        Utterance(speaker="interviewer", text=f"Q {i}", start=float(i), end=float(i + 1))
        for i in range(60)
    ]
    t = Transcript(utterances=utterances, duration=60.0)

    # Mock returns a pass-through response for whatever chunk the LLM gets.
    # The implementation uses the original chunk's text to verify the response.
    # We just need to return any valid-length response; the test only
    # checks call count and that the cleaned text is sensible.
    def side_effect(request):
        body = json.loads(request.content)
        # Count utterances in the prompt by counting "Q " appearances
        n_in_prompt = body["messages"][1]["content"].count('"text":')
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": _make_pass_through_response(
                                [
                                    Utterance(
                                        speaker="interviewer",
                                        text=f"passthrough {j}",
                                        start=float(j),
                                        end=float(j + 1),
                                    )
                                    for j in range(n_in_prompt)
                                ]
                            )
                        }
                    }
                ]
            },
        )

    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        side_effect=side_effect
    )
    cleaned = clean_transcript(client, t)
    assert len(cleaned.utterances) == 60
    assert route.call_count == 2
    # The mock returns "passthrough j" for each chunk (where j is the
    # index within the chunk), so each utterance's text is "passthrough N"
    # where N is its index within its chunk.
    assert cleaned.utterances[0].text == "passthrough 0"
    assert cleaned.utterances[29].text == "passthrough 29"
    assert cleaned.utterances[30].text == "passthrough 0"
    assert cleaned.utterances[59].text == "passthrough 29"


@respx.mock
def test_clean_transcript_chunking_respects_custom_size(client):
    """Custom chunk_size=10 should make 3 LLM calls for a 30-utterance transcript."""
    utterances = [
        Utterance(speaker="interviewer", text=f"X {i}", start=float(i), end=float(i + 1))
        for i in range(30)
    ]
    t = Transcript(utterances=utterances, duration=30.0)

    def side_effect(request):
        n_in_prompt = json.loads(request.content)["messages"][1]["content"].count('"text":')
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": _make_pass_through_response(
                                [
                                    Utterance(
                                        speaker="interviewer",
                                        text="x",
                                        start=float(j),
                                        end=float(j + 1),
                                    )
                                    for j in range(n_in_prompt)
                                ]
                            )
                        }
                    }
                ]
            },
        )

    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        side_effect=side_effect
    )
    cleaned = clean_transcript(client, t, chunk_size=10)
    assert len(cleaned.utterances) == 30
    assert route.call_count == 3
