"""Tests for the multi-stage transcript-cleaning pipeline."""

import json

import httpx
import pytest
import respx

from audio_interview_analyze.analysis.chunking import chunk_transcript
from audio_interview_analyze.analysis.clean_transcript import clean_transcript
from audio_interview_analyze.analysis.terminology import (
    CorrectionCandidate,
    TermEntry,
    Terminology,
    build_candidate_summary,
)
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
    return json.dumps({"corrected": entries}, ensure_ascii=False)


def _make_final_validate_response() -> str:
    """Build a final-validation response with no revisions."""
    return json.dumps({"revisions": [], "notes": []}, ensure_ascii=False)


@respx.mock
def test_clean_transcript_corrections_applied(client, sample_transcript):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
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
            ),
            # Review call: confirm corrections.
            httpx.Response(
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
            ),
            # Final validation: no revisions.
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": _make_final_validate_response()}}]},
            ),
        ]
    )
    cleaned = clean_transcript(client, sample_transcript)
    assert cleaned.utterances[0].text == "讲讲 Vue 框架"
    assert cleaned.utterances[1].text == "Vue 是一个渐进式框架"
    assert cleaned.utterances[2].text == "怎么实现流火"


@respx.mock
def test_clean_transcript_preserves_speaker_and_timing(client, sample_transcript):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
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
            ),
            # Review confirms.
            httpx.Response(
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
            ),
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": _make_final_validate_response()}}]},
            ),
        ]
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
        side_effect=[
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": _make_correction_response(entries)}}]},
            ),
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": _make_correction_response(entries)}}]},
            ),
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": _make_final_validate_response()}}]},
            ),
        ]
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
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
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
            # Review call.
            httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": _make_correction_response(corrected_entries)}}
                    ]
                },
            ),
            # Final validation.
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": _make_final_validate_response()}}]},
            ),
        ]
    )
    cleaned = clean_transcript(client, sample_transcript)
    assert cleaned.utterances[0].text == "corrected 讲讲 BUD 框架"


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
        side_effect=[
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps(wrapped, ensure_ascii=False)}}]},
            ),
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps(wrapped, ensure_ascii=False)}}]},
            ),
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": _make_final_validate_response()}}]},
            ),
        ]
    )
    cleaned = clean_transcript(client, sample_transcript)
    assert cleaned.utterances[0].text == "fixed 讲讲 BUD 框架"


@respx.mock
def test_clean_transcript_uses_response_format_json_object(client, sample_transcript):
    """The LLM call should request JSON-object response format."""
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
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
            ),
            httpx.Response(
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
            ),
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": _make_final_validate_response()}}]},
            ),
        ]
    )
    clean_transcript(client, sample_transcript)
    for call in route.calls:
        body = json.loads(call.request.content)
        assert body["response_format"] == {"type": "json_object"}


def test_clean_transcript_empty_passthrough(client):
    """Empty transcript returns the same empty transcript without calling LLM."""
    t = Transcript(utterances=[], duration=0.0)
    cleaned = clean_transcript(client, t)
    assert cleaned.utterances == []
    assert cleaned.duration == 0.0


@respx.mock
def test_clean_transcript_length_mismatch_falls_back(client, sample_transcript):
    """If the LLM keeps returning the wrong entry count, degrade gracefully
    instead of failing the whole cleaning run."""
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
    cleaned = clean_transcript(client, sample_transcript)
    # The mismatched chunk is left uncorrected; the run completes.
    assert [u.text for u in cleaned.utterances] == [
        u.text for u in sample_transcript.utterances
    ]
    assert len(cleaned.utterances) == len(sample_transcript.utterances)


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
    2 correction + 2 review + 1 final validation calls."""
    utterances = [
        Utterance(speaker="interviewer", text=f"Q {i}", start=float(i), end=float(i + 1))
        for i in range(60)
    ]
    t = Transcript(utterances=utterances, duration=60.0)

    def side_effect(request):
        # The mock returns a pass-through for whatever target chunk size the
        # implementation asks for.  Because the prompt also includes context,
        # we cannot simply count "text" occurrences; instead we trust the
        # chunking to produce two 30-utterance target chunks.
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
                                    for j in range(30)
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
    # 2 correct + 2 review; final validation is skipped when no corrections.
    assert route.call_count == 4
    assert cleaned.utterances[0].text == "passthrough 0"
    assert cleaned.utterances[29].text == "passthrough 29"
    assert cleaned.utterances[30].text == "passthrough 0"
    assert cleaned.utterances[59].text == "passthrough 29"


@respx.mock
def test_clean_transcript_chunking_respects_custom_size(client):
    """Custom chunk_size=10 should make 3 correction + 3 review + 1 final calls."""
    utterances = [
        Utterance(speaker="interviewer", text=f"X {i}", start=float(i), end=float(i + 1))
        for i in range(30)
    ]
    t = Transcript(utterances=utterances, duration=30.0)

    def side_effect(request):
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
                                    for j in range(10)
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
    # 3 correct + 3 review; final validation skipped when no corrections.
    assert route.call_count == 6


@respx.mock
def test_clean_transcript_can_disable_review_and_final_validation(client, sample_transcript):
    """When review/final validation are disabled, only correction calls are made."""
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
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": _make_correction_response(entries)}}]},
        )
    )
    cleaned = clean_transcript(
        client,
        sample_transcript,
        enable_review=False,
        enable_final_validation=False,
    )
    assert route.call_count == 1
    assert len(cleaned.utterances) == 3


def test_terminology_exact_match():
    term = Terminology([TermEntry(correct="前端", aliases=["前段"])])
    cands = term.find_candidates("我搞前段的", 0)
    assert len(cands) == 1
    assert cands[0].suggested == "前端"
    assert cands[0].source == "terminology_exact"


def test_terminology_phonetic_match():
    term = Terminology([TermEntry(correct="Sequelize", aliases=["撕扣"])])
    cands = term.find_candidates("撕扣", 0)
    assert any(c.suggested == "Sequelize" for c in cands)


def test_build_candidate_summary_dedupes():
    cands = [
        CorrectionCandidate("前段", "前端", 0, "terminology_exact", 0.9),
        CorrectionCandidate("前段", "前端", 1, "terminology_exact", 0.9),
    ]
    summary = build_candidate_summary(cands)
    assert len(summary) == 1
    assert summary[0]["original"] == "前段"


def test_chunk_transcript_respects_gap():
    utterances = [
        Utterance(speaker="a", text="x", start=0.0, end=1.0),
        Utterance(speaker="a", text="y", start=1.5, end=2.0),
        Utterance(speaker="a", text="z", start=10.0, end=11.0),
    ]
    t = Transcript(utterances=utterances, duration=11.0)
    chunks = chunk_transcript(t, max_utterances=10, max_gap_seconds=2.0)
    assert len(chunks) == 2
    assert chunks[0].start_index == 0
    assert chunks[1].start_index == 2


def test_chunk_transcript_keeps_speaker_turn_together():
    utterances = [
        Utterance(speaker="a", text="q", start=0.0, end=1.0),
        Utterance(speaker="b", text="a", start=1.5, end=2.0),
        Utterance(speaker="a", text="f", start=2.5, end=3.0),
    ]
    t = Transcript(utterances=utterances, duration=3.0)
    chunks = chunk_transcript(t, max_utterances=10, max_gap_seconds=2.0)
    assert len(chunks) == 1
    assert len(chunks[0].utterances) == 3
