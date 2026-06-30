"""Tests for the per-pair analysis call."""

import json

import httpx
import pytest
import respx

from audio_interview_analyze.analysis.analyze import analyze_pair
from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.report.model import PairAnalysis, QAPair


@pytest.fixture
def client() -> DeepSeekClient:
    return DeepSeekClient(api_key="k", base_url="https://api.deepseek.com")


@pytest.fixture
def sample_pair() -> QAPair:
    return QAPair(
        question="讲讲红黑树",
        answer="红黑树是一种自平衡二叉查找树",
        q_start=0.0,
        q_end=1.0,
        a_start=2.0,
        a_end=5.0,
        pair_index=1,
    )


@respx.mock
def test_analyze_pair_parses_json_response(client, sample_pair):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "question_summary": "红黑树的定义",
                                    "answer_evaluation": "回答正确。",
                                    "knowledge_points": ["红黑树", "自平衡"],
                                    "highlights": ["简洁"],
                                    "weaknesses": [],
                                    "improvement_suggestions": "可以举例。",
                                }
                            )
                        }
                    }
                ]
            },
        )
    )
    result = analyze_pair(client, sample_pair)
    assert isinstance(result, PairAnalysis)
    assert result.pair_index == 1
    assert result.question_summary == "红黑树的定义"
    assert result.knowledge_points == ["红黑树", "自平衡"]


@respx.mock
def test_analyze_pair_retries_on_invalid_json(client, sample_pair):
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "question_summary": "X",
                                        "answer_evaluation": "Y",
                                        "knowledge_points": [],
                                        "highlights": [],
                                        "weaknesses": [],
                                        "improvement_suggestions": "Z",
                                    }
                                )
                            }
                        }
                    ]
                },
            ),
        ]
    )
    result = analyze_pair(client, sample_pair)
    assert result.question_summary == "X"
    assert route.call_count == 2


@respx.mock
def test_analyze_pair_uses_response_format_json_object(client, sample_pair):
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "question_summary": "X",
                                    "answer_evaluation": "Y",
                                    "knowledge_points": [],
                                    "highlights": [],
                                    "weaknesses": [],
                                    "improvement_suggestions": "Z",
                                }
                            )
                        }
                    }
                ]
            },
        )
    )
    analyze_pair(client, sample_pair)
    import json as _json

    body = _json.loads(route.calls.last.request.content)
    assert body["response_format"] == {"type": "json_object"}