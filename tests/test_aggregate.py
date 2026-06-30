"""Tests for the aggregator (LLM #2) and its token-overflow summarizer."""

import json

import httpx
import pytest
import respx

from audio_interview_analyze.analysis.aggregate import (
    AGGREGATE_MAX_INPUT_TOKENS,
    aggregate,
    maybe_summarize,
)
from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.report.model import PairAnalysis


@pytest.fixture
def client() -> DeepSeekClient:
    return DeepSeekClient(api_key="k", base_url="https://api.deepseek.com")


@pytest.fixture
def many_pair_analyses() -> list[PairAnalysis]:
    """Build a list large enough to exceed the token threshold when serialized."""
    return [
        PairAnalysis(
            pair_index=i,
            question_summary=f"题目 {i} 考察的知识点：包括数据结构、算法分析以及对应的工程实现细节问题",
            answer_evaluation=(
                "回答涉及多个方面，整体表现尚可但缺少深入分析。"
                "候选人在描述方案时没有给出复杂度论证，也没有讨论边界条件和异常路径。"
                "在追问阶段对扩展性问题的回答偏弱，建议补充更多的系统设计经验。"
            ),
            knowledge_points=[f"知识点-{i}-A", f"知识点-{i}-B", f"知识点-{i}-C"],
            highlights=[f"亮点 {i}"],
            weaknesses=[f"不足 {i}"],
            improvement_suggestions=(
                f"建议 {i}：可以先复述题意并给出 2-3 个候选方案再分析取舍；"
                "同时补充时间空间复杂度、边界用例与可扩展性讨论。"
            ),
        )
        for i in range(1, 800)
    ]


@respx.mock
def test_aggregate_parses_full_report(client):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "overall_evaluation": "整体表现良好。",
                                    "overall_score": 82,
                                    "dimension_scores": {
                                        "技术深度": 80,
                                        "思路结构": 85,
                                        "表达清晰度": 80,
                                        "反应速度": 83,
                                    },
                                    "consolidated_highlights": ["基础扎实"],
                                    "consolidated_weaknesses": ["深度不足"],
                                    "consolidated_knowledge_points": [
                                        {
                                            "name": "红黑树",
                                            "category": "数据结构",
                                            "explanation": "自平衡二叉查找树。",
                                            "learning_directions": ["读《算法导论》"],
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        )
    )
    pairs = [
        PairAnalysis(
            pair_index=1,
            question_summary="红黑树",
            answer_evaluation="回答正确。",
            knowledge_points=["红黑树"],
            highlights=[],
            weaknesses=[],
            improvement_suggestions="补充复杂度。",
        )
    ]
    report = aggregate(client, pairs, candidate_background="")
    assert report.overall_score == 82
    assert report.dimension_scores["技术深度"] == 80
    assert report.consolidated_knowledge_points[0].name == "红黑树"
    assert report.pair_analyses == pairs


@respx.mock
def test_aggregate_retries_on_invalid_json(client):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
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
                                        "overall_evaluation": "X",
                                        "overall_score": 70,
                                        "dimension_scores": {
                                            "技术深度": 70,
                                            "思路结构": 70,
                                            "表达清晰度": 70,
                                            "反应速度": 70,
                                        },
                                        "consolidated_highlights": [],
                                        "consolidated_weaknesses": [],
                                        "consolidated_knowledge_points": [],
                                    }
                                )
                            }
                        }
                    ]
                },
            ),
        ]
    )
    pairs = [
        PairAnalysis(
            pair_index=1,
            question_summary="Q",
            answer_evaluation="A",
            knowledge_points=[],
            highlights=[],
            weaknesses=[],
            improvement_suggestions="S",
        )
    ]
    report = aggregate(client, pairs, candidate_background="")
    assert report.overall_score == 70


@respx.mock
def test_maybe_summarize_skipped_when_small(client):
    """When the serialized pair list is small, summarizer should not be called."""
    pairs = [
        PairAnalysis(
            pair_index=1,
            question_summary="Q",
            answer_evaluation="A",
            knowledge_points=[],
            highlights=[],
            weaknesses=[],
            improvement_suggestions="S",
        )
    ]
    result = maybe_summarize(client, pairs)
    assert result == pairs  # unchanged


@respx.mock
def test_maybe_summarize_called_when_large(client, many_pair_analyses):
    summarized = [
        {
            "pair_index": p.pair_index,
            "question_summary": p.question_summary,
            "answer_evaluation": "压缩后的评价。",
            "knowledge_points": p.knowledge_points,
        }
        for p in many_pair_analyses
    ]
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(summarized)}}]},
        )
    )
    result = maybe_summarize(client, many_pair_analyses)
    assert len(result) == len(many_pair_analyses)
    assert all("answer_evaluation" in item for item in result)
    # The summarized form should be smaller than the original PairAnalysis dicts
    original_size = sum(len(json.dumps(p.model_dump())) for p in many_pair_analyses)
    summarized_size = sum(len(json.dumps(item)) for item in result)
    assert summarized_size < original_size


def test_aggregate_max_input_tokens_constant_is_reasonable():
    """The constant should be positive and well below DeepSeek's 64K context."""
    assert 10_000 < AGGREGATE_MAX_INPUT_TOKENS < 60_000
