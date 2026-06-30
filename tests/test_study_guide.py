"""Tests for the study-guide generation and rendering."""

import json

import httpx
import pytest
import respx

from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.report.model import (
    FinalReport,
    KnowledgePoint,
    PairAnalysis,
    Transcript,
)
from audio_interview_analyze.study.generate import generate_study_guide
from audio_interview_analyze.study.render import render_study_guide


@pytest.fixture
def client() -> DeepSeekClient:
    return DeepSeekClient(api_key="k", base_url="https://api.deepseek.com")


@pytest.fixture
def sample_report() -> FinalReport:
    return FinalReport(
        overall_evaluation="整体表现尚可。",
        overall_score=70,
        dimension_scores={"技术深度": 70, "思路结构": 70, "表达清晰度": 70, "反应速度": 70},
        consolidated_highlights=[],
        consolidated_weaknesses=[],
        consolidated_knowledge_points=[
            KnowledgePoint(
                name="React Fiber 架构",
                category="前端框架",
                explanation="React 16 引入的协调引擎。",
                learning_directions=["读 React 源码"],
            ),
            KnowledgePoint(
                name="单点登录（SSO）",
                category="系统设计",
                explanation="一种身份认证机制。",
                learning_directions=["OAuth 2.0 协议"],
            ),
        ],
        pair_analyses=[
            PairAnalysis(
                pair_index=1,
                question_summary="Q",
                answer_evaluation="A",
                knowledge_points=[],
                highlights=[],
                weaknesses=[],
                improvement_suggestions="",
            )
        ],
        full_transcript=Transcript(utterances=[], duration=0.0),
    )


def _sample_study_guide_response() -> dict:
    return {
        "guide_title": "前端面试核心知识点学习指南",
        "preface": "本场面试覆盖了前端框架和系统设计两个方向。",
        "knowledge_points": [
            {
                "name": "React Fiber 架构",
                "category": "前端框架",
                "one_liner": "React 16 的协调引擎，把渲染拆成可中断的小任务。",
                "why_interview": "考察对 React 核心架构的理解，区分资深/初中级。",
                "core_concepts": [
                    "Fiber 节点是链表结构",
                    "支持优先级调度",
                    "可中断恢复",
                    "双缓冲机制",
                ],
                "code_example": "function fiberNode() {\n  return { child: null, sibling: null };\n}",
                "code_language": "javascript",
                "common_mistakes": [
                    "以为 Fiber 是数据结构而非调度机制",
                    "混淆 Fiber 与 Virtual DOM",
                    "不知道时间分片（time slicing）",
                ],
                "self_check": [
                    {
                        "question": "Fiber 节点之间如何连接？",
                        "answer": "通过 child / sibling / return 形成链表树。",
                    },
                    {
                        "question": "为什么 React 要用 Fiber 重写 stack reconciler？",
                        "answer": "为了支持可中断渲染与优先级调度。",
                    },
                ],
                "further_reading": [
                    "React 官方文档：Reconciliation",
                    "Lin Clark 的 Fiber 介绍演讲",
                ],
            },
            {
                "name": "单点登录（SSO）",
                "category": "系统设计",
                "one_liner": "用一套凭据登录多个系统。",
                "why_interview": "考察分布式身份认证的理解。",
                "core_concepts": [
                    "认证中心统一签发票据",
                    "客户端存储 token",
                    "跨域跳转",
                ],
                "code_example": "GET /sso/login?redirect=https://app.example.com",
                "code_language": "bash",
                "common_mistakes": [
                    "混淆 SSO 与 OAuth",
                    "不知道 token 怎么传递",
                    "忽略安全细节（如 state 参数）",
                ],
                "self_check": [
                    {
                        "question": "SSO 和 OAuth 的区别？",
                        "answer": "SSO 是目标，OAuth 是实现手段之一。",
                    },
                    {
                        "question": "为什么 SSO 登录要带 state 参数？",
                        "answer": "防止 CSRF 攻击。",
                    },
                ],
                "further_reading": [
                    "OAuth 2.0 RFC 6749",
                    "OpenID Connect 规范",
                ],
            },
        ],
    }


@respx.mock
def test_generate_study_guide_returns_typed_structure(client, sample_report):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(_sample_study_guide_response(), ensure_ascii=False)}}
                ]
            },
        )
    )
    guide = generate_study_guide(client, sample_report)
    assert guide["guide_title"] == "前端面试核心知识点学习指南"
    assert len(guide["knowledge_points"]) == 2
    kp = guide["knowledge_points"][0]
    assert kp["name"] == "React Fiber 架构"
    assert kp["code_example"]  # not empty
    assert len(kp["common_mistakes"]) == 3
    assert len(kp["self_check"]) == 2


@respx.mock
def test_generate_study_guide_sends_knowledge_points_in_prompt(client, sample_report):
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(_sample_study_guide_response())}}
                ]
            },
        )
    )
    generate_study_guide(client, sample_report)
    import json as _json
    body = _json.loads(route.calls.last.request.content)
    user_msg = body["messages"][1]["content"]
    assert "React Fiber 架构" in user_msg
    assert "单点登录" in user_msg


@respx.mock
def test_generate_study_guide_retries_on_invalid_json(client, sample_report):
    guide = _sample_study_guide_response()
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": json.dumps(guide, ensure_ascii=False)}}
                    ]
                },
            ),
        ]
    )
    out = generate_study_guide(client, sample_report)
    assert out["guide_title"] == "前端面试核心知识点学习指南"


def test_generate_study_guide_empty_kps_returns_minimal(client):
    """An empty FinalReport.knowledge_points list should still produce
    a valid (minimal) guide without calling the LLM."""
    empty_report = FinalReport(
        overall_evaluation="",
        overall_score=0,
        dimension_scores={},
        consolidated_highlights=[],
        consolidated_weaknesses=[],
        consolidated_knowledge_points=[],
        pair_analyses=[],
        full_transcript=Transcript(utterances=[], duration=0.0),
    )
    guide = generate_study_guide(client, empty_report)
    assert guide["knowledge_points"] == []


def test_render_study_guide_contains_all_sections(sample_report):
    guide = _sample_study_guide_response()
    md = render_study_guide(guide)
    # Title and preface
    assert "前端面试核心知识点学习指南" in md
    assert "本场面试覆盖了前端框架和系统设计两个方向" in md
    # Per-KP sections
    assert "### React Fiber 架构" in md
    assert "### 单点登录" in md
    # Required content
    assert "**一句话核心**" in md
    assert "**为什么面试爱问**" in md
    assert "**核心概念**" in md
    assert "**代码示例**" in md
    assert "**反面教材**" in md or "**常见误区" in md
    assert "**自检**" in md
    assert "**进一步学习**" in md or "**拓展学习**" in md


def test_render_study_guide_includes_code_block(sample_report):
    guide = _sample_study_guide_response()
    md = render_study_guide(guide)
    # Code block with language
    assert "```javascript" in md
    assert "fiberNode" in md
    assert "```bash" in md
    assert "/sso/login" in md


def test_render_study_guide_includes_self_check_qa(sample_report):
    guide = _sample_study_guide_response()
    md = render_study_guide(guide)
    assert "Fiber 节点之间如何连接" in md
    assert "通过 child / sibling / return 形成链表树" in md
    assert "为什么 React 要用 Fiber 重写" in md


def test_render_study_guide_groups_by_category(sample_report):
    guide = _sample_study_guide_response()
    md = render_study_guide(guide)
    # Category headers as H2 (or some grouping)
    assert "## 前端框架" in md or "### 前端框架" in md
    assert "## 系统设计" in md or "### 系统设计" in md
