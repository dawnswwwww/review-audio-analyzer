"""Tests for the Markdown report renderer."""

import json
from pathlib import Path

import pytest

from audio_interview_analyze.report.model import FinalReport
from audio_interview_analyze.report.render import render_markdown


@pytest.fixture
def sample_report() -> FinalReport:
    fixture = Path(__file__).parent / "fixtures" / "sample_final_report.json"
    return FinalReport.model_validate(json.loads(fixture.read_text(encoding="utf-8")))


def test_render_contains_all_six_sections(sample_report):
    md = render_markdown(sample_report)
    for section in [
        "## 1. 面试总评",
        "## 2. 亮点",
        "## 3. 不足",
        "## 4. 逐题分析",
        "## 5. 知识点总结",
        "## 附录：完整转写",
    ]:
        assert section in md, f"Missing section: {section}"


def test_render_includes_overall_score(sample_report):
    md = render_markdown(sample_report)
    assert "**78**" in md or "78" in md
    assert "技术深度" in md
    assert "80" in md  # one of the dimension scores


def test_render_lists_consolidated_highlights(sample_report):
    md = render_markdown(sample_report)
    assert "数据结构基础扎实" in md
    assert "能够区分相似概念" in md


def test_render_lists_consolidated_weaknesses(sample_report):
    md = render_markdown(sample_report)
    assert "缺少时间复杂度分析" in md


def test_render_knowledge_point_with_category_and_directions(sample_report):
    md = render_markdown(sample_report)
    assert "红黑树" in md
    assert "数据结构" in md
    assert "《算法导论》第 13 章" in md


def test_render_per_question_block(sample_report):
    md = render_markdown(sample_report)
    assert "第 1 题" in md
    assert "红黑树与 AVL 树的区别" in md
    assert "回答基本正确" in md


def test_render_handles_empty_highlights_and_weaknesses():
    report = FinalReport(
        overall_evaluation="简洁。",
        overall_score=70,
        dimension_scores={"技术深度": 70, "思路结构": 70, "表达清晰度": 70, "反应速度": 70},
        consolidated_highlights=[],
        consolidated_weaknesses=[],
        consolidated_knowledge_points=[],
        pair_analyses=[],
        full_transcript={"duration": 0.0, "utterances": []},
    )
    md = render_markdown(report)
    assert "（无）" in md


def test_render_includes_transcript_appendix(sample_report):
    md = render_markdown(sample_report)
    assert "讲讲红黑树" in md
    assert "自平衡" in md
    # Speaker labels appear in the transcript
    assert "[面试官]" in md or "面试官" in md
    assert "[候选人]" in md or "候选人" in md


def test_render_uses_markdown_heading_hierarchy(sample_report):
    md = render_markdown(sample_report)
    lines = md.splitlines()
    h1 = [line for line in lines if line.startswith("# ")]
    h2 = [line for line in lines if line.startswith("## ")]
    assert len(h1) >= 1
    assert len(h2) >= 6
