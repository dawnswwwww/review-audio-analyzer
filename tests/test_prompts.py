"""Tests for prompt templates."""

from audio_interview_analyze.analysis.prompts import (
    aggregate_prompt,
    analyze_pair_prompt,
    summarize_pairs_prompt,
)


def test_analyze_pair_prompt_substitutes_question_and_answer():
    user_msg = analyze_pair_prompt(
        question="讲讲红黑树",
        answer="红黑树是一种自平衡的二叉查找树",
    )
    assert "讲讲红黑树" in user_msg
    assert "红黑树是一种自平衡的二叉查找树" in user_msg
    assert "{{" not in user_msg
    assert "}}" not in user_msg


def test_analyze_pair_prompt_includes_json_schema():
    user_msg = analyze_pair_prompt(question="Q", answer="A")
    for field in [
        "question_summary",
        "answer_evaluation",
        "knowledge_points",
        "highlights",
        "weaknesses",
        "improvement_suggestions",
    ]:
        assert field in user_msg, f"Missing field: {field}"


def test_aggregate_prompt_includes_all_pair_summaries():
    pair_analyses = [
        {
            "pair_index": 1,
            "question_summary": "Q1",
            "answer_evaluation": "E1",
            "knowledge_points": ["KP1"],
            "highlights": ["H1"],
            "weaknesses": ["W1"],
            "improvement_suggestions": "S1",
        }
    ]
    user_msg = aggregate_prompt(pair_analyses=pair_analyses, candidate_background="")
    assert "Q1" in user_msg
    assert "E1" in user_msg
    assert "KP1" in user_msg
    assert "{{" not in user_msg
    assert "}}" not in user_msg


def test_aggregate_prompt_handles_empty_background():
    user_msg = aggregate_prompt(pair_analyses=[], candidate_background="")
    assert "未提供" in user_msg


def test_aggregate_prompt_includes_scoring_dimensions():
    user_msg = aggregate_prompt(pair_analyses=[], candidate_background="")
    for dim in ["技术深度", "思路结构", "表达清晰度", "反应速度"]:
        assert dim in user_msg, f"Missing dimension: {dim}"


def test_aggregate_prompt_includes_output_fields():
    user_msg = aggregate_prompt(pair_analyses=[], candidate_background="")
    for field in [
        "overall_evaluation",
        "overall_score",
        "dimension_scores",
        "consolidated_highlights",
        "consolidated_weaknesses",
        "consolidated_knowledge_points",
    ]:
        assert field in user_msg, f"Missing field: {field}"


def test_summarize_pairs_prompt_compresses_pair_analyses():
    pair_analyses = [
        {
            "pair_index": 1,
            "question_summary": "红黑树与 AVL 树的区别",
            "answer_evaluation": "回答基本正确但缺少复杂度分析。",
            "knowledge_points": ["红黑树", "AVL 树"],
            "highlights": ["提到了自平衡"],
            "weaknesses": ["没有分析时间复杂度"],
            "improvement_suggestions": "补充复杂度分析。",
        }
    ]
    user_msg = summarize_pairs_prompt(pair_analyses=pair_analyses)
    assert "红黑树与 AVL 树的区别" in user_msg
    assert "红黑树" in user_msg
    assert "{{" not in user_msg


def test_summarize_pairs_prompt_includes_compression_schema():
    user_msg = summarize_pairs_prompt(pair_analyses=[])
    for field in ["pair_index", "question_summary", "answer_evaluation", "knowledge_points"]:
        assert field in user_msg, f"Missing field: {field}"