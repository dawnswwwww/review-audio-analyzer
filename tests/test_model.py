"""Tests for the Pydantic data models."""

from audio_interview_analyze.report.model import (
    FinalReport,
    KnowledgePoint,
    PairAnalysis,
    QAPair,
    Transcript,
    Utterance,
)


def test_utterance_round_trip():
    u = Utterance(speaker="interviewer", text="What is a binary tree?", start=1.0, end=3.5)
    assert u.model_dump() == {
        "speaker": "interviewer",
        "text": "What is a binary tree?",
        "start": 1.0,
        "end": 3.5,
    }


def test_transcript_round_trip():
    t = Transcript(
        utterances=[
            Utterance(speaker="interviewer", text="Q1", start=0.0, end=1.0),
            Utterance(speaker="candidate", text="A1", start=1.5, end=3.0),
        ],
        duration=3.0,
    )
    restored = Transcript.model_validate(t.model_dump())
    assert restored == t


def test_qa_pair_requires_positive_pair_index():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QAPair(
            question="Q",
            answer="A",
            q_start=0.0,
            q_end=1.0,
            a_start=1.0,
            a_end=2.0,
            pair_index=0,  # must be >= 1
        )


def test_knowledge_point_fields():
    kp = KnowledgePoint(
        name="红黑树",
        category="数据结构",
        explanation="一种自平衡二叉查找树。",
        learning_directions=["阅读《算法导论》第 13 章", "实现一个简单的红黑树"],
    )
    assert kp.name == "红黑树"
    assert kp.category == "数据结构"
    assert len(kp.learning_directions) == 2


def test_final_report_round_trip():
    report = FinalReport(
        overall_evaluation="整体表现良好。",
        overall_score=80,
        dimension_scores={"技术深度": 75, "思路结构": 85, "表达清晰度": 80, "反应速度": 80},
        consolidated_highlights=["基础扎实"],
        consolidated_weaknesses=["系统设计经验不足"],
        consolidated_knowledge_points=[
            KnowledgePoint(
                name="红黑树",
                category="数据结构",
                explanation="自平衡二叉查找树。",
                learning_directions=["读《算法导论》"],
            )
        ],
        pair_analyses=[],
        full_transcript=Transcript(utterances=[], duration=0.0),
    )
    restored = FinalReport.model_validate(report.model_dump())
    assert restored.overall_score == 80
    assert restored.dimension_scores["技术深度"] == 75


def test_pair_analysis_default_lists():
    pa = PairAnalysis(
        pair_index=1,
        question_summary="测试",
        answer_evaluation="回答正确。",
        knowledge_points=[],
        highlights=[],
        weaknesses=[],
        improvement_suggestions="可以更简洁。",
    )
    assert pa.knowledge_points == []
    assert pa.highlights == []
    assert pa.weaknesses == []
