"""Tests for Q+A pair extraction."""

import json
from pathlib import Path

import pytest

from audio_interview_analyze.analysis.qa_pairs import build_qa_pairs, label_speakers
from audio_interview_analyze.report.model import QAPair, Transcript, Utterance


@pytest.fixture
def sample_transcript() -> Transcript:
    fixture = Path(__file__).parent / "fixtures" / "sample_transcript.json"
    return Transcript.model_validate(json.loads(fixture.read_text(encoding="utf-8")))


def test_label_speakers_candidate_talks_more(sample_transcript):
    labeled = label_speakers(sample_transcript)
    speakers = {u.speaker for u in labeled.utterances}
    assert speakers == {"interviewer", "candidate"}


def test_build_qa_pairs_emits_pairs_for_each_question(sample_transcript):
    """Sample has: 1 candidate-only greeting, then 2 merged Q+A pairs
    (red-black + AVL merged into one question, then process vs thread)."""
    labeled = label_speakers(sample_transcript)
    pairs = build_qa_pairs(labeled)
    qa_pairs = [p for p in pairs if p.question]
    assert len(qa_pairs) == 2


def test_build_qa_pairs_merges_consecutive_interviewer_turns(sample_transcript):
    """The two interviewer turns about red-black tree and AVL come before
    any candidate answer, so they should merge into one question per spec
    section 9."""
    labeled = label_speakers(sample_transcript)
    pairs = build_qa_pairs(labeled)
    qa_pairs = [p for p in pairs if p.question]
    assert "红黑树" in qa_pairs[0].question
    assert "AVL" in qa_pairs[0].question
    assert "红黑树是近似平衡" in qa_pairs[0].answer
    assert "AVL 是严格平衡" in qa_pairs[0].answer


def test_build_qa_pairs_pair_index_starts_at_one(sample_transcript):
    labeled = label_speakers(sample_transcript)
    pairs = build_qa_pairs(labeled)
    assert all(p.pair_index >= 1 for p in pairs)
    assert pairs[0].pair_index == 1


def test_build_qa_pairs_skips_empty_question_pairs(sample_transcript):
    """The candidate's greeting has no question, so the pair has empty
    question text and is still emitted (the analyzer is responsible for
    skipping it)."""
    labeled = label_speakers(sample_transcript)
    pairs = build_qa_pairs(labeled)
    empty_q = [p for p in pairs if not p.question]
    assert len(empty_q) == 1
    assert empty_q[0].answer == "你好面试官"


def test_build_qa_pairs_silence_gap_closes_answer():
    """An 8+ second silence in the middle of a candidate's answer closes
    the answer per spec section 9, so the candidate's next utterance
    starts a new pair with an empty question. The empty-question pair
    is marked as ``is_conversation`` since the candidate wasn't asked a
    fresh question."""
    t = Transcript(
        utterances=[
            Utterance(speaker="interviewer", text="讲讲 TCP 三次握手", start=0.0, end=1.0),
            Utterance(speaker="candidate", text="客户端发送 SYN", start=2.0, end=3.0),
            Utterance(speaker="candidate", text="服务端回应 SYN+ACK", start=12.0, end=14.0),  # 9s gap
        ],
        duration=14.0,
    )
    labeled = label_speakers(t)
    pairs = build_qa_pairs(labeled)
    # The silence gap splits the answer into two pairs per spec.
    assert len(pairs) == 2
    assert pairs[0].question == "讲讲 TCP 三次握手"
    assert "客户端发送 SYN" in pairs[0].answer
    assert "SYN+ACK" not in pairs[0].answer  # second candidate utt is in pair 1
    assert pairs[0].is_conversation is False
    assert pairs[1].question == ""
    assert "SYN+ACK" in pairs[1].answer
    assert pairs[1].is_conversation is True


def test_build_qa_pairs_short_gap_keeps_answer_running():
    """A sub-8s gap between candidate utterances should keep the answer
    running, not close it."""
    t = Transcript(
        utterances=[
            Utterance(speaker="interviewer", text="讲讲 TCP", start=0.0, end=1.0),
            Utterance(speaker="candidate", text="第一次握手", start=2.0, end=3.0),
            Utterance(speaker="candidate", text="第二次握手", start=5.0, end=6.0),  # 2s gap
        ],
        duration=6.0,
    )
    labeled = label_speakers(t)
    pairs = build_qa_pairs(labeled)
    assert len(pairs) == 1
    assert "第一次握手" in pairs[0].answer
    assert "第二次握手" in pairs[0].answer


def test_build_qa_pairs_merges_interviewer_interjections():
    """Short interviewer affirmations should not start a new pair."""
    t = Transcript(
        utterances=[
            Utterance(speaker="interviewer", text="Vue 和 React 有什么区别", start=0.0, end=1.0),
            Utterance(speaker="candidate", text="Vue 适合中小型项目", start=2.0, end=3.0),
            Utterance(speaker="interviewer", text="好了", start=4.0, end=4.5),
            Utterance(speaker="candidate", text="React 适合大型项目", start=5.0, end=6.0),
        ],
        duration=6.0,
    )
    labeled = label_speakers(t)
    pairs = build_qa_pairs(labeled)
    assert len(pairs) == 1
    assert pairs[0].question == "Vue 和 React 有什么区别"
    assert "Vue 适合中小型项目" in pairs[0].answer
    assert "React 适合大型项目" in pairs[0].answer


def test_build_qa_pairs_short_answer_kept_not_skipped():
    """The Q+A builder emits all pairs; downstream is responsible for
    skipping short ones. This test documents that contract."""
    t = Transcript(
        utterances=[
            Utterance(speaker="interviewer", text="你叫什么", start=0.0, end=1.0),
            Utterance(speaker="candidate", text="小王", start=2.0, end=3.0),
        ],
        duration=3.0,
    )
    labeled = label_speakers(t)
    pairs = build_qa_pairs(labeled)
    assert len(pairs) == 1
    assert pairs[0].answer == "小王"


def test_merge_qa_pairs_merges_consecutive_groups():
    """_merge_qa_pairs should combine rephrased questions correctly."""
    from audio_interview_analyze.analysis.qa_pairs import _merge_qa_pairs

    pairs = [
        QAPair(question="Q1", answer="A1", q_start=0, q_end=1, a_start=2, a_end=3, pair_index=1),
        QAPair(question="Q1 restated", answer="A1 final", q_start=4, q_end=5, a_start=6, a_end=7, pair_index=2),
        QAPair(question="Q3", answer="A3", q_start=10, q_end=11, a_start=12, a_end=13, pair_index=3),
    ]
    merged = _merge_qa_pairs(pairs, [[1, 2]])
    assert len(merged) == 2
    assert merged[0].pair_index == 1
    assert "Q1" in merged[0].question
    assert "Q1 restated" in merged[0].question
    assert "A1" in merged[0].answer
    assert "A1 final" in merged[0].answer
    assert merged[1].pair_index == 3
