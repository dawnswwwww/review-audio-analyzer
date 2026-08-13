"""Extract Q+A pairs from a labeled transcript.

Heuristic (see spec section 9):
- The speaker with more total speech time is the ``candidate``; the other
  is the ``interviewer``. Robust for technical rounds.
- Walk utterances in order. A new pair starts at each interviewer utterance.
- Consecutive interviewer utterances before any candidate answer are merged
  into the same question.
- An 8+ second silence between utterances closes the current pair (treat
  the gap as a natural Q-end).
- Consecutive candidate utterances with sub-8s gaps are appended to the
  running answer until the next interviewer turn or silence gap.
- The builder never silently drops pairs (including ones with empty
  questions like greetings); downstream is responsible for skipping them.
"""

from __future__ import annotations

from collections import defaultdict

from audio_interview_analyze.analysis.prompts import split_qa_prompt
from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.report.model import QAPair, Transcript, Utterance

SILENCE_GAP_SECONDS = 8.0

# Interviewer utterances that are just affirmations / back-channelling.
# These should not start a new Q+A pair; instead they are absorbed into the
# ongoing answer.
_INTERJECTIONS = {
    "嗯",
    "啊",
    "哦",
    "对",
    "好",
    "行",
    "是",
    "好了",
    "说吧",
    "说了吧",
    "然后呢",
    "可以",
    "OK",
    "ok",
    "嗯对",
    "对对",
}


def _is_interjection(text: str) -> bool:
    """Return True if the text is a short back-channel / affirmation."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped in _INTERJECTIONS:
        return True
    # Pure repetition of a single char like "对对对" or "嗯嗯".
    if len(stripped) >= 2 and len(set(stripped)) == 1:
        return True
    return False


def _merge_interjection_pairs(pairs: list[QAPair]) -> list[QAPair]:
    """Merge pairs whose 'question' is just an interjection into the previous pair.

    Preserves the ``is_conversation`` flag of the head pair so casual
    exchanges stay grouped together.
    """
    if not pairs:
        return []

    merged: list[QAPair] = [pairs[0]]
    for p in pairs[1:]:
        if _is_interjection(p.question) and merged:
            last = merged[-1]
            merged[-1] = QAPair(
                question=last.question,
                answer=(last.answer + " " + p.answer).strip(),
                q_start=last.q_start,
                q_end=last.q_end,
                a_start=last.a_start,
                a_end=p.a_end,
                pair_index=last.pair_index,
                is_conversation=last.is_conversation,
            )
        else:
            # Re-index so pair_index stays sequential after merges.
            p = p.model_copy(update={"pair_index": len(merged) + 1})
            merged.append(p)
    return merged


def label_speakers(transcript: Transcript) -> Transcript:
    """Assign ``interviewer`` / ``candidate`` labels based on total speech time.

    Speakers are identified by the existing ``speaker`` values (typically
    ``"SPEAKER_00"``, ``"SPEAKER_01"`` from pyannote). The one with the
    larger cumulative duration becomes ``"candidate"``; the other becomes
    ``"interviewer"``. Tie-broken by first appearance for stability.

    If speakers are already labeled as ``"interviewer"`` / ``"candidate"``,
    they are preserved as-is (no re-labeling).
    """
    existing = {u.speaker for u in transcript.utterances}
    if existing <= {"interviewer", "candidate"}:
        return transcript

    durations: dict[str, float] = defaultdict(float)
    for u in transcript.utterances:
        durations[u.speaker] += max(0.0, u.end - u.start)

    first_seen: dict[str, int] = {}
    for i, u in enumerate(transcript.utterances):
        if u.speaker not in first_seen:
            first_seen[u.speaker] = i
    ranked = sorted(durations.keys(), key=lambda s: (-durations[s], first_seen[s]))

    mapping: dict[str, str] = {}
    if ranked:
        mapping[ranked[0]] = "candidate"
    for s in ranked[1:]:
        mapping[s] = "interviewer"

    labeled = [
        Utterance(
            speaker=mapping.get(u.speaker, "unknown"),
            text=u.text,
            start=u.start,
            end=u.end,
        )
        for u in transcript.utterances
    ]
    return Transcript(utterances=labeled, duration=transcript.duration)


def build_qa_pairs(transcript: Transcript) -> list[QAPair]:
    """Group utterances into Q+A pairs.

    The returned list is ordered by appearance. The first pair's
    ``pair_index`` is 1.

    Closing conditions (any one closes the current pair):
    1. A new ``interviewer`` utterance appears.
    2. A silence gap > ``SILENCE_GAP_SECONDS`` since the previous utterance.
    3. End of input.
    """
    pairs: list[QAPair] = []
    current_question: list[Utterance] = []
    current_answer: list[Utterance] = []
    in_question = False
    in_answer = False
    last_end: float | None = None

    def close_pair() -> None:
        nonlocal in_question, in_answer
        if not current_question and not current_answer:
            return
        pair_index_local = len(pairs) + 1
        q_text = " ".join(u.text.strip() for u in current_question).strip()
        a_text = " ".join(u.text.strip() for u in current_answer).strip()
        q_start = current_question[0].start if current_question else current_answer[0].start
        q_end = current_question[-1].end if current_question else current_answer[0].start
        a_start = current_answer[0].start if current_answer else q_end
        a_end = current_answer[-1].end if current_answer else q_end
        pairs.append(
            QAPair(
                question=q_text,
                answer=a_text,
                q_start=q_start,
                q_end=q_end,
                a_start=a_start,
                a_end=a_end,
                pair_index=pair_index_local,
            )
        )
        current_question.clear()
        current_answer.clear()
        in_question = False
        in_answer = False

    for u in transcript.utterances:
        # Silence gap rule: closes any in-progress pair.
        if last_end is not None and (u.start - last_end) > SILENCE_GAP_SECONDS:
            close_pair()

        if u.speaker == "interviewer":
            if in_answer:
                close_pair()
            current_question.append(u)
            in_question = True
        elif u.speaker == "candidate":
            current_answer.append(u)
            in_answer = True
        # "unknown" speakers are skipped.

        last_end = u.end

    close_pair()
    return _classify_and_merge_pairs(pairs)


# Phrases that signal the "question" is just chitchat / back-channeling
# rather than something that deserves technical evaluation.
_BACKCHANNEL_QUESTIONS = {
    "",
    "嗯",
    "啊",
    "哦",
    "对",
    "好",
    "是",
    "对吧",
    "对呃",
    "对对",
    "对吗",
    "好的",
    "是吗",
    "然后呢",
    "这样",
    "行",
    "然后",
    "这样啊",
    "OK",
    "ok",
    "Yeah",
    "嗯嗯",
}

# Question markers — a string that contains one of these is a real
# interview question that deserves evaluation.
_QUESTION_MARKERS = (
    "什么",
    "怎么",
    "为何",
    "为什么",
    "如何",
    "是否",
    "能不能",
    "有没有",
    "讲讲",
    "说一下",
    "讲一下",
    "聊一下",
    "介绍",
    "聊聊",
    "分析",
    "解释",
    "比较",
    "区别",
    "谈谈",
    "怎么看",
    "如何看",
    "?",
    "？",
    "能",
    "会",
    "要",
    "有",  # 包含"有没有"已经能覆盖
    "哪些",
    "如何去",
    "怎么做",
    "为什么",
    "区别",
)


def _is_real_question(question: str, answer: str) -> bool:
    """Return True if the pair looks like a real interview question that
    deserves technical evaluation.

    Returns False for:
    - empty / very short questions
    - back-channel utterances (just "嗯" / "对" / etc.)
    - self-introduction prompts
    - candidate asking interviewer about the team
    - business / logistics questions unrelated to technical evaluation
    """
    q = question.strip()
    a = answer.strip()

    if not q or len(q) < 4:
        return False
    # The 10-char answer filter is too aggressive — short technical
    # answers (e.g. "客户端发送 SYN" from a half-sentence) are valid.
    # We only treat short answers as conversation if the question
    # itself is a known back-channel.
    if q in _BACKCHANNEL_QUESTIONS:
        return False

    # Self-introduction prompt (very narrow check)
    if q in ("做个自我介绍", "简单自我介绍", "自我介绍下", "自我介绍", "简单自我介绍一下"):
        return False

    # Contains any question marker
    if any(marker in q for marker in _QUESTION_MARKERS):
        return True

    # Long questions almost always contain implicit questions
    if len(q) >= 15:
        return True

    # Otherwise: borderline — treat as conversation
    return False


def _classify_and_merge_pairs(pairs: list[QAPair]) -> list[QAPair]:
    """Tag each pair with ``is_conversation`` and merge back-to-back
    conversation pairs into single entries.
    """
    classified: list[QAPair] = []
    for p in pairs:
        is_question = _is_real_question(p.question, p.answer)
        if not is_question and classified and classified[-1].is_conversation:
            # Merge into previous conversation entry. Both pairs are
            # conversation so the merged result is also conversation.
            prev = classified[-1]
            merged = QAPair(
                question=prev.question,
                answer=(prev.answer + " " + p.answer).strip(),
                q_start=prev.q_start,
                q_end=p.q_end,
                a_start=prev.a_start,
                a_end=p.a_end,
                pair_index=prev.pair_index,
                is_conversation=True,
            )
            classified[-1] = merged
        else:
            p_dict = p.model_dump()
            p_dict["is_conversation"] = not is_question
            classified.append(QAPair(**p_dict))
    return _merge_interjection_pairs(classified)


SPLIT_QA_MODEL = "deepseek-v4-flash"
SPLIT_QA_TEMPERATURE = 0.2
SPLIT_QA_MAX_TOKENS = 2000


def _call_llm_json(client: DeepSeekClient, system: str, user: str) -> dict:
    """Call the LLM with JSON response_format, retry once on parse failure."""
    import json as _json
    import re as _re

    fence_re = _re.compile(r"^```(?:json)?\s*|\s*```$", _re.MULTILINE)
    last_error: Exception | None = None
    for attempt in range(2):
        suffix = "\n请确保输出是合法 JSON。" if attempt == 1 else ""
        raw = client.chat(
            system=system,
            user=user + suffix,
            model=SPLIT_QA_MODEL,
            temperature=SPLIT_QA_TEMPERATURE,
            max_tokens=SPLIT_QA_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        try:
            cleaned = fence_re.sub("", raw).strip()
            return _json.loads(cleaned)
        except _json.JSONDecodeError as e:
            last_error = e
            continue
    raise ValueError(f"split_qa failed to parse JSON: {last_error}")


def _merge_qa_pairs(
    pairs: list[QAPair], merge_groups: list[list[int]]
) -> list[QAPair]:
    """Merge pairs according to merge_groups.

    Each group is a list of consecutive pair_index values that should be combined
    into a single pair.  The earliest pair's index is kept; subsequent pairs are
    absorbed (question merged, answer appended).
    """
    if not merge_groups:
        return pairs

    # Build a mapping from pair_index -> target_index
    target_of: dict[int, int] = {}
    for group in merge_groups:
        if not group:
            continue
        first = group[0]
        for idx in group:
            target_of[idx] = first

    # Group pairs by target index
    groups: dict[int, list[QAPair]] = {}
    for p in pairs:
        tgt = target_of.get(p.pair_index, p.pair_index)
        groups.setdefault(tgt, []).append(p)

    merged: list[QAPair] = []
    for original_idx in sorted(groups.keys()):
        items = groups[original_idx]
        if len(items) == 1:
            merged.append(items[0])
            continue
        # Use the earliest item's pair_index
        head = items[0]
        q_text = " ".join(p.question for p in items if p.question).strip()
        a_text = " ".join(p.answer for p in items if p.answer).strip()
        merged.append(
            QAPair(
                question=q_text,
                answer=a_text,
                q_start=head.q_start,
                q_end=items[-1].q_end,
                a_start=head.a_start,
                a_end=items[-1].a_end,
                pair_index=head.pair_index,
            )
        )
    return merged


def split_qa_with_llm(
    client: DeepSeekClient,
    pairs: list[QAPair],
    *,
    max_pairs: int = 80,
) -> list[QAPair]:
    """Use the LLM to merge Q&A pairs that actually belong to the same
    interview question.

    Only pairs with non-empty questions are presented to the LLM.  If
    `len(pairs)` exceeds ``max_pairs``, only the first ``max_pairs`` are
    considered (downstream short pairs are still kept).
    """
    if len(pairs) <= 1:
        return pairs

    from audio_interview_analyze.analysis.prompts import SPLIT_QA_SYSTEM

    candidates = [p for p in pairs if p.question.strip()]
    if len(candidates) <= 1:
        return pairs

    candidates_to_send = candidates[:max_pairs]
    payload = [
        {
            "pair_index": p.pair_index,
            "question": p.question,
            "answer": p.answer[:500],
        }
        for p in candidates_to_send
    ]
    user_msg = split_qa_prompt(pairs=payload)
    try:
        data = _call_llm_json(client, SPLIT_QA_SYSTEM, user_msg)
    except ValueError:
        return pairs

    raw_groups = data.get("merge_groups", [])
    if not isinstance(raw_groups, list):
        return pairs

    # Filter to valid consecutive groups (group members must be sorted and
    # consecutive pair indices).
    valid_groups: list[list[int]] = []
    for group in raw_groups:
        if not isinstance(group, list) or len(group) < 2:
            continue
        try:
            nums = sorted({int(x) for x in group})
        except (TypeError, ValueError):
            continue
        if nums != list(range(nums[0], nums[-1] + 1)):
            continue
        valid_groups.append(nums)

    if not valid_groups:
        return pairs

    return _merge_qa_pairs(pairs, valid_groups)


def build_qa_pairs_with_llm(
    client: DeepSeekClient,
    transcript: Transcript,
    *,
    chunk_size: int = 60,
    max_gap_seconds: float = SILENCE_GAP_SECONDS,
) -> list[QAPair]:
    """Build Q&A pairs then use the LLM to merge rephrased questions."""
    labeled = label_speakers(transcript)
    candidates = build_qa_pairs(labeled)
    return split_qa_with_llm(client, candidates)
