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

from audio_interview_analyze.report.model import QAPair, Transcript, Utterance

SILENCE_GAP_SECONDS = 8.0


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
    return pairs
