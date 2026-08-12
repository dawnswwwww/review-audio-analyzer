"""Context-aware chunking for transcript cleaning.

The default fixed-size chunking can split a single technical answer in half,
robbing the LLM of context.  This module splits at speaker changes and long
pauses while still respecting a maximum chunk size.
"""

from __future__ import annotations

from dataclasses import dataclass

from audio_interview_analyze.report.model import Transcript, Utterance

DEFAULT_MAX_CHUNK_UTTERANCES = 30
DEFAULT_MAX_GAP_SECONDS = 2.0
DEFAULT_CONTEXT_WINDOW = 3


@dataclass
class Chunk:
    """A slice of the transcript prepared for cleaning."""

    utterances: list[Utterance]
    start_index: int
    context_before: list[Utterance]
    context_after: list[Utterance]

    @property
    def end_index(self) -> int:
        return self.start_index + len(self.utterances)


def chunk_transcript(
    transcript: Transcript,
    *,
    max_utterances: int = DEFAULT_MAX_CHUNK_UTTERANCES,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> list[Chunk]:
    """Split a transcript into context-aware chunks.

    A new chunk is started when any of the following is true:

    - the current chunk has reached ``max_utterances`` utterances;
    - there is a pause greater than ``max_gap_seconds`` between consecutive
      utterances (a long pause usually marks a topic boundary).

    Speaker changes alone are **not** used as chunk boundaries.  In a Q+A
    interview, splitting on every speaker turn would destroy the context the
    LLM needs to resolve pronouns and topic continuity.

    Each chunk carries a small window of ``context_window`` utterances before
    and after so the LLM can resolve pronouns and topic continuity.
    """
    utterances = transcript.utterances
    if not utterances:
        return []

    boundaries: list[int] = [0]
    for i in range(1, len(utterances)):
        prev = utterances[i - 1]
        cur = utterances[i]
        gap = cur.start - prev.end
        chunk_len = i - boundaries[-1]

        must_split = chunk_len >= max_utterances or gap > max_gap_seconds
        if must_split:
            boundaries.append(i)

    boundaries.append(len(utterances))

    chunks: list[Chunk] = []
    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        before_start = max(0, start - context_window)
        after_end = min(len(utterances), end + context_window)
        chunks.append(
            Chunk(
                utterances=utterances[start:end],
                start_index=start,
                context_before=utterances[before_start:start],
                context_after=utterances[end:after_end],
            )
        )
    return chunks
