"""Clean a transcript by fixing ASR-misheard technical terms.

After Whisper transcribes audio, the resulting text often contains errors
where technical terms (framework names, algorithm names, etc.) are
misheard as phonetically-similar but wrong words ("BUD" → "Vue",
"灵火" → "流火", etc.). This module uses the LLM to proofread the
transcript before downstream Q+A extraction.

The cleaning stage preserves ``speaker`` / ``start`` / ``end`` and
utterance order; only ``text`` is corrected.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from audio_interview_analyze.analysis.prompts import (
    CLEAN_TRANSCRIPT_SYSTEM,
    clean_transcript_prompt,
)
from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.report.model import Transcript, Utterance

CLEAN_TRANSCRIPT_MODEL = "deepseek-chat"
# Low temperature: we want faithful corrections, not creative rewrites.
CLEAN_TRANSCRIPT_TEMPERATURE = 0.2
CLEAN_TRANSCRIPT_MAX_TOKENS = 8000
DEFAULT_CHUNK_SIZE = 30
RETRY_SUFFIX = "\n请确保输出是合法 JSON。"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _clean_chunk(
    client: DeepSeekClient,
    chunk: list[Utterance],
    domain: str,
) -> list[Utterance]:
    """Clean a single chunk of utterances via one LLM call."""
    payload = [
        {
            "index": i,
            "speaker": u.speaker,
            "start": u.start,
            "end": u.end,
            "text": u.text,
        }
        for i, u in enumerate(chunk)
    ]
    user_msg = clean_transcript_prompt(
        domain=domain,
        transcript_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )

    last_raw: str | None = None
    last_error: Exception | None = None
    for attempt in range(2):
        suffix = RETRY_SUFFIX if attempt == 1 else ""
        raw = client.chat(
            system=CLEAN_TRANSCRIPT_SYSTEM,
            user=user_msg + suffix,
            model=CLEAN_TRANSCRIPT_MODEL,
            temperature=CLEAN_TRANSCRIPT_TEMPERATURE,
            max_tokens=CLEAN_TRANSCRIPT_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        last_raw = raw
        try:
            data = json.loads(_strip_code_fence(raw))
            return _extract_cleaned(chunk, data)
        except (json.JSONDecodeError, ValidationError, ValueError, KeyError) as e:
            last_error = e
            continue

    raise ValueError(
        f"clean_transcript chunk failed after 2 attempts: {last_error}. "
        f"Last raw response: {(last_raw or '')[:500]}"
    )


def clean_transcript(
    client: DeepSeekClient,
    transcript: Transcript,
    *,
    domain: str = "软件工程 / 前端开发 / AI Agent",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Transcript:
    """Call DeepSeek to fix ASR errors in the transcript text.

    Long transcripts are split into ``chunk_size``-utterance chunks and
    cleaned independently. Each chunk's output must fit within
    ``CLEAN_TRANSCRIPT_MAX_TOKENS`` (8K for deepseek-chat); a 30-utterance
    chunk produces ~5-6 KB of JSON, leaving headroom.

    Returns a new ``Transcript`` with corrected ``text`` fields; speaker,
    start, end, duration, and utterance order are preserved.

    Args:
        client: Configured DeepSeek client.
        transcript: The transcript to clean.
        domain: Free-text description of the interview domain, used to
            give the LLM context for technical-term corrections.
        chunk_size: Number of utterances per LLM call. Lower for very
            verbose utterances, higher for terse ones.

    Raises:
        ValueError: If both attempts of any chunk fail to produce valid
            JSON, or the LLM returns a different number of entries than
            the chunk size.
    """
    if not transcript.utterances:
        return transcript

    if len(transcript.utterances) <= chunk_size:
        cleaned = _clean_chunk(client, transcript.utterances, domain)
        return Transcript(utterances=cleaned, duration=transcript.duration)

    # Chunk by utterance count, preserving order
    cleaned_all: list[Utterance] = []
    for i in range(0, len(transcript.utterances), chunk_size):
        chunk = transcript.utterances[i : i + chunk_size]
        cleaned_chunk = _clean_chunk(client, chunk, domain)
        cleaned_all.extend(cleaned_chunk)

    return Transcript(utterances=cleaned_all, duration=transcript.duration)


def _extract_cleaned(chunk: list[Utterance], data: dict | list) -> list[Utterance]:
    """Apply the LLM's corrections to a single chunk.

    Accepts either a list directly or a dict wrapping the list under a
    common key (``utterances`` / ``items`` / ``results`` / ``corrected``).
    """
    if isinstance(data, dict):
        for key in ("utterances", "items", "results", "corrected"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data).__name__}")

    if len(data) != len(chunk):
        raise ValueError(
            f"Length mismatch: LLM returned {len(data)} entries, chunk has {len(chunk)} utterances"
        )

    cleaned: list[Utterance] = []
    for orig, item in zip(chunk, data, strict=True):
        new_text = item.get("text", orig.text)
        if not isinstance(new_text, str):
            raise ValueError(f"Expected text to be str, got {type(new_text).__name__}")
        cleaned.append(
            Utterance(
                speaker=orig.speaker,
                text=new_text,
                start=orig.start,
                end=orig.end,
            )
        )
    return cleaned
