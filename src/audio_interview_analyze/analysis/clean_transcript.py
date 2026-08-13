"""Clean a transcript by fixing ASR-misheard technical terms.

After Whisper transcribes audio, the resulting text often contains errors
where technical terms (framework names, algorithm names, etc.) are
misheard as phonetically-similar but wrong words ("BUD" → "Vue",
"灵火" → "流火", etc.). This module uses a multi-stage LLM pipeline to
proofread the transcript before downstream Q+A extraction.

Pipeline (top-to-bottom, context flows forward):
  1. Context-aware chunking (speaker boundaries + pauses).
  2. Candidate marking using an optional terminology dictionary.
  3. LLM correction per chunk, with previous corrections as reference.
  4. LLM review of original-vs-corrected per chunk.
  5. Compose full transcript.
  6. Final global consistency validation, with optional revisions.

The cleaning stage preserves ``speaker`` / ``start`` / ``end`` and
utterance order; only ``text`` is corrected.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from audio_interview_analyze.analysis.chunking import chunk_transcript
from audio_interview_analyze.analysis.prompts import (
    CORRECT_CHUNK_SYSTEM,
    FINAL_VALIDATE_SYSTEM,
    REVIEW_CHUNK_SYSTEM,
    correct_chunk_prompt,
    final_validate_prompt,
    review_chunk_prompt,
)
from audio_interview_analyze.analysis.terminology import (
    CorrectionCandidate,
    Terminology,
    build_candidate_summary,
)
from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.report.model import Transcript, Utterance

logger = logging.getLogger(__name__)

CLEAN_TRANSCRIPT_MODEL = "deepseek-v4-flash"
# Low temperature: we want faithful corrections, not creative rewrites.
CLEAN_TRANSCRIPT_TEMPERATURE = 0.2
CORRECT_MAX_TOKENS = 8000
REVIEW_MAX_TOKENS = 8000
FINAL_VALIDATE_MAX_TOKENS = 8000
DEFAULT_CHUNK_SIZE = 30
DEFAULT_MAX_GAP_SECONDS = 2.0
DEFAULT_CONTEXT_WINDOW = 3
RETRY_SUFFIX = "\n请确保输出是合法 JSON。"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


_CorrectionRecord = dict[str, str | int]


def _call_llm_json(
    client: DeepSeekClient,
    system: str,
    user: str,
    max_tokens: int,
    response_format: dict[str, str] | None = None,
) -> dict:
    """Call the LLM and parse JSON, retrying once on parse failure."""
    last_raw: str | None = None
    last_error: Exception | None = None
    for attempt in range(2):
        suffix = RETRY_SUFFIX if attempt == 1 else ""
        raw = client.chat(
            system=system,
            user=user + suffix,
            model=CLEAN_TRANSCRIPT_MODEL,
            temperature=CLEAN_TRANSCRIPT_TEMPERATURE,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        last_raw = raw
        try:
            return json.loads(_strip_code_fence(raw))
        except json.JSONDecodeError as e:
            last_error = e
            continue

    raise ValueError(
        f"LLM call failed after 2 attempts: {last_error}. "
        f"Last raw response: {(last_raw or '')[:500]}"
    )


def _to_utterance_dicts(
    utterances: list[Utterance], start_index: int
) -> list[dict[str, object]]:
    return [
        {
            "index": start_index + i,
            "speaker": u.speaker,
            "start": u.start,
            "end": u.end,
            "text": u.text,
        }
        for i, u in enumerate(utterances)
    ]


def _parse_correction_string(correction: str) -> tuple[str, str, str]:
    """Parse '原词 → 改后词：依据' into (original, corrected, reason)."""
    if "→" not in correction:
        return correction, "", ""
    original, rest = correction.split("→", 1)
    original = original.strip()
    if "：" in rest:
        corrected, reason = rest.split("：", 1)
    elif ":" in rest:
        corrected, reason = rest.split(":", 1)
    else:
        corrected, reason = rest, ""
    return original, corrected.strip(), reason.strip()


def _extract_chunk_correction_records(
    chunk: list[Utterance],
    corrected_data: dict | list,
    chunk_start_index: int,
) -> tuple[list[Utterance], list[_CorrectionRecord]]:
    """Apply LLM output to a chunk and build structured correction records."""
    if isinstance(corrected_data, dict):
        for key in ("corrected", "utterances", "items", "results"):
            if key in corrected_data and isinstance(corrected_data[key], list):
                corrected_data = corrected_data[key]
                break
    if not isinstance(corrected_data, list):
        raise ValueError(f"Expected list, got {type(corrected_data).__name__}")

    if len(corrected_data) != len(chunk):
        raise ValueError(
            f"Length mismatch: LLM returned {len(corrected_data)} entries, "
            f"chunk has {len(chunk)} utterances. "
            f"First few returned: {corrected_data[:3]!r}"
        )

    cleaned: list[Utterance] = []
    records: list[_CorrectionRecord] = []
    for i, (orig, item) in enumerate(zip(chunk, corrected_data, strict=True)):
        new_text = item.get("text", orig.text)
        if not isinstance(new_text, str):
            raise ValueError(
                f"Expected text to be str, got {type(new_text).__name__}"
            )
        cleaned.append(
            Utterance(
                speaker=orig.speaker,
                text=new_text,
                start=orig.start,
                end=orig.end,
            )
        )

        index = chunk_start_index + i
        corrections = item.get("corrections", [])
        if not isinstance(corrections, list):
            corrections = []
        for corr in corrections:
            if not isinstance(corr, str) or not corr.strip():
                continue
            orig_word, corr_word, reason = _parse_correction_string(corr)
            if orig_word and corr_word:
                records.append(
                    {
                        "index": index,
                        "original": orig_word,
                        "corrected": corr_word,
                        "reason": reason,
                    }
                )
    return cleaned, records


def _extract_with_retry(
    client: DeepSeekClient,
    system: str,
    user_msg: str,
    max_tokens: int,
    target: list[Utterance],
    start_index: int,
    *,
    retries: int = 3,
) -> tuple[list[Utterance], list[_CorrectionRecord]]:
    """Call the LLM and extract correction records, retrying on mismatch.

    The LLM occasionally returns a different number of entries than the chunk
    has (a known nondeterminism with chunked JSON output). Re-calling usually
    realigns the output; if it still fails after ``retries`` attempts, the
    ``ValueError`` propagates so the caller can decide how to degrade.
    """
    last_error: ValueError | None = None
    for _ in range(retries):
        data = _call_llm_json(
            client,
            system,
            user_msg,
            max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            return _extract_chunk_correction_records(target, data, start_index)
        except ValueError as e:
            last_error = e
    assert last_error is not None
    raise last_error


def _apply_revisions(
    utterances: list[Utterance],
    revisions: list[dict],
) -> list[Utterance]:
    """Apply final-validation revisions to the full transcript."""
    revision_map: dict[int, str] = {}
    for rev in revisions:
        if not isinstance(rev, dict):
            continue
        idx = rev.get("index")
        new_text = rev.get("new_text")
        if isinstance(idx, int) and isinstance(new_text, str):
            revision_map[idx] = new_text

    if not revision_map:
        return utterances

    result: list[Utterance] = []
    for i, u in enumerate(utterances):
        if i in revision_map:
            result.append(
                Utterance(
                    speaker=u.speaker,
                    text=revision_map[i],
                    start=u.start,
                    end=u.end,
                )
            )
        else:
            result.append(u)
    return result


def _correct_chunk(
    client: DeepSeekClient,
    chunk,
    terminology: Terminology,
    previous_records: list[_CorrectionRecord],
    domain: str,
) -> tuple[list[Utterance], list[_CorrectionRecord]]:
    """Run correction LLM on a single chunk."""
    all_utterances = chunk.context_before + chunk.utterances + chunk.context_after
    start_index = chunk.start_index - len(chunk.context_before)
    utterance_dicts = _to_utterance_dicts(all_utterances, start_index)

    target_start = len(chunk.context_before)
    target_end = target_start + len(chunk.utterances)
    context_before = utterance_dicts[:target_start]
    target_utterances = utterance_dicts[target_start:target_end]
    context_after = utterance_dicts[target_end:]

    # Candidate marking across the whole chunk (including context) so the LLM
    # can see why a term might be suspicious.
    candidates: list[CorrectionCandidate] = []
    for u in all_utterances:
        idx = start_index + all_utterances.index(u)
        candidates.extend(terminology.find_candidates(u.text, idx))

    user_msg = correct_chunk_prompt(
        domain=domain,
        previous_corrections=previous_records[-50:],  # keep token budget sane
        candidate_terms=build_candidate_summary(candidates),
        context_before=context_before,
        target_utterances=target_utterances,
        context_after=context_after,
    )

    try:
        corrected, records = _extract_with_retry(
            client,
            CORRECT_CHUNK_SYSTEM,
            user_msg,
            CORRECT_MAX_TOKENS,
            chunk.utterances,
            chunk.start_index,
        )
    except ValueError as e:
        # Graceful degradation: keep the chunk uncorrected rather than fail
        # the whole cleaning run on a transient LLM glitch.
        logger.warning(
            "Chunk correction failed after retries (%s); keeping original text", e
        )
        corrected, records = list(chunk.utterances), []
    return corrected, records


def _review_chunk(
    client: DeepSeekClient,
    chunk,
    original: list[Utterance],
    corrected: list[Utterance],
    terminology: Terminology,
    previous_records: list[_CorrectionRecord],
    domain: str,
) -> tuple[list[Utterance], list[_CorrectionRecord]]:
    """Run review LLM on a single chunk."""
    candidates: list[CorrectionCandidate] = []
    for u in original:
        idx = chunk.start_index + original.index(u)
        candidates.extend(terminology.find_candidates(u.text, idx))

    review_items = [
        {
            "index": chunk.start_index + i,
            "speaker": orig.speaker,
            "start": orig.start,
            "end": orig.end,
            "original_text": orig.text,
            "corrected_text": corr.text,
            "corrections": [
                f"{r['original']} → {r['corrected']}：{r['reason']}"
                for r in previous_records
                if r["index"] == chunk.start_index + i
            ],
        }
        for i, (orig, corr) in enumerate(zip(original, corrected, strict=True))
    ]

    user_msg = review_chunk_prompt(
        domain=domain,
        previous_corrections=previous_records[-50:],
        candidate_terms=build_candidate_summary(candidates),
        review_items=review_items,
    )

    try:
        reviewed, records = _extract_with_retry(
            client,
            REVIEW_CHUNK_SYSTEM,
            user_msg,
            REVIEW_MAX_TOKENS,
            original,
            chunk.start_index,
        )
    except ValueError as e:
        logger.warning(
            "Chunk review failed after retries (%s); keeping corrected text", e
        )
        reviewed, records = list(original), []
    return reviewed, records


def _final_validate(
    client: DeepSeekClient,
    transcript: Transcript,
    all_records: list[_CorrectionRecord],
    domain: str,
) -> Transcript:
    """Run a final global consistency validation and apply revisions."""
    if not all_records:
        return transcript

    # Build a compact full-text view.  Include every changed utterance plus a
    # small window, so the model sees global context without blowing the token
    # budget.
    changed_indices = {r["index"] for r in all_records if isinstance(r["index"], int)}
    window = 2
    included: set[int] = set()
    for idx in changed_indices:
        included.update(range(max(0, idx - window), min(len(transcript.utterances), idx + window + 1)))

    full_text_lines = [
        {"index": i, "text": transcript.utterances[i].text}
        for i in sorted(included)
    ]

    user_msg = final_validate_prompt(
        domain=domain,
        all_corrections=all_records,
        full_text_lines=full_text_lines,
    )

    data = _call_llm_json(
        client,
        FINAL_VALIDATE_SYSTEM,
        user_msg,
        FINAL_VALIDATE_MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    revisions = data.get("revisions", []) if isinstance(data, dict) else []
    if not isinstance(revisions, list):
        revisions = []

    revised_utterances = _apply_revisions(transcript.utterances, revisions)
    return Transcript(utterances=revised_utterances, duration=transcript.duration)


def clean_transcript(
    client: DeepSeekClient,
    transcript: Transcript,
    *,
    domain: str = "软件工程 / 前端开发 / AI Agent",
    terminology: Terminology | str | Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    enable_review: bool = True,
    enable_final_validation: bool = True,
    concurrency: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Transcript:
    """Clean a transcript using a multi-stage LLM pipeline.

    Args:
        client: Configured DeepSeek client.
        transcript: The transcript to clean.
        domain: Free-text description of the interview domain.
        terminology: Optional ``Terminology`` instance or path to a JSON
            terminology file.  If provided, terms are used to pre-mark
            candidates for the LLM.
        chunk_size: Maximum utterances per chunk.
        max_gap_seconds: Start a new chunk when the pause exceeds this value
            or when the speaker changes after such a pause.
        context_window: Number of utterances before/after each chunk to
            include as context.
        enable_review: Whether to run the per-chunk review stage.
        enable_final_validation: Whether to run the final global validation
            stage.
        concurrency: Number of chunks to process in parallel. ``1`` (the
            default) preserves the original strictly-sequential behaviour:
            each chunk's prompts include the full correction history of all
            earlier chunks. With ``concurrency > 1`` each chunk instead sees
            a snapshot of the corrections completed at the moment it starts
            — the per-chunk prompts, review stage, and final global
            validation are otherwise unchanged.
        progress_callback: Optional callable invoked with
            ``(chunks_done, total_chunks)`` as chunks finish.

    Returns:
        A new ``Transcript`` with corrected ``text`` fields.
    """
    if not transcript.utterances:
        return transcript

    # Resolve terminology argument.
    if terminology is None:
        term_obj: Terminology = Terminology.empty()
    elif isinstance(terminology, (str, Path)):
        term_obj = Terminology.load(terminology)
    else:
        term_obj = terminology

    chunks = chunk_transcript(
        transcript,
        max_utterances=chunk_size,
        max_gap_seconds=max_gap_seconds,
        context_window=context_window,
    )

    if concurrency > 1 and len(chunks) > 1:
        cleaned_all, all_records = _clean_chunks_parallel(
            client, chunks, term_obj, domain, enable_review, concurrency,
            progress_callback,
        )
    else:
        cleaned_all, all_records = _clean_chunks_sequential(
            client, chunks, term_obj, domain, enable_review, progress_callback
        )

    composed = Transcript(utterances=cleaned_all, duration=transcript.duration)

    if enable_final_validation:
        composed = _final_validate(client, composed, all_records, domain)

    return composed


def _correct_and_review_chunk(
    client: DeepSeekClient,
    chunk,
    term_obj: Terminology,
    domain: str,
    enable_review: bool,
    previous_records: list[_CorrectionRecord],
) -> tuple[list[Utterance], list[_CorrectionRecord]]:
    """Run the correct (+ optional review) stages for one chunk.

    ``previous_records`` is the correction history visible to this chunk's
    correct-stage prompt (in sequential mode: everything so far; in
    parallel mode: a snapshot taken when the chunk started).
    """
    corrected, records = _correct_chunk(
        client, chunk, term_obj, previous_records, domain
    )
    all_records = list(previous_records) + records

    if enable_review:
        # Pass only records from *before* this chunk as "previous" context;
        # the review prompt should judge the current chunk afresh.
        prior_records = [r for r in all_records if r["index"] < chunk.start_index]
        corrected, records = _review_chunk(
            client,
            chunk,
            chunk.utterances,
            corrected,
            term_obj,
            prior_records,
            domain,
        )
        # Replace correction records for this chunk with the reviewed set.
        all_records = [
            r
            for r in all_records
            if r["index"] < chunk.start_index or r["index"] >= chunk.end_index
        ]
        all_records.extend(records)

    return corrected, all_records


def _clean_chunks_sequential(
    client: DeepSeekClient,
    chunks,
    term_obj: Terminology,
    domain: str,
    enable_review: bool,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[Utterance], list[_CorrectionRecord]]:
    """Original strictly-sequential chunk processing."""
    all_records: list[_CorrectionRecord] = []
    cleaned_all: list[Utterance] = []

    for i, chunk in enumerate(chunks):
        corrected, all_records = _correct_and_review_chunk(
            client, chunk, term_obj, domain, enable_review, all_records
        )
        cleaned_all.extend(corrected)
        if progress_callback is not None:
            progress_callback(i + 1, len(chunks))

    return cleaned_all, all_records


def _clean_chunks_parallel(
    client: DeepSeekClient,
    chunks,
    term_obj: Terminology,
    domain: str,
    enable_review: bool,
    concurrency: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[Utterance], list[_CorrectionRecord]]:
    """Process chunks with a bounded thread pool.

    Each chunk's correct-stage prompt receives a snapshot of the correction
    records completed when the chunk *starts* (chunks running concurrently
    don't see each other's corrections). Chunk results are re-assembled in
    original order. The final global validation stage still runs afterwards
    and catches any cross-chunk inconsistency.
    """
    lock = threading.Lock()
    done_records: list[_CorrectionRecord] = []
    results: dict[int, list[Utterance]] = {}

    def process(index: int, chunk) -> tuple[int, list[Utterance], list[_CorrectionRecord]]:
        with lock:
            snapshot = list(done_records)
        corrected, produced = _correct_and_review_chunk(
            client, chunk, term_obj, domain, enable_review, snapshot
        )
        # ``produced`` = snapshot + this chunk's records; only the new tail
        # (records belonging to this chunk) needs to be merged back.
        new_records = [r for r in produced if chunk.start_index <= r["index"] < chunk.end_index]
        with lock:
            done_records.extend(new_records)
        return index, corrected, new_records

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(process, i, c) for i, c in enumerate(chunks)]
        for done_count, fut in enumerate(as_completed(futures), start=1):
            index, corrected, _ = fut.result()
            results[index] = corrected
            if progress_callback is not None:
                progress_callback(done_count, len(chunks))

    cleaned_all: list[Utterance] = []
    for i in sorted(results):
        cleaned_all.extend(results[i])

    return cleaned_all, sorted(done_records, key=lambda r: r["index"])


__all__ = ["clean_transcript", "CLEAN_TRANSCRIPT_MODEL"]
