"""Aggregator (LLM #2) and the token-overflow summarizer (Prompt 2.5).

The aggregator consumes all per-pair analyses and produces a ``FinalReport``.
When the serialized per-pair list is large enough that sending it directly
would exceed ``AGGREGATE_MAX_INPUT_TOKENS`` input tokens, a hidden
summarization step compresses each pair to its essentials first.

Token estimation is naive (4 chars ≈ 1 token) — good enough for the
"small vs huge" decision we need to make.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from audio_interview_analyze.analysis.prompts import (
    AGGREGATE_SYSTEM,
    SUMMARIZE_PAIRS_SYSTEM,
    aggregate_prompt,
    summarize_pairs_prompt,
)
from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.report.model import FinalReport, KnowledgePoint, PairAnalysis

AGGREGATE_MODEL = "deepseek-chat"
AGGREGATE_TEMPERATURE = 0.4
AGGREGATE_MAX_TOKENS = 6000
AGGREGATE_MAX_INPUT_TOKENS = 50_000
SUMMARIZE_TEMPERATURE = 0.2
SUMMARIZE_MAX_TOKENS = 4000
RETRY_SUFFIX = "\n请确保输出是合法 JSON。"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def maybe_summarize(
    client: DeepSeekClient, pairs: list[PairAnalysis]
) -> list[dict[str, Any]] | list[PairAnalysis]:
    """If the serialized pair list would exceed the input token limit, run the
    summarizer and return a list of compressed dicts. Otherwise return the
    original ``PairAnalysis`` list unchanged.
    """
    serialized = json.dumps([p.model_dump() for p in pairs], ensure_ascii=False)
    if _approx_tokens(serialized) <= AGGREGATE_MAX_INPUT_TOKENS:
        return pairs
    user_msg = summarize_pairs_prompt(
        pair_analyses=[p.model_dump() for p in pairs]
    )
    raw = client.chat(
        system=SUMMARIZE_PAIRS_SYSTEM,
        user=user_msg,
        model=AGGREGATE_MODEL,
        temperature=SUMMARIZE_TEMPERATURE,
        max_tokens=SUMMARIZE_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    cleaned = _strip_code_fence(raw)
    data = json.loads(cleaned)
    # The summarizer returns an array; tolerate the model wrapping it in an
    # object under a "results" or "pairs" key.
    if isinstance(data, dict):
        for key in ("results", "pairs", "summaries", "items"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Summarizer returned non-list payload: {type(data).__name__}")
    return data


def aggregate(
    client: DeepSeekClient,
    pair_analyses: list[PairAnalysis],
    candidate_background: str,
) -> FinalReport:
    """Run the aggregator and return a validated ``FinalReport``."""
    payload = maybe_summarize(client, pair_analyses)
    user_msg = aggregate_prompt(
        pair_analyses=[p.model_dump() if isinstance(p, PairAnalysis) else p for p in payload],
        candidate_background=candidate_background,
    )

    last_raw: str | None = None
    last_error: Exception | None = None
    for attempt in range(2):
        suffix = RETRY_SUFFIX if attempt == 1 else ""
        raw = client.chat(
            system=AGGREGATE_SYSTEM,
            user=user_msg + suffix,
            model=AGGREGATE_MODEL,
            temperature=AGGREGATE_TEMPERATURE,
            max_tokens=AGGREGATE_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        last_raw = raw
        try:
            data = json.loads(_strip_code_fence(raw))
            return _build_final_report(data, pair_analyses)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = e
            continue

    raise ValueError(
        f"Aggregator failed to produce valid JSON after 2 attempts: {last_error}. "
        f"Last raw response: {(last_raw or '')[:500]}"
    )


def _build_final_report(data: dict[str, Any], pair_analyses: list[PairAnalysis]) -> FinalReport:
    """Build a ``FinalReport`` from the aggregator's JSON, attaching the pair
    analyses and an empty transcript (the pipeline fills in the transcript
    later).
    """
    from audio_interview_analyze.report.model import Transcript

    knowledge_points = [
        KnowledgePoint.model_validate(kp) for kp in data.get("consolidated_knowledge_points", [])
    ]
    return FinalReport(
        overall_evaluation=data["overall_evaluation"],
        overall_score=data["overall_score"],
        dimension_scores=data["dimension_scores"],
        consolidated_highlights=data.get("consolidated_highlights", []),
        consolidated_weaknesses=data.get("consolidated_weaknesses", []),
        consolidated_knowledge_points=knowledge_points,
        pair_analyses=pair_analyses,
        full_transcript=Transcript(utterances=[], duration=0.0),
    )
