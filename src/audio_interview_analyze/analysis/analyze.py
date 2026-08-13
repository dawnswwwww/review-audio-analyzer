"""Per-pair LLM analysis.

For each QAPair, call DeepSeek with the analyze_pair prompt and parse the
JSON response into a ``PairAnalysis``. On JSON parse / Pydantic validation
failure, retry once with a "请确保输出是合法 JSON" suffix. On second failure,
dump the raw response to the cache directory and raise.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from audio_interview_analyze.analysis.prompts import (
    ANALYZE_PAIR_SYSTEM,
    analyze_pair_prompt,
)
from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.report.model import PairAnalysis, QAPair

ANALYZE_MODEL = "deepseek-v4-flash"
ANALYZE_TEMPERATURE = 0.3
ANALYZE_MAX_TOKENS = 2000
RETRY_SUFFIX = "\n请确保输出是合法 JSON。"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    """Remove leading/trailing Markdown code fences if present."""
    return _FENCE_RE.sub("", text).strip()


def _extract_json(text: str) -> dict:
    """Parse a JSON object from the model output, tolerating code fences."""
    cleaned = _strip_code_fence(text)
    return json.loads(cleaned)


def analyze_pair(
    client: DeepSeekClient,
    pair: QAPair,
    *,
    cache_dir: Path | None = None,
) -> PairAnalysis:
    """Call the LLM once (twice on retry) and return a validated ``PairAnalysis``.

    Args:
        client: Configured DeepSeek client.
        pair: The Q+A pair to analyze.
        cache_dir: If set, write raw responses to ``<cache_dir>/pair_<n>_raw.txt``
            for debugging on the second failure.

    Raises:
        ValueError: If both attempts fail to produce valid JSON / a valid
            ``PairAnalysis``.
    """
    user_msg = analyze_pair_prompt(question=pair.question, answer=pair.answer)
    last_raw: str | None = None
    last_error: Exception | None = None

    for attempt in range(2):
        suffix = RETRY_SUFFIX if attempt == 1 else ""
        raw = client.chat(
            system=ANALYZE_PAIR_SYSTEM,
            user=user_msg + suffix,
            model=ANALYZE_MODEL,
            temperature=ANALYZE_TEMPERATURE,
            max_tokens=ANALYZE_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        last_raw = raw
        try:
            data = _extract_json(raw)
            data["pair_index"] = pair.pair_index
            data["question"] = pair.question
            data["answer"] = pair.answer
            return PairAnalysis.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            continue

    if cache_dir is not None and last_raw is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"pair_{pair.pair_index}_raw.txt").write_text(last_raw, encoding="utf-8")
    raise ValueError(
        f"Failed to get valid JSON for pair {pair.pair_index} after 2 attempts: {last_error}"
    )


def analyze_conversation(pair: QAPair) -> PairAnalysis:
    """Build a minimal PairAnalysis for casual conversation that doesn't
    need LLM evaluation.

    The returned object has ``is_conversation=True`` and an empty
    ``conversation_note``; the renderer will display it as a quoted
    exchange instead of a scored analysis.
    """
    return PairAnalysis(
        pair_index=pair.pair_index,
        question=pair.question,
        answer=pair.answer,
        question_summary="",
        answer_evaluation="",
        knowledge_points=[],
        highlights=[],
        weaknesses=[],
        improvement_suggestions="",
        is_conversation=True,
        conversation_note="",
    )