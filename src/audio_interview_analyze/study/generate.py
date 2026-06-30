"""Generate a per-knowledge-point study guide via the LLM.

Takes a ``FinalReport`` (already aggregated) and asks the LLM to produce
a JSON "study card" for each knowledge point: one-liner core, why
interviewers ask, core concepts, code example, common mistakes,
self-check Q&A, and further reading.

The LLM output is plain JSON (not Markdown) so the renderer can produce
consistent, validated output. Output is a plain dict (not a Pydantic
model) because the LLM response shape is large and loosely-typed; we
trust the prompt's constraints rather than enforce them via a model.
"""

from __future__ import annotations

import json
import re

from audio_interview_analyze.analysis.prompts import (
    STUDY_GUIDE_SYSTEM,
    study_guide_prompt,
)
from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.report.model import FinalReport

STUDY_GUIDE_MODEL = "deepseek-chat"
STUDY_GUIDE_TEMPERATURE = 0.4
STUDY_GUIDE_MAX_TOKENS = 8000
RETRY_SUFFIX = "\n请确保输出是合法 JSON。"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def generate_study_guide(client: DeepSeekClient, report: FinalReport) -> dict:
    """Call the LLM to produce a study guide for all knowledge points.

    Returns a dict with keys ``guide_title``, ``preface``, and
    ``knowledge_points`` (list of per-KP dicts). If the report has no
    knowledge points, returns a minimal structure without calling the LLM.
    """
    if not report.consolidated_knowledge_points:
        return {
            "guide_title": "面试学习指南",
            "preface": "这场面试没有提取到可学习的知识点。",
            "knowledge_points": [],
        }

    kp_payload = [
        {
            "name": kp.name,
            "category": kp.category,
            "explanation": kp.explanation,
            "learning_directions": kp.learning_directions,
        }
        for kp in report.consolidated_knowledge_points
    ]
    user_msg = study_guide_prompt(
        knowledge_points_json=json.dumps(kp_payload, ensure_ascii=False, indent=2)
    )

    last_raw: str | None = None
    last_error: Exception | None = None
    for attempt in range(2):
        suffix = RETRY_SUFFIX if attempt == 1 else ""
        raw = client.chat(
            system=STUDY_GUIDE_SYSTEM,
            user=user_msg + suffix,
            model=STUDY_GUIDE_MODEL,
            temperature=STUDY_GUIDE_TEMPERATURE,
            max_tokens=STUDY_GUIDE_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        last_raw = raw
        try:
            return _parse_response(raw)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            continue

    raise ValueError(
        f"generate_study_guide failed after 2 attempts: {last_error}. "
        f"Last raw response: {(last_raw or '')[:500]}"
    )


def _parse_response(raw: str) -> dict:
    """Parse the LLM's JSON response, tolerating code-fence wrapping."""
    data = json.loads(_strip_code_fence(raw))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")
    if "knowledge_points" not in data or not isinstance(data["knowledge_points"], list):
        raise ValueError("Response missing 'knowledge_points' list")
    return data
