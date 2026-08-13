"""Generate a per-knowledge-point study guide via the LLM.

Takes a ``FinalReport`` (already aggregated) and asks the LLM to produce
a JSON "study card" for each knowledge point: one-liner core, why
interviewers ask, core concepts, code example, common mistakes,
self-check Q&A, and further reading.

The LLM output is plain JSON (not Markdown) so the renderer can produce
consistent, validated output. Output is a plain dict (not a Pydantic
model) because the LLM response shape is large and loosely-typed; we
trust the prompt's constraints rather than enforce them via a model.

When the report has many knowledge points (>10), the call is batched
so the LLM doesn't truncate the response. The first batch produces
``guide_title`` + ``preface``; subsequent batches only produce
``knowledge_points``.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from audio_interview_analyze.analysis.prompts import (
    STUDY_GUIDE_SYSTEM,
    study_guide_prompt,
)
from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.report.model import FinalReport

STUDY_GUIDE_MODEL = "deepseek-v4-flash"
STUDY_GUIDE_TEMPERATURE = 0.4
STUDY_GUIDE_MAX_TOKENS = 8000
STUDY_GUIDE_BATCH_THRESHOLD = 10
STUDY_GUIDE_BATCH_SIZE = 7
RETRY_SUFFIX = "\n请确保输出是合法 JSON。"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _chat_json(client: DeepSeekClient, system: str, user: str) -> dict:
    """Call the LLM with response_format=json_object, retrying once on parse failure."""
    last_raw: str | None = None
    last_error: Exception | None = None
    for attempt in range(2):
        suffix = RETRY_SUFFIX if attempt == 1 else ""
        raw = client.chat(
            system=system,
            user=user + suffix,
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
        f"LLM call failed after 2 attempts: {last_error}. "
        f"Last raw response: {(last_raw or '')[:500]}"
    )


def generate_study_guide(
    client: DeepSeekClient, report: FinalReport, *, concurrency: int = 1
) -> dict:
    """Call the LLM to produce a study guide for all knowledge points.

    Returns a dict with keys ``guide_title``, ``preface``, and
    ``knowledge_points`` (list of per-KP dicts). If the report has no
    knowledge points, returns a minimal structure without calling the LLM.

    ``concurrency`` controls how many batch calls run in parallel. Batch
    prompts are independent (only the first batch additionally produces
    ``guide_title``/``preface``), so parallel execution yields the same
    content as sequential.
    """
    kps = report.consolidated_knowledge_points
    if not kps:
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
        for kp in kps
    ]

    # Single batch when there are few knowledge points.
    if len(kp_payload) <= STUDY_GUIDE_BATCH_THRESHOLD:
        user_msg = study_guide_prompt(
            knowledge_points_json=json.dumps(kp_payload, ensure_ascii=False, indent=2)
        )
        return _chat_json(client, STUDY_GUIDE_SYSTEM, user_msg)

    # Batched generation: split into chunks of STUDY_GUIDE_BATCH_SIZE.
    # The first batch produces guide_title/preface; subsequent batches
    # only emit knowledge_points which get appended to the result.
    batches = [
        kp_payload[start : start + STUDY_GUIDE_BATCH_SIZE]
        for start in range(0, len(kp_payload), STUDY_GUIDE_BATCH_SIZE)
    ]

    def run_batch(batch: list[dict], *, is_first: bool) -> dict:
        user_msg = study_guide_prompt(
            knowledge_points_json=json.dumps(batch, ensure_ascii=False, indent=2),
            is_first_batch=is_first,
        )
        return _chat_json(client, STUDY_GUIDE_SYSTEM, user_msg)

    guide_title = "面试学习指南"
    preface = ""
    all_points: list[dict] = []

    if concurrency > 1 and len(batches) > 1:
        batch_results: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=min(concurrency, len(batches))) as pool:
            futures = {
                pool.submit(run_batch, batch, is_first=(i == 0)): i
                for i, batch in enumerate(batches)
            }
            for fut in as_completed(futures):
                batch_results[futures[fut]] = fut.result()
        first = batch_results[0]
        guide_title = first.get("guide_title") or guide_title
        preface = first.get("preface") or preface
        for i in sorted(batch_results):
            all_points.extend(batch_results[i].get("knowledge_points") or [])
    else:
        for i, batch in enumerate(batches):
            result = run_batch(batch, is_first=(i == 0))
            if i == 0:
                guide_title = result.get("guide_title") or guide_title
                preface = result.get("preface") or preface
            all_points.extend(result.get("knowledge_points") or [])

    return {
        "guide_title": guide_title,
        "preface": preface,
        "knowledge_points": all_points,
    }


def _parse_response(raw: str) -> dict:
    """Parse the LLM's JSON response, tolerating code-fence wrapping."""
    data = json.loads(_strip_code_fence(raw))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")
    if "knowledge_points" not in data or not isinstance(data["knowledge_points"], list):
        raise ValueError("Response missing 'knowledge_points' list")
    return data
