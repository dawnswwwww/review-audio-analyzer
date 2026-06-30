# Audio Interview Analyzer — Design

**Date:** 2026-06-30
**Status:** Approved (pending user review of this written spec)
**Repo:** `audio-interview-analyze` (Python 3.12, managed with `uv`)

---

## 1. Overview

A single-purpose CLI tool that takes an interview audio or video file, transcribes it with speaker labels, analyzes each interviewer-question / candidate-answer pair with an LLM, and produces a Chinese Markdown report containing: overall evaluation, highlights, weaknesses, per-question analysis, and a deduplicated knowledge-points summary. Designed to help a candidate learn from their own interview recordings.

The product is a CLI, not a service or web app. The user is a single person running it on their own machine; there is no multi-user, no scheduling, no batch mode.

## 2. Goals

1. Transcribe Chinese-dominant technical interviews (with mixed English technical terms) with interviewer / candidate labels.
2. Evaluate the candidate's answers in the context of the interviewer's questions — not as free-floating text.
3. Produce a structured, readable Markdown report with the four required sections.
4. Be re-runnable: re-running the analysis on the same audio should be cheap (cache transcription, retry the LLM step).
5. Have clear, actionable errors — first-run failures should explain how to fix them, not just dump a stack trace.

## 3. Non-goals (YAGNI for v1)

- Web UI / Gradio / Streamlit.
- Three-or-more-speaker diarization (only the binary interviewer / candidate split).
- PDF or HTML report formats (Markdown only).
- Multi-file / batch mode.
- Spaced-repetition flashcard generation from the extracted knowledge points.
- Implementing multiple LLM providers in code (the interface exists, only DeepSeek ships).
- Streaming token output for LLM calls (batch analysis, not interactive chat).
- Speaker identification across files (e.g., "this is the same candidate as last time").

## 4. Key decisions (the constraints the design was built around)

| Decision | Choice | Why |
|---|---|---|
| Interview type | Technical (algorithm / system design) | Defines the evaluation framework and knowledge categories. |
| Language | Chinese-primary, English technical terms mixed | Standard scenario for Chinese tech interviews. |
| Transcription | **Local** — `faster-whisper` (large-v3) | No per-interview cost, no audio upload. |
| Diarization | **Local** — `pyannote-audio` | Same reason. |
| Analysis LLM | **Cloud** — DeepSeek (`deepseek-chat`) | Cheap, strong at Chinese technical content. |
| Delivery | **Pure CLI** → Markdown report + console progress | No UI surface to maintain. |
| Analysis unit | **Q+A pair** (interviewer question + candidate answer as a unit) | Natural map-reduce unit; lets the LLM reason over question+answer together. |
| Length strategy | **Map-reduce** (30–90 min range) | Single-pass prompt won't fit at the upper end; per-pair fits comfortably. |
| LLM stage count | **Two** — per-pair analyze, then aggregate | Re-runnable aggregator; clean separation of concerns. |

## 5. Architecture — the 7-stage pipeline

```
extract → diarize → transcribe → build_qa_pairs → analyze_pair (LLM #1) → aggregate (LLM #2) → render_md
```

1. **`extract_audio`** — `ffmpeg` via `pydub`. If input is video, extract 16 kHz mono WAV. If input is already audio, convert to the same WAV format. Skipped on cache hit.
2. **`diarize`** — `pyannote-audio` returns speaker segments `[start, end, speaker_label]`.
3. **`transcribe`** — `faster-whisper` (default `large-v3`) with VAD; word timestamps are intersected with pyannote segments to emit ordered `Utterance(speaker, text, start, end)`.
4. **`build_qa_pairs`** — Pure function: `Transcript → list[QAPair]` using the heuristic in §8.
5. **`analyze_pair` (LLM #1)** — One DeepSeek call per `QAPair`; outputs `PairAnalysis` JSON.
6. **`aggregate` (LLM #2)** — One DeepSeek call over all `PairAnalysis` objects; outputs `FinalReport`. If the input would exceed ~50K input tokens, an intermediate summarization call is made first.
7. **`render_markdown`** — Pure function: `FinalReport → Markdown string`. Renders in order: ① 面试总评 ② 亮点 ③ 不足 ④ 逐题分析 ⑤ 知识点总结 ⑥ 完整转写 (附录).

All intermediate artifacts are written to `./.interview-cache/<content-hash>/` keyed by the SHA-256 of the input file. The aggregator and renderer can be re-run cheaply with `--reuse-cache`.

## 6. Project structure

```
audio-interview-analyze/
├── pyproject.toml
├── README.md
├── .python-version
├── .gitignore                  # add: .interview-cache/, .env, *.wav
├── src/
│   └── audio_interview_analyze/
│       ├── __init__.py
│       ├── cli.py              # argparse, Rich progress, entry point
│       ├── pipeline.py         # orchestrates the 7 stages
│       ├── cache.py            # artifact caching by content hash
│       ├── audio/
│       │   ├── extract.py
│       │   ├── diarize.py
│       │   └── transcribe.py
│       ├── analysis/
│       │   ├── qa_pairs.py
│       │   ├── prompts.py
│       │   ├── analyze.py
│       │   └── aggregate.py
│       ├── llm/
│       │   └── deepseek.py
│       └── report/
│           ├── model.py
│           └── render.py
└── tests/
    ├── fixtures/
    ├── test_qa_pairs.py
    ├── test_render.py
    ├── test_prompts.py
    ├── test_cache.py
    ├── test_analyze.py
    ├── test_analyze_retry.py
    ├── test_aggregate.py
    └── test_pipeline_smoke.py
```

## 7. Dependencies

**Runtime:** `faster-whisper`, `pyannote-audio`, `pydub`, `httpx`, `pydantic`, `rich`, `jinja2`, `python-dotenv`.

**Dev:** `pytest`, `pytest-cov`, `ruff`, `respx`.

**External prerequisites (documented in README, not pip-installable):**

- `ffmpeg` on PATH.
- HuggingFace account + token accepting `pyannote-audio` model terms.
- `DEEPSEEK_API_KEY` in `.env` (gitignored).

## 8. Data model (Pydantic v2)

```python
class Utterance(BaseModel):
    speaker: str            # "interviewer" | "candidate" | "unknown"
    text: str
    start: float
    end: float

class Transcript(BaseModel):
    utterances: list[Utterance]
    duration: float

class QAPair(BaseModel):
    question: str
    answer: str
    q_start: float
    q_end: float
    a_start: float
    a_end: float
    pair_index: int

class PairAnalysis(BaseModel):
    pair_index: int
    question_summary: str
    answer_evaluation: str
    knowledge_points: list[str]
    highlights: list[str]
    weaknesses: list[str]
    improvement_suggestions: str

class KnowledgePoint(BaseModel):
    name: str
    category: str            # 数据结构 / 算法 / 系统设计 / 编程语言 / 计算机网络 / 操作系统 / 数据库 / 其它
    explanation: str
    learning_directions: list[str]

class FinalReport(BaseModel):
    overall_evaluation: str
    overall_score: int
    dimension_scores: dict[str, int]   # 技术深度 / 思路结构 / 表达清晰度 / 反应速度
    consolidated_highlights: list[str]
    consolidated_weaknesses: list[str]
    consolidated_knowledge_points: list[KnowledgePoint]
    pair_analyses: list[PairAnalysis]
    full_transcript: Transcript
```

## 9. Q+A pair extraction heuristic

1. **Speaker labeling** — The speaker with more total speech time is `candidate`; the other is `interviewer`. This is robust for technical rounds where the candidate does most of the talking.
2. **Pairing rule** — Walk utterances in order:
   - A new pair starts at each `interviewer` utterance (the previous pair is closed).
   - The `interviewer` text becomes the `question`; the question window runs until the first `candidate` utterance.
   - Subsequent `candidate` utterances are appended to the `answer` until one of: another `interviewer` utterance, a silence gap > 8 s, or a `candidate` question that is treated as a clarification (kept in the running answer).
3. **Edge cases**:
   - `candidate` speaks first (greeting / self-intro) → emit a pair with empty `question`. Skipped by the analyzer.
   - Multiple `interviewer` turns before a `candidate` answer (clarifying questions) → merge into the same `question`.
   - `candidate` answer < 20 characters after stripping → pair is logged and skipped.
4. **Why heuristic, not LLM?** An LLM splitter would mean two LLM calls per pair (split, then analyze). The heuristic is deterministic, fast, debuggable, and the natural unit of "question then answer" is recoverable from speaker alternation and silence. If real-data tests show it's brittle, swap in an LLM splitter behind the same interface.

## 10. LLM prompts

Two prompts, both in Chinese, both with strict JSON output (`response_format={"type": "json_object"}` + Pydantic validation client-side).

**Prompt 1 — `analyze_pair` (per Q+A pair):**
- System: senior technical interviewer reviewing a candidate's answer; strict on objectivity and JSON output.
- User: question + answer, with fields `question_summary`, `answer_evaluation`, `knowledge_points`, `highlights`, `weaknesses`, `improvement_suggestions`.
- Call: `temperature=0.3, max_tokens=2000`.

**Prompt 2 — `aggregate` (over all pair analyses):**
- System: senior technical interviewer producing a total evaluation; must score four dimensions (技术深度、思路结构、表达清晰度、反应速度), merge knowledge points by name (deduplicating "红黑树" vs "RB-Tree"), categorize knowledge points, give explanations and learning directions.
- User: candidate background (optional) + all per-pair analyses.
- Call: `temperature=0.4, max_tokens=6000`.

**Failure handling:** JSON parse / Pydantic validation failure → retry once with a "请确保输出是合法 JSON" suffix. Second failure → dump raw LLM response to the cache directory and raise with a clear message.

**Token-overflow summarizer (Prompt 2.5, hidden from user):** When the total size of the `pair_analyses` JSON would exceed ~50K input tokens, the aggregator first calls a hidden summarization prompt that compresses each `PairAnalysis` to its `question_summary` + one-line `answer_evaluation` + knowledge-point names. The aggregator then runs on the compressed list. This is the only complexity added by the 30–90 min range; it is invisible in the final report. If interviews ever exceed ~3 hours, this summarization would need to become recursive (hierarchical map-reduce) — out of scope for v1.

## 11. Error handling

| Failure | Detection | Behavior |
|---|---|---|
| Input file missing | `Path.exists()` + `ffprobe` | One-line error pointing at the path. |
| `ffmpeg` not on PATH | `shutil.which` at startup | Error: install command for the user's OS. |
| Audio < 5s | Post-extract duration check | Error: audio too short. |
| pyannote model not downloaded | First-run `Pipeline.from_pretrained` raises | Catch, print setup instructions, exit 1. |
| Whisper download fails | `faster_whisper` raises | Catch, suggest HF login / retry. |
| `DEEPSEEK_API_KEY` missing | Env var check at startup | Error pointing to README. |
| DeepSeek rate limit / 5xx | `httpx.HTTPStatusError` | Exponential backoff (3 tries, 2s/4s/8s). |
| LLM invalid JSON | Pydantic `ValidationError` | Retry once, then dump raw to cache, raise. |
| 0 Q+A pairs detected | Post `build_qa_pairs` | Warn, don't crash. |
| Disk full on cache write | `OSError` | Warn, continue without cache. |

Errors follow a **"one sentence cause + one sentence fix"** policy. Stack traces are reserved for genuinely unexpected exceptions.

## 12. Testing strategy

**Layer 1 — Unit (fast, no network, no models):**

- `test_render.py` — Build `FinalReport` by hand, assert Markdown contains expected sections / scores / knowledge names. Locks the report format.
- `test_qa_pairs.py` — Cover greeting, single Q+A, multi-turn, candidate-only, silence gap, two clarifications.
- `test_prompts.py` — Render Jinja templates, assert no `{{ ... }}` left over, JSON schema instructions present.
- `test_cache.py` — Hash determinism, round-trip, key includes input content hash.

**Layer 2 — LLM (mocked HTTP via `respx`):**

- `test_analyze.py` — Canned JSON response; assert `PairAnalysis` is built and the prompt contains the Q+A text.
- `test_analyze_retry.py` — First response invalid, second valid; assert retry + final result.
- `test_aggregate.py` — Canned aggregator response; assert `FinalReport`. Second test: 50K-token overflow path triggers the intermediate summarizer.

**Layer 3 — Smoke (real models, gated by `RUN_SLOW=1`):**

- `test_pipeline_smoke.py` — Full pipeline on a 5-second silent WAV. No LLM call (zero pairs). Asserts: `report.md` is written with the expected section headers, no crashes.

**Coverage target:** ≥ 80% on `report/`, `analysis/qa_pairs.py`, `llm/`, `cache.py`. The `audio/diarize.py` and `audio/transcribe.py` modules are excluded from coverage — they're thin wrappers and exercised by the smoke test.

## 13. First-run UX

The first run is the most fragile moment — three sequential downloads plus a missing API key. Mitigations:

- **Pre-flight check** at CLI start: `ffmpeg` on PATH, `DEEPSEEK_API_KEY` set, HF token set. The preflight runs on every invocation and prints a checklist with pass/fail before any download starts. A `--preflight` flag runs the preflight only (no actual processing) so users can verify their environment. Failures show a link to the README section for that dep.
- **Whisper model size is configurable** — default `large-v3`, but `--model small` is a one-line override for first-time users who want a 500 MB download first.
- **Per-stage progress** via `rich`: "Downloading pyannote model", "Transcribing (12:34 / 30:00)", "Analyzing pair 3 / 12", "Aggregating". Each shows ETA.
- **Cache visibility** — print the cache directory at the end of the run.

## 14. CLI shape

```bash
audio-interview-analyze path/to/interview.mp4
audio-interview-analyze path/to/interview.mp4 --out ./reports/ --model large-v3
audio-interview-analyze path/to/interview.mp4 --reuse-cache   # skip transcription, rerun analysis
audio-interview-analyze path/to/interview.mp4 --preflight     # check deps, no run
```

## 15. Risks / open questions

1. **pyannote-audio model gating** — Requires accepting model terms on HuggingFace. If the user has never used HF, this is a real friction point. README must show the exact UI steps.
2. **"Candidate talks more" heuristic** — Fails for HR-style interviews where the interviewer talks more. We have already constrained to technical interviews, so this is acceptable; if it bites, expose a `--role-of-speaker-1` flag.
3. **Chinese technical-term ASR accuracy** — Whisper `large-v3` is good but occasionally hallucinates or mis-transcribes niche terms (e.g., "epoll" → "e-pull"). If this matters in practice, a post-transcription dictionary pass can be added.
4. **Token overflow on long interviews** — The aggregator summarization is the only complexity added by the 30–90 min range. If a single 3-hour interview is ever attempted, the summarization needs to be hierarchical (recursive), not single-pass.
5. **DeepSeek rate limits** — Default is reasonable but tight at scale. Not an issue for single-user batch use; documented as a known constraint.
