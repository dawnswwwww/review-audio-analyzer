# Audio Interview Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI that takes an interview audio/video file, transcribes it with speaker labels, analyzes each interviewer-question / candidate-answer pair with DeepSeek, and emits a Chinese Markdown report (overall evaluation, highlights, weaknesses, per-question analysis, knowledge points).

**Architecture:** 7-stage pipeline orchestrated by `pipeline.py` (extract → diarize → transcribe → build Q+A → per-pair LLM analyze → aggregate → render). Pure functions for Q+A extraction, prompts, and rendering; thin wrappers around `faster-whisper`, `pyannote-audio`, and DeepSeek's HTTP API. All intermediate artifacts cached by content hash so the aggregator is re-runnable without re-transcribing.

**Tech Stack:** Python 3.12, `uv` for deps, `faster-whisper` (local transcription), `pyannote-audio` (local diarization), `httpx` (DeepSeek API, OpenAI-compatible), `pydantic` (typed data model), `jinja2` (prompts), `rich` (console progress), `pydub` + `ffmpeg` (audio extraction), `pytest` + `respx` + `ruff` (dev).

**Reference spec:** `docs/superpowers/specs/2026-06-30-audio-interview-analyzer-design.md` — read this first.

---

## File Structure (locked in up front)

```
audio-interview-analyze/
├── pyproject.toml              # uv-managed
├── README.md
├── .python-version
├── .gitignore                  # add .interview-cache/, .env
├── src/
│   └── audio_interview_analyze/
│       ├── __init__.py
│       ├── cli.py              # argparse + rich progress
│       ├── pipeline.py         # 7-stage orchestration
│       ├── cache.py            # content-hash artifact cache
│       ├── preflight.py        # ffmpeg / API key / HF token check
│       ├── audio/
│       │   ├── __init__.py
│       │   ├── extract.py      # video/audio → 16kHz mono WAV
│       │   ├── diarize.py      # pyannote wrapper
│       │   └── transcribe.py   # faster-whisper + alignment
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── qa_pairs.py     # Transcript → QAPair[]
│       │   ├── prompts.py      # Jinja templates
│       │   ├── analyze.py      # per-pair LLM call
│       │   └── aggregate.py    # aggregator + summarizer
│       ├── llm/
│       │   ├── __init__.py
│       │   └── deepseek.py     # httpx + retry/backoff
│       └── report/
│           ├── __init__.py
│           ├── model.py        # Pydantic models
│           └── render.py       # FinalReport → Markdown
└── tests/
    ├── __init__.py
    ├── conftest.py             # shared fixtures
    ├── fixtures/
    │   ├── sample_transcript.json
    │   └── sample_final_report.json
    ├── test_model.py
    ├── test_cache.py
    ├── test_qa_pairs.py
    ├── test_prompts.py
    ├── test_render.py
    ├── test_deepseek.py
    ├── test_analyze.py
    ├── test_aggregate.py
    ├── test_preflight.py
    └── test_pipeline_smoke.py  # gated by RUN_SLOW=1
```

**Decomposition rules followed:**
- One file, one responsibility. `cache.py` is hash + read/write; `pipeline.py` only orchestrates.
- Pure functions live in their own modules so they're testable without mocks.
- Wrappers around external libs (`audio/`, `llm/`) are thin — exception translation only.
- All Pydantic models live in `report/model.py` so the data shape is defined in one place.

---

## Phase 0: Project Setup

### Task 1: Restructure to `src/` layout and create package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/audio_interview_analyze/__init__.py`
- Delete: `main.py` (replaced by package)
- Modify: `.gitignore`

- [ ] **Step 1: Update `pyproject.toml`**

Replace the contents of `pyproject.toml` with:

```toml
[project]
name = "audio-interview-analyze"
version = "0.1.0"
description = "Analyze technical interview recordings: transcribe, evaluate, and summarize knowledge points."
readme = "README.md"
requires-python = ">=3.12"
dependencies = []

[project.scripts]
audio-interview-analyze = "audio_interview_analyze.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/audio_interview_analyze"]
```

- [ ] **Step 2: Create the package directory and `__init__.py`**

```bash
mkdir -p src/audio_interview_analyze
```

Write `src/audio_interview_analyze/__init__.py`:

```python
"""Audio Interview Analyzer - transcribe, evaluate, and summarize technical interviews."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Update `.gitignore`**

Append to `.gitignore`:

```
# Cache and runtime artifacts
.interview-cache/
.env
*.wav
```

- [ ] **Step 4: Delete the old `main.py`**

```bash
rm main.py
```

- [ ] **Step 5: Verify the package builds**

```bash
uv sync
uv run python -c "import audio_interview_analyze; print(audio_interview_analyze.__version__)"
```

Expected: prints `0.1.0`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ .gitignore
git rm main.py
git commit -m "feat: restructure to src/ layout, add package skeleton"
```

---

### Task 2: Add runtime dependencies

**Files:** `pyproject.toml` (modified by uv), `uv.lock` (created)

- [ ] **Step 1: Add runtime dependencies**

```bash
uv add faster-whisper pyannote-audio pydub httpx pydantic rich jinja2 python-dotenv
```

Expected: `uv` adds the packages to `[project.dependencies]` and creates `uv.lock`.

- [ ] **Step 2: Verify imports**

```bash
uv run python -c "import faster_whisper, pyannote, pydub, httpx, pydantic, rich, jinja2, dotenv; print('all imports ok')"
```

Expected: prints `all imports ok`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add runtime dependencies"
```

---

### Task 3: Add dev dependencies and configure ruff/pytest

**Files:** `pyproject.toml`, `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Add dev dependencies**

```bash
uv add --dev pytest pytest-cov ruff respx
```

- [ ] **Step 2: Append dev tooling config to `pyproject.toml`**

Add to the end of `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --strict-markers"
markers = [
    "slow: end-to-end tests that download models or call real APIs",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N"]
ignore = ["E501"]

[tool.coverage.run]
source = ["src/audio_interview_analyze"]
omit = ["src/audio_interview_analyze/audio/diarize.py", "src/audio_interview_analyze/audio/transcribe.py"]
```

- [ ] **Step 3: Create `tests/__init__.py` and `tests/conftest.py`**

```bash
mkdir -p tests/fixtures
touch tests/__init__.py
```

Write `tests/conftest.py`:

```python
"""Shared test fixtures."""

import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Return a temporary cache directory for the test."""
    cache = tmp_path / ".interview-cache"
    cache.mkdir()
    return cache
```

- [ ] **Step 4: Run pytest to verify the setup**

```bash
uv run pytest --collect-only
```

Expected: `no tests ran` (or `0 tests collected`) with exit code 5 — that's fine, just verifying pytest is wired up.

- [ ] **Step 5: Run ruff to verify config**

```bash
uv run ruff check .
```

Expected: `All checks passed!` (no Python files yet, so no checks to run).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/
git commit -m "feat: add dev dependencies, configure ruff and pytest"
```

---

## Phase 1: Pure Core Modules (TDD)

### Task 4: Define the Pydantic data model

**Files:**
- Create: `src/audio_interview_analyze/report/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_model.py`:

```python
"""Tests for the Pydantic data models."""

from audio_interview_analyze.report.model import (
    FinalReport,
    KnowledgePoint,
    PairAnalysis,
    QAPair,
    Transcript,
    Utterance,
)


def test_utterance_round_trip():
    u = Utterance(speaker="interviewer", text="What is a binary tree?", start=1.0, end=3.5)
    assert u.model_dump() == {
        "speaker": "interviewer",
        "text": "What is a binary tree?",
        "start": 1.0,
        "end": 3.5,
    }


def test_transcript_round_trip():
    t = Transcript(
        utterances=[
            Utterance(speaker="interviewer", text="Q1", start=0.0, end=1.0),
            Utterance(speaker="candidate", text="A1", start=1.5, end=3.0),
        ],
        duration=3.0,
    )
    restored = Transcript.model_validate(t.model_dump())
    assert restored == t


def test_qa_pair_requires_positive_pair_index():
    import pydantic

    with pydantic.ValidationError.__class_getitem__ and pydantic.ValidationError:
        pass

    import pytest

    with pytest.raises(Exception):
        QAPair(
            question="Q",
            answer="A",
            q_start=0.0,
            q_end=1.0,
            a_start=1.0,
            a_end=2.0,
            pair_index=0,  # must be >= 1
        )


def test_knowledge_point_fields():
    kp = KnowledgePoint(
        name="红黑树",
        category="数据结构",
        explanation="一种自平衡二叉查找树。",
        learning_directions=["阅读《算法导论》第 13 章", "实现一个简单的红黑树"],
    )
    assert kp.name == "红黑树"
    assert kp.category == "数据结构"
    assert len(kp.learning_directions) == 2


def test_final_report_round_trip():
    report = FinalReport(
        overall_evaluation="整体表现良好。",
        overall_score=80,
        dimension_scores={"技术深度": 75, "思路结构": 85, "表达清晰度": 80, "反应速度": 80},
        consolidated_highlights=["基础扎实"],
        consolidated_weaknesses=["系统设计经验不足"],
        consolidated_knowledge_points=[
            KnowledgePoint(
                name="红黑树",
                category="数据结构",
                explanation="自平衡二叉查找树。",
                learning_directions=["读《算法导论》"],
            )
        ],
        pair_analyses=[],
        full_transcript=Transcript(utterances=[], duration=0.0),
    )
    restored = FinalReport.model_validate(report.model_dump())
    assert restored.overall_score == 80
    assert restored.dimension_scores["技术深度"] == 75


def test_pair_analysis_default_lists():
    pa = PairAnalysis(
        pair_index=1,
        question_summary="测试",
        answer_evaluation="回答正确。",
        knowledge_points=[],
        highlights=[],
        weaknesses=[],
        improvement_suggestions="可以更简洁。",
    )
    assert pa.knowledge_points == []
    assert pa.highlights == []
    assert pa.weaknesses == []
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_model.py -v
```

Expected: `ModuleNotFoundError: No module named 'audio_interview_analyze.report.model'`.

- [ ] **Step 3: Implement the data model**

Write `src/audio_interview_analyze/report/model.py`:

```python
"""Pydantic data models for the analysis pipeline."""

from pydantic import BaseModel, Field


class Utterance(BaseModel):
    """A single transcribed utterance with speaker label and timing."""

    speaker: str  # "interviewer" | "candidate" | "unknown"
    text: str
    start: float
    end: float


class Transcript(BaseModel):
    """Ordered list of utterances covering the full interview."""

    utterances: list[Utterance]
    duration: float


class QAPair(BaseModel):
    """A single interviewer question and the candidate's answer."""

    question: str
    answer: str
    q_start: float
    q_end: float
    a_start: float
    a_end: float
    pair_index: int = Field(ge=1)


class PairAnalysis(BaseModel):
    """LLM-generated analysis of a single QAPair."""

    pair_index: int = Field(ge=1)
    question_summary: str
    answer_evaluation: str
    knowledge_points: list[str]
    highlights: list[str]
    weaknesses: list[str]
    improvement_suggestions: str


class KnowledgePoint(BaseModel):
    """A canonicalized knowledge point with explanation and learning directions."""

    name: str
    category: str
    explanation: str
    learning_directions: list[str]


class FinalReport(BaseModel):
    """The complete report consumed by the Markdown renderer."""

    overall_evaluation: str
    overall_score: int = Field(ge=0, le=100)
    dimension_scores: dict[str, int]
    consolidated_highlights: list[str]
    consolidated_weaknesses: list[str]
    consolidated_knowledge_points: list[KnowledgePoint]
    pair_analyses: list[PairAnalysis]
    full_transcript: Transcript
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/test_model.py -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/audio_interview_analyze/report/model.py tests/test_model.py
git add src/audio_interview_analyze/report/model.py tests/test_model.py
git commit -m "feat: define Pydantic data models for the pipeline"
```

---

### Task 5: Cache module — content-hash based artifact storage

**Files:**
- Create: `src/audio_interview_analyze/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_cache.py`:

```python
"""Tests for the content-hash cache."""

import json
from pathlib import Path

from audio_interview_analyze.cache import (
    artifact_path,
    cache_root,
    content_hash,
    read_json,
    write_json,
)


def test_content_hash_is_deterministic(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello world")
    b.write_bytes(b"hello world")
    assert content_hash(a) == content_hash(b)


def test_content_hash_changes_with_content(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"world")
    assert content_hash(a) != content_hash(b)


def test_cache_root_under_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cache_root() == tmp_path / ".interview-cache"


def test_artifact_path_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = "abc123"
    assert artifact_path("transcript.json", hash_key=h) == tmp_path / ".interview-cache" / h / "transcript.json"


def test_write_then_read_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = {"score": 80, "highlights": ["a", "b"]}
    write_json("transcript.json", data, hash_key="hash1")
    assert read_json("transcript.json", hash_key="hash1") == data


def test_write_creates_parent_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_json("nested/file.json", {"x": 1}, hash_key="hash2")
    assert (tmp_path / ".interview-cache" / "hash2" / "nested" / "file.json").exists()


def test_read_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert read_json("missing.json", hash_key="hash3") is None
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_cache.py -v
```

Expected: `ModuleNotFoundError: No module named 'audio_interview_analyze.cache'`.

- [ ] **Step 3: Implement the cache module**

Write `src/audio_interview_analyze/cache.py`:

```python
"""Content-hash based artifact cache.

All intermediate artifacts (extracted audio, diarization, transcript,
per-pair analyses) are stored under ``./.interview-cache/<content-hash>/``
so the analysis can be re-run cheaply.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_DIRNAME = ".interview-cache"


def content_hash(path: Path | str) -> str:
    """Return a SHA-256 hex digest of the file at ``path``."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_root(base: Path | None = None) -> Path:
    """Return the cache root directory, creating it if necessary."""
    base = base or Path.cwd()
    root = base / CACHE_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def artifact_path(relative: str, *, hash_key: str, base: Path | None = None) -> Path:
    """Return the absolute path for a cached artifact under ``<hash>/<relative>``."""
    return cache_root(base) / hash_key / relative


def write_json(relative: str, data: Any, *, hash_key: str, base: Path | None = None) -> Path:
    """Write ``data`` as JSON to the cache, creating parent dirs as needed.

    Returns the path written.
    """
    path = artifact_path(relative, hash_key=hash_key, base=base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_json(relative: str, *, hash_key: str, base: Path | None = None) -> Any | None:
    """Read a JSON artifact from the cache, or return ``None`` if missing."""
    path = artifact_path(relative, hash_key=hash_key, base=base)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/test_cache.py -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/audio_interview_analyze/cache.py tests/test_cache.py
git add src/audio_interview_analyze/cache.py tests/test_cache.py
git commit -m "feat: add content-hash artifact cache"
```

---

### Task 6: Q+A pair extraction

**Files:**
- Create: `src/audio_interview_analyze/analysis/qa_pairs.py`
- Create: `tests/test_qa_pairs.py`
- Create: `tests/fixtures/sample_transcript.json`

- [ ] **Step 1: Create the sample transcript fixture**

Write `tests/fixtures/sample_transcript.json`:

```json
{
  "duration": 60.0,
  "utterances": [
    {"speaker": "candidate", "text": "你好面试官", "start": 0.0, "end": 1.0},
    {"speaker": "interviewer", "text": "请介绍一下红黑树", "start": 2.0, "end": 5.0},
    {"speaker": "interviewer", "text": "和 AVL 树有什么区别", "start": 5.5, "end": 7.0},
    {"speaker": "candidate", "text": "红黑树是近似平衡，AVL 是严格平衡", "start": 8.0, "end": 12.0},
    {"speaker": "interviewer", "text": "进程和线程的区别呢", "start": 18.0, "end": 20.0},
    {"speaker": "candidate", "text": "进程是资源分配的最小单位，线程是 CPU 调度的最小单位", "start": 21.0, "end": 25.0}
  ]
}
```

The two interviewer turns (red-black tree → AVL comparison) come **before** any candidate answer, so they should merge into one question per spec section 9.

- [ ] **Step 2: Write the failing tests**

Write `tests/test_qa_pairs.py`:

```python
"""Tests for Q+A pair extraction."""

import json
from pathlib import Path

import pytest

from audio_interview_analyze.analysis.qa_pairs import build_qa_pairs, label_speakers
from audio_interview_analyze.report.model import Transcript, Utterance


@pytest.fixture
def sample_transcript() -> Transcript:
    fixture = Path(__file__).parent / "fixtures" / "sample_transcript.json"
    return Transcript.model_validate(json.loads(fixture.read_text(encoding="utf-8")))


def test_label_speakers_candidate_talks_more(sample_transcript):
    labeled = label_speakers(sample_transcript)
    speakers = {u.speaker for u in labeled.utterances}
    assert speakers == {"interviewer", "candidate"}


def test_build_qa_pairs_emits_pairs_for_each_question(sample_transcript):
    """Sample has: 1 candidate-only greeting, then 2 merged Q+A pairs
    (red-black + AVL merged into one question, then process vs thread)."""
    labeled = label_speakers(sample_transcript)
    pairs = build_qa_pairs(labeled)
    qa_pairs = [p for p in pairs if p.question]
    assert len(qa_pairs) == 2


def test_build_qa_pairs_merges_consecutive_interviewer_turns(sample_transcript):
    """The two interviewer turns about red-black tree and AVL come before
    any candidate answer, so they should merge into one question per spec
    section 9."""
    labeled = label_speakers(sample_transcript)
    pairs = build_qa_pairs(labeled)
    qa_pairs = [p for p in pairs if p.question]
    assert "红黑树" in qa_pairs[0].question
    assert "AVL" in qa_pairs[0].question
    assert "红黑树是近似平衡" in qa_pairs[0].answer
    assert "AVL 是严格平衡" in qa_pairs[0].answer


def test_build_qa_pairs_pair_index_starts_at_one(sample_transcript):
    labeled = label_speakers(sample_transcript)
    pairs = build_qa_pairs(labeled)
    assert all(p.pair_index >= 1 for p in pairs)
    assert pairs[0].pair_index == 1


def test_build_qa_pairs_skips_empty_question_pairs(sample_transcript):
    """The candidate's greeting has no question, so the pair has empty
    question text and is still emitted (the analyzer is responsible for
    skipping it)."""
    labeled = label_speakers(sample_transcript)
    pairs = build_qa_pairs(labeled)
    empty_q = [p for p in pairs if not p.question]
    assert len(empty_q) == 1
    assert empty_q[0].answer == "你好面试官"


def test_build_qa_pairs_silence_gap_closes_answer():
    """A 8+ second silence in the middle of a candidate's answer closes
    the answer per spec section 9, so the candidate's next utterance
    starts a new pair with an empty question."""
    t = Transcript(
        utterances=[
            Utterance(speaker="interviewer", text="讲讲 TCP 三次握手", start=0.0, end=1.0),
            Utterance(speaker="candidate", text="客户端发送 SYN", start=2.0, end=3.0),
            Utterance(speaker="candidate", text="服务端回应 SYN+ACK", start=12.0, end=14.0),  # 9s gap
        ],
        duration=14.0,
    )
    labeled = label_speakers(t)
    pairs = build_qa_pairs(labeled)
    assert len(pairs) == 2
    assert pairs[0].question == "讲讲 TCP 三次握手"
    assert "客户端发送 SYN" in pairs[0].answer
    assert pairs[1].question == ""
    assert "SYN+ACK" in pairs[1].answer


def test_build_qa_pairs_short_gap_keeps_answer_running():
    """A sub-8s gap between candidate utterances should keep the answer
    running, not close it."""
    t = Transcript(
        utterances=[
            Utterance(speaker="interviewer", text="讲讲 TCP", start=0.0, end=1.0),
            Utterance(speaker="candidate", text="第一次握手", start=2.0, end=3.0),
            Utterance(speaker="candidate", text="第二次握手", start=5.0, end=6.0),  # 2s gap
        ],
        duration=6.0,
    )
    labeled = label_speakers(t)
    pairs = build_qa_pairs(labeled)
    assert len(pairs) == 1
    assert "第一次握手" in pairs[0].answer
    assert "第二次握手" in pairs[0].answer


def test_build_qa_pairs_short_answer_kept_not_skipped():
    """The Q+A builder emits all pairs; downstream is responsible for
    skipping short ones. This test documents that contract."""
    t = Transcript(
        utterances=[
            Utterance(speaker="interviewer", text="你叫什么", start=0.0, end=1.0),
            Utterance(speaker="candidate", text="小王", start=2.0, end=3.0),
        ],
        duration=3.0,
    )
    labeled = label_speakers(t)
    pairs = build_qa_pairs(labeled)
    assert len(pairs) == 1
    assert pairs[0].answer == "小王"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_qa_pairs.py -v
```

Expected: `ModuleNotFoundError: No module named 'audio_interview_analyze.analysis.qa_pairs'`.

- [ ] **Step 4: Implement the Q+A extraction**

Write `src/audio_interview_analyze/analysis/__init__.py` (empty):

```python
"""Analysis pipeline: Q+A extraction, LLM prompts, per-pair analysis, aggregation."""
```

Write `src/audio_interview_analyze/analysis/qa_pairs.py`:

```python
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
    """
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
    state = {"in_question": False, "in_answer": False}
    last_end: float | None = None

    def close_pair() -> None:
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
        state["in_question"] = False
        state["in_answer"] = False

    for u in transcript.utterances:
        # Silence gap rule: closes any in-progress pair.
        if last_end is not None and (u.start - last_end) > SILENCE_GAP_SECONDS:
            close_pair()

        if u.speaker == "interviewer":
            if state["in_answer"]:
                close_pair()
            current_question.append(u)
            state["in_question"] = True
        elif u.speaker == "candidate":
            current_answer.append(u)
            state["in_answer"] = True
        # "unknown" speakers are skipped.

        last_end = u.end

    close_pair()
    return pairs
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_qa_pairs.py -v
```

Expected: 8 tests pass.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src/audio_interview_analyze/analysis/ tests/test_qa_pairs.py tests/fixtures/
git add src/audio_interview_analyze/analysis/ tests/test_qa_pairs.py tests/fixtures/sample_transcript.json
git commit -m "feat: implement Q+A pair extraction from labeled transcript"
```

---

### Task 7: Prompt templates

**Files:**
- Create: `src/audio_interview_analyze/analysis/prompts.py`
- Create: `tests/test_prompts.py`
- Create: `tests/fixtures/sample_pair_analysis.json`

- [ ] **Step 1: Create the sample pair analysis fixture**

Write `tests/fixtures/sample_pair_analysis.json`:

```json
[
  {
    "pair_index": 1,
    "question_summary": "红黑树和 AVL 树的区别",
    "answer_evaluation": "回答基本正确但缺少复杂度分析。",
    "knowledge_points": ["红黑树", "AVL 树", "自平衡二叉查找树"],
    "highlights": ["提到了自平衡的概念"],
    "weaknesses": ["没有分析时间复杂度"],
    "improvement_suggestions": "可以补充红黑树的插入 / 删除操作复杂度。"
  },
  {
    "pair_index": 2,
    "question_summary": "进程与线程的区别",
    "answer_evaluation": "回答准确。",
    "knowledge_points": ["进程", "线程", "操作系统"],
    "highlights": [],
    "weaknesses": [],
    "improvement_suggestions": "可以举例说明。"
  }
]
```

- [ ] **Step 2: Write the failing tests**

Write `tests/test_prompts.py`:

```python
"""Tests for prompt templates."""

import json
from pathlib import Path

import pytest

from audio_interview_analyze.analysis.prompts import (
    aggregate_prompt,
    analyze_pair_prompt,
    summarize_pairs_prompt,
)


def test_analyze_pair_prompt_substitutes_question_and_answer():
    user_msg = analyze_pair_prompt(
        question="讲讲红黑树",
        answer="红黑树是一种自平衡的二叉查找树",
    )
    assert "讲讲红黑树" in user_msg
    assert "红黑树是一种自平衡的二叉查找树" in user_msg
    assert "{{" not in user_msg
    assert "}}" not in user_msg


def test_analyze_pair_prompt_includes_json_schema():
    user_msg = analyze_pair_prompt(question="Q", answer="A")
    for field in [
        "question_summary",
        "answer_evaluation",
        "knowledge_points",
        "highlights",
        "weaknesses",
        "improvement_suggestions",
    ]:
        assert field in user_msg, f"Missing field: {field}"


def test_aggregate_prompt_includes_all_pair_summaries():
    pair_analyses = [
        {
            "pair_index": 1,
            "question_summary": "Q1",
            "answer_evaluation": "E1",
            "knowledge_points": ["KP1"],
            "highlights": ["H1"],
            "weaknesses": ["W1"],
            "improvement_suggestions": "S1",
        }
    ]
    user_msg = aggregate_prompt(pair_analyses=pair_analyses, candidate_background="")
    assert "Q1" in user_msg
    assert "E1" in user_msg
    assert "KP1" in user_msg
    assert "{{" not in user_msg
    assert "}}" not in user_msg


def test_aggregate_prompt_handles_empty_background():
    user_msg = aggregate_prompt(pair_analyses=[], candidate_background="")
    assert "未提供" in user_msg


def test_aggregate_prompt_includes_scoring_dimensions():
    user_msg = aggregate_prompt(pair_analyses=[], candidate_background="")
    for dim in ["技术深度", "思路结构", "表达清晰度", "反应速度"]:
        assert dim in user_msg, f"Missing dimension: {dim}"


def test_aggregate_prompt_includes_output_fields():
    user_msg = aggregate_prompt(pair_analyses=[], candidate_background="")
    for field in [
        "overall_evaluation",
        "overall_score",
        "dimension_scores",
        "consolidated_highlights",
        "consolidated_weaknesses",
        "consolidated_knowledge_points",
    ]:
        assert field in user_msg, f"Missing field: {field}"


def test_summarize_pairs_prompt_compresses_pair_analyses():
    pair_analyses = [
        {
            "pair_index": 1,
            "question_summary": "红黑树与 AVL 树的区别",
            "answer_evaluation": "回答基本正确但缺少复杂度分析。",
            "knowledge_points": ["红黑树", "AVL 树"],
            "highlights": ["提到了自平衡"],
            "weaknesses": ["没有分析时间复杂度"],
            "improvement_suggestions": "补充复杂度分析。",
        }
    ]
    user_msg = summarize_pairs_prompt(pair_analyses=pair_analyses)
    assert "红黑树与 AVL 树的区别" in user_msg
    assert "红黑树" in user_msg
    assert "{{" not in user_msg


def test_summarize_pairs_prompt_includes_compression_schema():
    user_msg = summarize_pairs_prompt(pair_analyses=[])
    for field in ["pair_index", "question_summary", "answer_evaluation", "knowledge_points"]:
        assert field in user_msg, f"Missing field: {field}"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_prompts.py -v
```

Expected: `ModuleNotFoundError: No module named 'audio_interview_analyze.analysis.prompts'`.

- [ ] **Step 4: Implement the prompt module**

Write `src/audio_interview_analyze/analysis/prompts.py`:

```python
"""Jinja2 prompt templates for the LLM calls.

The system prompt is constant across calls; the user prompt is rendered from
the per-call data. All prompts are in Chinese and require JSON output.
"""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, StrictUndefined

_ENV = Environment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)

ANALYZE_PAIR_SYSTEM = (
    "你是一位资深的技术面试官，正在评审一位候选人的面试表现。"
    "你要逐题分析一道「面试官提问 → 候选人回答」的片段。\n\n"
    "严格要求：\n"
    "1. 严格基于提供的文本做出判断；不要凭空补充未提及的内容。\n"
    "2. 评价要具体、客观、可操作；避免空泛的「不错」「一般」等评价。\n"
    "3. 知识点的抽取要忠于原文中实际出现或明确暗示的概念；避免编造。\n"
    "4. 严格按 JSON 格式输出，字段名和类型与给出的 schema 一致。"
)

ANALYZE_PAIR_USER_TEMPLATE = """## 面试官提问
{{ question }}

## 候选人回答
{{ answer }}

## 任务
请以 JSON 格式输出以下字段：
- `question_summary`（字符串）：用一句话概括这道题在考察什么知识点。
- `answer_evaluation`（字符串）：2-4 句评价候选人的回答（正确性 / 思路 / 完整性）。
- `knowledge_points`（字符串数组）：这道题涉及的知识点（候选词 / 术语）。
- `highlights`（字符串数组）：候选人回答中的亮点（可为空数组）。
- `weaknesses`（字符串数组）：候选人回答中的不足或错误（可为空数组）。
- `improvement_suggestions`（字符串）：如何更好地回答这道题（1-2 句）。

仅输出 JSON，不要任何额外文本。
"""


def analyze_pair_prompt(*, question: str, answer: str) -> str:
    """Render the user prompt for analyzing a single Q+A pair."""
    tmpl = _ENV.from_string(ANALYZE_PAIR_USER_TEMPLATE)
    return tmpl.render(question=question, answer=answer)


AGGREGATE_SYSTEM = (
    "你是一位资深的技术面试官，要根据多道「逐题分析」的结果，"
    "给出整场面试的总评、亮点、不足，并整理一份去重、带解释的「知识点总结」。\n\n"
    "严格要求：\n"
    "1. 总评分要分维度（技术深度、思路结构、表达清晰度、反应速度），每项 0-100。\n"
    "2. 整体得分是上述维度的加权平均（你决定权重，但要合理）。\n"
    "3. 知识点要合并同义项（如「红黑树」和「RB-Tree」是同一个），按类别分组，"
    "给出 2-4 句的中文解释和 1-3 条拓展学习方向。\n"
    "4. 亮点 / 不足要去重、按主题归类。\n"
    "5. 严格按 JSON 格式输出，字段名和类型与给出的 schema 一致。"
)

AGGREGATE_USER_TEMPLATE = """## 候选人简历背景（可为空）
{{ candidate_background or "未提供" }}

## 整场面试的逐题分析
{% for p in pair_analyses %}
### 第 {{ p.pair_index }} 题
- 题目摘要：{{ p.question_summary }}
- 答案评价：{{ p.answer_evaluation }}
- 涉及知识点：{{ p.knowledge_points | join("、") }}
- 亮点：{{ p.highlights | join("；") }}
- 不足：{{ p.weaknesses | join("；") }}
- 改进建议：{{ p.improvement_suggestions }}
{% endfor %}

## 任务
请以 JSON 格式输出：
- `overall_evaluation`（字符串）：3-5 句整场面试总评。
- `overall_score`（整数 0-100）：整体得分。
- `dimension_scores`（对象）：{"技术深度": int, "思路结构": int, "表达清晰度": int, "反应速度": int}。
- `consolidated_highlights`（字符串数组）：合并去重后的亮点。
- `consolidated_weaknesses`（字符串数组）：合并去重后的不足。
- `consolidated_knowledge_points`（对象数组）：每个对象含
  `name`（规范化名称）, `category`（类别，如 数据结构 / 算法 / 系统设计 / 编程语言 / 计算机网络 / 操作系统 / 数据库 / 其它）,
  `explanation`（2-4 句中文解释）, `learning_directions`（1-3 条拓展学习方向）。

仅输出 JSON，不要任何额外文本。
"""


def aggregate_prompt(*, pair_analyses: list[dict[str, Any]], candidate_background: str) -> str:
    """Render the user prompt for the aggregator."""
    tmpl = _ENV.from_string(AGGREGATE_USER_TEMPLATE)
    return tmpl.render(
        pair_analyses=pair_analyses,
        candidate_background=candidate_background,
    )


SUMMARIZE_PAIRS_SYSTEM = (
    "你是一位助手，负责将多道「逐题分析」压缩成简短形式，"
    "保留每道题的题目摘要、答案评价一行、知识点列表。严格输出 JSON 数组。"
)

SUMMARIZE_PAIRS_USER_TEMPLATE = """## 待压缩的逐题分析
{% for p in pair_analyses %}
### 第 {{ p.pair_index }} 题
- 题目摘要：{{ p.question_summary }}
- 答案评价：{{ p.answer_evaluation }}
- 涉及知识点：{{ p.knowledge_points | join("、") }}
{% endfor %}

## 任务
请以 JSON 数组输出，每个元素是一个对象：
- `pair_index`（整数）
- `question_summary`（字符串）
- `answer_evaluation`（字符串，一句话）
- `knowledge_points`（字符串数组）

仅输出 JSON 数组，不要任何额外文本。
"""


def summarize_pairs_prompt(*, pair_analyses: list[dict[str, Any]]) -> str:
    """Render the user prompt for the token-overflow summarizer (Prompt 2.5)."""
    tmpl = _ENV.from_string(SUMMARIZE_PAIRS_USER_TEMPLATE)
    return tmpl.render(pair_analyses=pair_analyses)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_prompts.py -v
```

Expected: 8 tests pass.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src/audio_interview_analyze/analysis/prompts.py tests/test_prompts.py
git add src/audio_interview_analyze/analysis/prompts.py tests/test_prompts.py tests/fixtures/sample_pair_analysis.json
git commit -m "feat: add Jinja2 prompt templates for the LLM calls"
```

---

### Task 8: Report renderer

**Files:**
- Create: `src/audio_interview_analyze/report/render.py`
- Create: `tests/test_render.py`
- Create: `tests/fixtures/sample_final_report.json`

- [ ] **Step 1: Create the sample final report fixture**

Write `tests/fixtures/sample_final_report.json`:

```json
{
  "overall_evaluation": "候选人技术基础扎实，系统设计经验略有不足。",
  "overall_score": 78,
  "dimension_scores": {"技术深度": 80, "思路结构": 75, "表达清晰度": 82, "反应速度": 75},
  "consolidated_highlights": ["数据结构基础扎实", "能够区分相似概念"],
  "consolidated_weaknesses": ["缺少时间复杂度分析", "未举例说明"],
  "consolidated_knowledge_points": [
    {
      "name": "红黑树",
      "category": "数据结构",
      "explanation": "一种自平衡的二叉查找树，通过在节点上记录颜色和约束规则保证近似平衡。",
      "learning_directions": ["阅读《算法导论》第 13 章", "实现一个简单的红黑树"]
    }
  ],
  "pair_analyses": [
    {
      "pair_index": 1,
      "question_summary": "红黑树与 AVL 树的区别",
      "answer_evaluation": "回答基本正确。",
      "knowledge_points": ["红黑树", "AVL 树"],
      "highlights": ["提到了自平衡"],
      "weaknesses": ["缺少时间复杂度分析"],
      "improvement_suggestions": "补充复杂度分析。"
    }
  ],
  "full_transcript": {
    "duration": 60.0,
    "utterances": [
      {"speaker": "interviewer", "text": "讲讲红黑树", "start": 0.0, "end": 1.0},
      {"speaker": "candidate", "text": "红黑树是一种自平衡二叉查找树", "start": 1.5, "end": 4.0}
    ]
  }
}
```

- [ ] **Step 2: Write the failing tests**

Write `tests/test_render.py`:

```python
"""Tests for the Markdown report renderer."""

import json
from pathlib import Path

import pytest

from audio_interview_analyze.report.model import FinalReport
from audio_interview_analyze.report.render import render_markdown


@pytest.fixture
def sample_report() -> FinalReport:
    fixture = Path(__file__).parent / "fixtures" / "sample_final_report.json"
    return FinalReport.model_validate(json.loads(fixture.read_text(encoding="utf-8")))


def test_render_contains_all_six_sections(sample_report):
    md = render_markdown(sample_report)
    for section in [
        "## 1. 面试总评",
        "## 2. 亮点",
        "## 3. 不足",
        "## 4. 逐题分析",
        "## 5. 知识点总结",
        "## 附录：完整转写",
    ]:
        assert section in md, f"Missing section: {section}"


def test_render_includes_overall_score(sample_report):
    md = render_markdown(sample_report)
    assert "**78**" in md or "78" in md
    assert "技术深度" in md
    assert "80" in md  # one of the dimension scores


def test_render_lists_consolidated_highlights(sample_report):
    md = render_markdown(sample_report)
    assert "数据结构基础扎实" in md
    assert "能够区分相似概念" in md


def test_render_lists_consolidated_weaknesses(sample_report):
    md = render_markdown(sample_report)
    assert "缺少时间复杂度分析" in md


def test_render_knowledge_point_with_category_and_directions(sample_report):
    md = render_markdown(sample_report)
    assert "红黑树" in md
    assert "数据结构" in md
    assert "《算法导论》第 13 章" in md


def test_render_per_question_block(sample_report):
    md = render_markdown(sample_report)
    assert "第 1 题" in md
    assert "红黑树与 AVL 树的区别" in md
    assert "回答基本正确" in md


def test_render_handles_empty_highlights_and_weaknesses():
    report = FinalReport(
        overall_evaluation="简洁。",
        overall_score=70,
        dimension_scores={"技术深度": 70, "思路结构": 70, "表达清晰度": 70, "反应速度": 70},
        consolidated_highlights=[],
        consolidated_weaknesses=[],
        consolidated_knowledge_points=[],
        pair_analyses=[],
        full_transcript={"duration": 0.0, "utterances": []},
    )
    md = render_markdown(report)
    assert "（无）" in md


def test_render_includes_transcript_appendix(sample_report):
    md = render_markdown(sample_report)
    assert "讲讲红黑树" in md
    assert "自平衡" in md
    # Speaker labels appear in the transcript
    assert "[面试官]" in md or "面试官" in md
    assert "[候选人]" in md or "候选人" in md


def test_render_uses_markdown_heading_hierarchy(sample_report):
    md = render_markdown(sample_report)
    lines = md.splitlines()
    h1 = [l for l in lines if l.startswith("# ")]
    h2 = [l for l in lines if l.startswith("## ")]
    assert len(h1) >= 1
    assert len(h2) >= 6
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_render.py -v
```

Expected: `ModuleNotFoundError: No module named 'audio_interview_analyze.report.render'`.

- [ ] **Step 4: Implement the renderer**

Write `src/audio_interview_analyze/report/render.py`:

```python
"""Render a FinalReport as Markdown."""

from __future__ import annotations

from audio_interview_analyze.report.model import FinalReport, Utterance

SPEAKER_LABEL = {
    "interviewer": "[面试官]",
    "candidate": "[候选人]",
    "unknown": "[未知]",
}


def _format_transcript_line(u: Utterance) -> str:
    label = SPEAKER_LABEL.get(u.speaker, f"[{u.speaker}]")
    ts = f"[{u.start:0.1f}s-{u.end:0.1f}s]"
    return f"- {ts} {label} {u.text}"


def render_markdown(report: FinalReport) -> str:
    """Render the report as a Markdown string."""
    sections: list[str] = []

    # Title
    sections.append("# 面试分析报告")
    sections.append("")

    # 1. Overall evaluation
    sections.append("## 1. 面试总评")
    sections.append("")
    sections.append(f"**整体得分：{report.overall_score} / 100**")
    sections.append("")
    sections.append(report.overall_evaluation.strip())
    sections.append("")
    sections.append("**分维度得分：**")
    sections.append("")
    for dim, score in report.dimension_scores.items():
        sections.append(f"- {dim}：{score}")
    sections.append("")

    # 2. Highlights
    sections.append("## 2. 亮点")
    sections.append("")
    if report.consolidated_highlights:
        for h in report.consolidated_highlights:
            sections.append(f"- {h}")
    else:
        sections.append("（无）")
    sections.append("")

    # 3. Weaknesses
    sections.append("## 3. 不足")
    sections.append("")
    if report.consolidated_weaknesses:
        for w in report.consolidated_weaknesses:
            sections.append(f"- {w}")
    else:
        sections.append("（无）")
    sections.append("")

    # 4. Per-question analysis
    sections.append("## 4. 逐题分析")
    sections.append("")
    if report.pair_analyses:
        for p in report.pair_analyses:
            sections.append(f"### 第 {p.pair_index} 题")
            sections.append("")
            sections.append(f"**题目摘要：** {p.question_summary}")
            sections.append("")
            sections.append(f"**答案评价：** {p.answer_evaluation}")
            sections.append("")
            if p.highlights:
                sections.append("**亮点：**")
                for h in p.highlights:
                    sections.append(f"- {h}")
                sections.append("")
            if p.weaknesses:
                sections.append("**不足：**")
                for w in p.weaknesses:
                    sections.append(f"- {w}")
                sections.append("")
            if p.improvement_suggestions:
                sections.append(f"**改进建议：** {p.improvement_suggestions}")
                sections.append("")
    else:
        sections.append("（无）")
        sections.append("")

    # 5. Knowledge points
    sections.append("## 5. 知识点总结")
    sections.append("")
    if report.consolidated_knowledge_points:
        # Group by category
        by_cat: dict[str, list] = {}
        for kp in report.consolidated_knowledge_points:
            by_cat.setdefault(kp.category, []).append(kp)
        for cat, kps in by_cat.items():
            sections.append(f"### {cat}")
            sections.append("")
            for kp in kps:
                sections.append(f"#### {kp.name}")
                sections.append("")
                sections.append(kp.explanation.strip())
                sections.append("")
                if kp.learning_directions:
                    sections.append("**拓展学习方向：**")
                    for d in kp.learning_directions:
                        sections.append(f"- {d}")
                    sections.append("")
    else:
        sections.append("（无）")
        sections.append("")

    # Appendix: full transcript
    sections.append("## 附录：完整转写")
    sections.append("")
    if report.full_transcript.utterances:
        for u in report.full_transcript.utterances:
            sections.append(_format_transcript_line(u))
    else:
        sections.append("（无）")
    sections.append("")

    return "\n".join(sections)
```

Write `src/audio_interview_analyze/report/__init__.py` (empty):

```python
"""Report data model and Markdown rendering."""
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_render.py -v
```

Expected: 9 tests pass.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src/audio_interview_analyze/report/ tests/test_render.py tests/fixtures/sample_final_report.json
git add src/audio_interview_analyze/report/ tests/test_render.py tests/fixtures/sample_final_report.json
git commit -m "feat: add Markdown report renderer"
```

---

## Phase 2: LLM Integration

### Task 9: DeepSeek client with retry

**Files:**
- Create: `src/audio_interview_analyze/llm/deepseek.py`
- Create: `tests/test_deepseek.py`
- Create: `src/audio_interview_analyze/llm/__init__.py` (empty)

- [ ] **Step 1: Write the failing tests**

Write `tests/test_deepseek.py`:

```python
"""Tests for the DeepSeek HTTP client."""

import httpx
import pytest
import respx

from audio_interview_analyze.llm.deepseek import (
    DeepSeekClient,
    DeepSeekError,
    DeepSeekRateLimitError,
)


@pytest.fixture
def client() -> DeepSeekClient:
    return DeepSeekClient(api_key="test-key", base_url="https://api.deepseek.com")


@respx.mock
def test_chat_sends_correct_request(client):
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            },
        )
    )
    result = client.chat(
        system="sys",
        user="user",
        model="deepseek-chat",
        temperature=0.3,
        max_tokens=100,
        response_format=None,
    )
    assert result == "hello"
    assert route.called
    sent = route.calls.last.request
    import json as _json

    body = _json.loads(sent.content)
    assert body["model"] == "deepseek-chat"
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 100
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]


@respx.mock
def test_chat_includes_response_format_when_set(client):
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    )
    client.chat(
        system="s",
        user="u",
        model="deepseek-chat",
        temperature=0.3,
        max_tokens=100,
        response_format={"type": "json_object"},
    )
    import json as _json

    body = _json.loads(route.calls.last.request.content)
    assert body["response_format"] == {"type": "json_object"}


@respx.mock
def test_chat_retries_on_5xx(client):
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(503, json={"error": "unavailable"}),
            httpx.Response(503, json={"error": "unavailable"}),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    result = client.chat(
        system="s",
        user="u",
        model="deepseek-chat",
        temperature=0.3,
        max_tokens=10,
        response_format=None,
    )
    assert result == "ok"
    assert route.call_count == 3


@respx.mock
def test_chat_raises_on_401(client):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    with pytest.raises(DeepSeekError, match="authentication"):
        client.chat(
            system="s",
            user="u",
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=10,
            response_format=None,
        )


@respx.mock
def test_chat_raises_on_429_after_retries(client):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": "rate limit"})
    )
    with pytest.raises(DeepSeekRateLimitError):
        client.chat(
            system="s",
            user="u",
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=10,
            response_format=None,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_deepseek.py -v
```

Expected: `ModuleNotFoundError: No module named 'audio_interview_analyze.llm.deepseek'`.

- [ ] **Step 3: Implement the DeepSeek client**

Write `src/audio_interview_analyze/llm/__init__.py`:

```python
"""LLM clients (DeepSeek for v1)."""
```

Write `src/audio_interview_analyze/llm/deepseek.py`:

```python
"""DeepSeek HTTP client with retry / backoff.

DeepSeek exposes an OpenAI-compatible chat-completions endpoint, so we use
plain httpx rather than the OpenAI SDK. Retry policy: 3 attempts with
exponential backoff (2s, 4s, 8s) on 5xx and 429. 4xx other than 429 are
treated as terminal errors with a clear message.
"""

from __future__ import annotations

import time
from typing import Any

import httpx


class DeepSeekError(Exception):
    """Base class for DeepSeek client errors."""


class DeepSeekAuthError(DeepSeekError):
    """401 / 403 — bad API key."""


class DeepSeekRateLimitError(DeepSeekError):
    """429 — rate limit, even after retries."""


class DeepSeekClient:
    """Thin wrapper around the DeepSeek chat-completions API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_seconds: tuple[float, ...] = (2.0, 4.0, 8.0),
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def chat(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> str:
        """Call DeepSeek chat-completions and return the assistant content string.

        Raises:
            DeepSeekAuthError: on 401 / 403.
            DeepSeekRateLimitError: on 429 after exhausting retries.
            DeepSeekError: on other non-2xx responses.
        """
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format is not None:
            body["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, headers=headers, json=body)
            except httpx.HTTPError as e:
                last_error = e
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_seconds[attempt])
                    continue
                raise DeepSeekError(f"HTTP transport error after {self.max_retries} attempts: {e}") from e

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise DeepSeekError("DeepSeek returned no choices")
                return choices[0]["message"]["content"]

            if resp.status_code in (401, 403):
                raise DeepSeekAuthError(
                    "DeepSeek authentication failed. Check DEEPSEEK_API_KEY."
                )
            if resp.status_code == 429:
                last_error = DeepSeekRateLimitError("DeepSeek rate limit hit.")
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_seconds[attempt])
                    continue
                raise last_error
            if 500 <= resp.status_code < 600:
                last_error = DeepSeekError(
                    f"DeepSeek server error: {resp.status_code} {resp.text[:200]}"
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_seconds[attempt])
                    continue
                raise last_error

            raise DeepSeekError(
                f"DeepSeek returned {resp.status_code}: {resp.text[:200]}"
            )

        # Should not be reached, but be defensive.
        if last_error is not None:
            raise last_error
        raise DeepSeekError("DeepSeek client exited retry loop unexpectedly")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_deepseek.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/audio_interview_analyze/llm/ tests/test_deepseek.py
git add src/audio_interview_analyze/llm/ tests/test_deepseek.py
git commit -m "feat: add DeepSeek HTTP client with retry and backoff"
```

---

### Task 10: Per-pair analysis LLM call

**Files:**
- Create: `src/audio_interview_analyze/analysis/analyze.py`
- Create: `tests/test_analyze.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/test_analyze.py`:

```python
"""Tests for the per-pair analysis call."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from audio_interview_analyze.analysis.analyze import analyze_pair
from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.report.model import PairAnalysis, QAPair


@pytest.fixture
def client() -> DeepSeekClient:
    return DeepSeekClient(api_key="k", base_url="https://api.deepseek.com")


@pytest.fixture
def sample_pair() -> QAPair:
    return QAPair(
        question="讲讲红黑树",
        answer="红黑树是一种自平衡二叉查找树",
        q_start=0.0,
        q_end=1.0,
        a_start=2.0,
        a_end=5.0,
        pair_index=1,
    )


@respx.mock
def test_analyze_pair_parses_json_response(client, sample_pair):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "question_summary": "红黑树的定义",
                                    "answer_evaluation": "回答正确。",
                                    "knowledge_points": ["红黑树", "自平衡"],
                                    "highlights": ["简洁"],
                                    "weaknesses": [],
                                    "improvement_suggestions": "可以举例。",
                                }
                            )
                        }
                    }
                ]
            },
        )
    )
    result = analyze_pair(client, sample_pair)
    assert isinstance(result, PairAnalysis)
    assert result.pair_index == 1
    assert result.question_summary == "红黑树的定义"
    assert result.knowledge_points == ["红黑树", "自平衡"]


@respx.mock
def test_analyze_pair_retries_on_invalid_json(client, sample_pair):
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "question_summary": "X",
                                        "answer_evaluation": "Y",
                                        "knowledge_points": [],
                                        "highlights": [],
                                        "weaknesses": [],
                                        "improvement_suggestions": "Z",
                                    }
                                )
                            }
                        }
                    ]
                },
            ),
        ]
    )
    result = analyze_pair(client, sample_pair)
    assert result.question_summary == "X"
    assert route.call_count == 2


@respx.mock
def test_analyze_pair_uses_response_format_json_object(client, sample_pair):
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "question_summary": "X",
                                    "answer_evaluation": "Y",
                                    "knowledge_points": [],
                                    "highlights": [],
                                    "weaknesses": [],
                                    "improvement_suggestions": "Z",
                                }
                            )
                        }
                    }
                ]
            },
        )
    )
    analyze_pair(client, sample_pair)
    import json as _json

    body = _json.loads(route.calls.last.request.content)
    assert body["response_format"] == {"type": "json_object"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_analyze.py -v
```

Expected: `ModuleNotFoundError: No module named 'audio_interview_analyze.analysis.analyze'`.

- [ ] **Step 3: Implement the per-pair analyzer**

Write `src/audio_interview_analyze/analysis/analyze.py`:

```python
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
from audio_interview_analyze.cache import write_json
from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.report.model import PairAnalysis, QAPair

ANALYZE_MODEL = "deepseek-chat"
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_analyze.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/audio_interview_analyze/analysis/analyze.py tests/test_analyze.py
git add src/audio_interview_analyze/analysis/analyze.py tests/test_analyze.py
git commit -m "feat: add per-pair LLM analysis with JSON retry"
```

---

### Task 11: Aggregator with token-overflow summarizer

**Files:**
- Create: `src/audio_interview_analyze/analysis/aggregate.py`
- Create: `tests/test_aggregate.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/test_aggregate.py`:

```python
"""Tests for the aggregator (LLM #2) and its token-overflow summarizer."""

import json

import httpx
import pytest
import respx

from audio_interview_analyze.analysis.aggregate import (
    AGGREGATE_MAX_INPUT_TOKENS,
    aggregate,
    maybe_summarize,
)
from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.report.model import PairAnalysis


@pytest.fixture
def client() -> DeepSeekClient:
    return DeepSeekClient(api_key="k", base_url="https://api.deepseek.com")


@pytest.fixture
def many_pair_analyses() -> list[PairAnalysis]:
    """Build a list large enough to exceed the token threshold when serialized."""
    return [
        PairAnalysis(
            pair_index=i,
            question_summary=f"题目 {i} 考察的知识点",
            answer_evaluation="回答涉及多个方面，整体表现尚可但缺少深入分析。",
            knowledge_points=[f"知识点-{i}-A", f"知识点-{i}-B"],
            highlights=[f"亮点 {i}"],
            weaknesses=[f"不足 {i}"],
            improvement_suggestions=f"建议 {i}",
        )
        for i in range(1, 200)
    ]


@respx.mock
def test_aggregate_parses_full_report(client):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "overall_evaluation": "整体表现良好。",
                                    "overall_score": 82,
                                    "dimension_scores": {
                                        "技术深度": 80,
                                        "思路结构": 85,
                                        "表达清晰度": 80,
                                        "反应速度": 83,
                                    },
                                    "consolidated_highlights": ["基础扎实"],
                                    "consolidated_weaknesses": ["深度不足"],
                                    "consolidated_knowledge_points": [
                                        {
                                            "name": "红黑树",
                                            "category": "数据结构",
                                            "explanation": "自平衡二叉查找树。",
                                            "learning_directions": ["读《算法导论》"],
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        )
    )
    pairs = [
        PairAnalysis(
            pair_index=1,
            question_summary="红黑树",
            answer_evaluation="回答正确。",
            knowledge_points=["红黑树"],
            highlights=[],
            weaknesses=[],
            improvement_suggestions="补充复杂度。",
        )
    ]
    report = aggregate(client, pairs, candidate_background="")
    assert report.overall_score == 82
    assert report.dimension_scores["技术深度"] == 80
    assert report.consolidated_knowledge_points[0].name == "红黑树"
    assert report.pair_analyses == pairs


@respx.mock
def test_aggregate_retries_on_invalid_json(client):
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "overall_evaluation": "X",
                                        "overall_score": 70,
                                        "dimension_scores": {
                                            "技术深度": 70,
                                            "思路结构": 70,
                                            "表达清晰度": 70,
                                            "反应速度": 70,
                                        },
                                        "consolidated_highlights": [],
                                        "consolidated_weaknesses": [],
                                        "consolidated_knowledge_points": [],
                                    }
                                )
                            }
                        }
                    ]
                },
            ),
        ]
    )
    pairs = [
        PairAnalysis(
            pair_index=1,
            question_summary="Q",
            answer_evaluation="A",
            knowledge_points=[],
            highlights=[],
            weaknesses=[],
            improvement_suggestions="S",
        )
    ]
    report = aggregate(client, pairs, candidate_background="")
    assert report.overall_score == 70


@respx.mock
def test_maybe_summarize_skipped_when_small(client):
    """When the serialized pair list is small, summarizer should not be called."""
    pairs = [
        PairAnalysis(
            pair_index=1,
            question_summary="Q",
            answer_evaluation="A",
            knowledge_points=[],
            highlights=[],
            weaknesses=[],
            improvement_suggestions="S",
        )
    ]
    result = maybe_summarize(client, pairs)
    assert result == pairs  # unchanged


@respx.mock
def test_maybe_summarize_called_when_large(client, many_pair_analyses):
    summarized = [
        {
            "pair_index": p.pair_index,
            "question_summary": p.question_summary,
            "answer_evaluation": "压缩后的评价。",
            "knowledge_points": p.knowledge_points,
        }
        for p in many_pair_analyses
    ]
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(summarized)}}]},
        )
    )
    result = maybe_summarize(client, many_pair_analyses)
    assert len(result) == len(many_pair_analyses)
    assert all("answer_evaluation" in item for item in result)
    # The summarized form should be smaller than the original PairAnalysis dicts
    original_size = sum(len(json.dumps(p.model_dump())) for p in many_pair_analyses)
    summarized_size = sum(len(json.dumps(item)) for item in result)
    assert summarized_size < original_size


def test_aggregate_max_input_tokens_constant_is_reasonable():
    """The constant should be positive and well below DeepSeek's 64K context."""
    assert 10_000 < AGGREGATE_MAX_INPUT_TOKENS < 60_000
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_aggregate.py -v
```

Expected: `ModuleNotFoundError: No module named 'audio_interview_analyze.analysis.aggregate'`.

- [ ] **Step 3: Implement the aggregator**

Write `src/audio_interview_analyze/analysis/aggregate.py`:

```python
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
    # The aggregator is not asked to produce the full transcript; the pipeline
    # attaches it after the fact.
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_aggregate.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/audio_interview_analyze/analysis/aggregate.py tests/test_aggregate.py
git add src/audio_interview_analyze/analysis/aggregate.py tests/test_aggregate.py
git commit -m "feat: add aggregator with token-overflow summarizer"
```

---

## Phase 3: Audio Pipeline

### Task 12: Audio extraction (video/audio → 16kHz mono WAV)

**Files:**
- Create: `src/audio_interview_analyze/audio/extract.py`
- Create: `src/audio_interview_analyze/audio/__init__.py` (empty)

- [ ] **Step 1: Write the audio extraction module**

This module has no unit tests in v1 (it shells out to `ffmpeg` via `pydub`); it is exercised by the smoke test in Task 18. Document the public API here.

Write `src/audio_interview_analyze/audio/__init__.py`:

```python
"""Audio processing: extraction, diarization, transcription."""
```

Write `src/audio_interview_analyze/audio/extract.py`:

```python
"""Extract a 16 kHz mono WAV from any input audio or video file.

Delegates to ``ffmpeg`` via ``pydub``. Caller is expected to have ``ffmpeg``
on PATH (verified by the preflight check).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pydub import AudioSegment


def extract_wav(input_path: Path | str, output_path: Path | str) -> Path:
    """Read ``input_path`` and write a 16 kHz mono WAV to ``output_path``.

    Returns the output path. Raises ``FileNotFoundError`` if the input is
    missing and ``RuntimeError`` if ``ffmpeg`` is not on PATH.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`).")
    in_path = Path(input_path)
    out_path = Path(output_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(out_path, format="wav")
    return out_path
```

- [ ] **Step 2: Verify the import works**

```bash
uv run python -c "from audio_interview_analyze.audio.extract import extract_wav; print(extract_wav.__doc__)"
```

Expected: prints the docstring.

- [ ] **Step 3: Lint and commit**

```bash
uv run ruff check src/audio_interview_analyze/audio/
git add src/audio_interview_analyze/audio/
git commit -m "feat: add audio extraction (video/audio → 16kHz mono WAV)"
```

---

### Task 13: Diarization wrapper

**Files:**
- Create: `src/audio_interview_analyze/audio/diarize.py`

- [ ] **Step 1: Implement the diarization wrapper**

Write `src/audio_interview_analyze/audio/diarize.py`:

```python
"""Speaker diarization using pyannote-audio.

Returns a list of ``(start, end, speaker_label)`` tuples. Thin wrapper:
exception translation only.
"""

from __future__ import annotations

from pathlib import Path


def diarize(audio_path: Path | str, *, hf_token: str | None = None) -> list[tuple[float, float, str]]:
    """Run pyannote-audio speaker diarization on ``audio_path``.

    Returns a list of ``(start_sec, end_sec, speaker_label)`` tuples. Raises
    ``RuntimeError`` if the pyannote model is not available locally — the
    caller is expected to have run the preflight check and downloaded the
    model before reaching this point.
    """
    try:
        from pyannote.audio import Pipeline  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError("pyannote-audio is not installed. Run `uv add pyannote-audio`.") from e

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load pyannote diarization model. "
            "Accept the model terms on HuggingFace and set HF_TOKEN. See README. "
            f"Underlying error: {e}"
        ) from e

    diarization = pipeline(str(audio_path))
    segments: list[tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((float(turn.start), float(turn.end), str(speaker)))
    return segments
```

- [ ] **Step 2: Verify the import works**

```bash
uv run python -c "from audio_interview_analyze.audio.diarize import diarize; print(diarize.__doc__)"
```

Expected: prints the docstring.

- [ ] **Step 3: Lint and commit**

```bash
uv run ruff check src/audio_interview_analyze/audio/diarize.py
git add src/audio_interview_analyze/audio/diarize.py
git commit -m "feat: add pyannote-audio diarization wrapper"
```

---

### Task 14: Transcription with speaker alignment

**Files:**
- Create: `src/audio_interview_analyze/audio/transcribe.py`

- [ ] **Step 1: Implement transcription and speaker alignment**

Write `src/audio_interview_analyze/audio/transcribe.py`:

```python
"""Transcription with speaker alignment.

Run ``faster-whisper`` on the audio, then intersect the word-level
timestamps with the diarization segments to produce ordered
``Utterance(speaker, text, start, end)`` objects.
"""

from __future__ import annotations

from faster_whisper import WhisperModel  # type: ignore[import-untyped]

from audio_interview_analyze.report.model import Transcript, Utterance


def transcribe(
    audio_path: str | bytes,
    *,
    model_size: str = "large-v3",
    device: str = "auto",
    compute_type: str = "default",
) -> list[dict]:
    """Run Whisper and return a list of ``{"start", "end", "text", "words"}``.

    Each ``words`` entry is ``{"start", "end", "word"}``. The model is
    loaded once per process; callers should reuse the returned list rather
    than re-running on the same file.
    """
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(audio_path, vad_filter=True, word_timestamps=True)
    out: list[dict] = []
    for seg in segments:
        out.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "words": [
                    {"start": float(w.start), "end": float(w.end), "word": w.word}
                    for w in (seg.words or [])
                ],
            }
        )
    return out


def assign_speakers(
    whisper_segments: list[dict],
    diarization: list[tuple[float, float, str]],
) -> Transcript:
    """Intersect Whisper word timestamps with diarization segments to produce
    a labeled ``Transcript``.

    Strategy: for each Whisper segment, find the diarization segment that
    overlaps the most (by total overlapping duration), and use its speaker
    label for the whole segment. This is good enough when Whisper segments
    are sentence-sized.
    """
    utterances: list[Utterance] = []
    for seg in whisper_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        text = seg["text"]
        if not text:
            continue
        best_label = "unknown"
        best_overlap = 0.0
        for d_start, d_end, label in diarization:
            overlap = max(0.0, min(seg_end, d_end) - max(seg_start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        utterances.append(
            Utterance(
                speaker=best_label,
                text=text,
                start=seg_start,
                end=seg_end,
            )
        )
    duration = max((u.end for u in utterances), default=0.0)
    return Transcript(utterances=utterances, duration=duration)
```

- [ ] **Step 2: Verify the import works**

```bash
uv run python -c "from audio_interview_analyze.audio.transcribe import transcribe, assign_speakers; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Lint and commit**

```bash
uv run ruff check src/audio_interview_analyze/audio/transcribe.py
git add src/audio_interview_analyze/audio/transcribe.py
git commit -m "feat: add Whisper transcription with speaker alignment"
```

---

## Phase 4: Pre-flight and Orchestration

### Task 15: Pre-flight check

**Files:**
- Create: `src/audio_interview_analyze/preflight.py`
- Create: `tests/test_preflight.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/test_preflight.py`:

```python
"""Tests for the pre-flight environment check."""

import pytest

from audio_interview_analyze.preflight import (
    PreflightResult,
    check_api_key,
    check_ffmpeg,
    check_hf_token,
    run_preflight,
)


def test_check_ffmpeg_passes_when_on_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")
    assert check_ffmpeg() is True


def test_check_ffmpeg_fails_when_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert check_ffmpeg() is False


def test_check_api_key_passes_when_set(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    assert check_api_key() is True


def test_check_api_key_fails_when_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert check_api_key() is False


def test_check_hf_token_passes_when_set(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    assert check_hf_token() is True


def test_check_hf_token_passes_when_hugging_face_token_set(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test-token")
    assert check_hf_token() is True


def test_check_hf_token_fails_when_missing(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    assert check_hf_token() is False


def test_run_preflight_returns_structured_result(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("HF_TOKEN", "t")
    result = run_preflight()
    assert isinstance(result, PreflightResult)
    assert result.ffmpeg is True
    assert result.deepseek_api_key is True
    assert result.hf_token is True
    assert result.all_ok is True
    assert result.failures == []


def test_run_preflight_collects_all_failures(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    result = run_preflight()
    assert result.all_ok is False
    assert len(result.failures) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_preflight.py -v
```

Expected: `ModuleNotFoundError: No module named 'audio_interview_analyze.preflight'`.

- [ ] **Step 3: Implement the preflight module**

Write `src/audio_interview_analyze/preflight.py`:

```python
"""Pre-flight environment check.

Verifies that the three external dependencies are present:
1. ``ffmpeg`` on PATH.
2. ``DEEPSEEK_API_KEY`` env var set.
3. HuggingFace token (``HF_TOKEN`` or ``HUGGINGFACE_TOKEN``) set, required
   for the pyannote-audio model.

Returns a structured ``PreflightResult`` so the CLI can render a clear
checklist and link to the relevant README sections.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field


@dataclass
class PreflightResult:
    ffmpeg: bool
    deepseek_api_key: bool
    hf_token: bool
    failures: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return not self.failures


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def check_api_key() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY", "").strip())


def check_hf_token() -> bool:
    return bool(os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACE_TOKEN", "").strip())


def run_preflight() -> PreflightResult:
    ffmpeg_ok = check_ffmpeg()
    api_key_ok = check_api_key()
    hf_ok = check_hf_token()
    failures: list[str] = []
    if not ffmpeg_ok:
        failures.append("ffmpeg not found on PATH (install: `brew install ffmpeg`)")
    if not api_key_ok:
        failures.append("DEEPSEEK_API_KEY not set (see README §Configuration)")
    if not hf_ok:
        failures.append("HuggingFace token not set (set HF_TOKEN; required for pyannote-audio)")
    return PreflightResult(
        ffmpeg=ffmpeg_ok,
        deepseek_api_key=api_key_ok,
        hf_token=hf_ok,
        failures=failures,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_preflight.py -v
```

Expected: 9 tests pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/audio_interview_analyze/preflight.py tests/test_preflight.py
git add src/audio_interview_analyze/preflight.py tests/test_preflight.py
git commit -m "feat: add pre-flight environment check"
```

---

### Task 16: Pipeline orchestration

**Files:**
- Create: `src/audio_interview_analyze/pipeline.py`

- [ ] **Step 1: Implement the pipeline orchestrator**

Write `src/audio_interview_analyze/pipeline.py`:

```python
"""Pipeline orchestrator.

Glues the seven stages together:
  extract → diarize → transcribe → build_qa_pairs → analyze_pair (LLM #1) →
  aggregate (LLM #2) → render_markdown

Owns the cache (writes intermediate artifacts to ``./.interview-cache/<hash>/``)
and the ``rich`` progress display. The actual logic of each stage lives in
its own module — this file is the glue.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from audio_interview_analyze.analysis.aggregate import aggregate
from audio_interview_analyze.analysis.analyze import analyze_pair
from audio_interview_analyze.analysis.qa_pairs import build_qa_pairs, label_speakers
from audio_interview_analyze.audio.diarize import diarize
from audio_interview_analyze.audio.extract import extract_wav
from audio_interview_analyze.audio.transcribe import assign_speakers, transcribe
from audio_interview_analyze.cache import (
    artifact_path,
    content_hash,
    read_json,
    write_json,
)
from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.preflight import run_preflight
from audio_interview_analyze.report.model import FinalReport, PairAnalysis, Transcript
from audio_interview_analyze.report.render import render_markdown


@dataclass
class PipelineConfig:
    input_path: Path
    output_path: Path
    whisper_model: str = "large-v3"
    reuse_cache: bool = False
    candidate_background: str = ""


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    )


def run_pipeline(
    config: PipelineConfig,
    *,
    client: DeepSeekClient,
    hf_token: str | None = None,
) -> str:
    """Run the full pipeline. Returns the Markdown report as a string.

    Also writes the Markdown to ``config.output_path``.
    """
    preflight = run_preflight()
    if not preflight.all_ok:
        msg = "Pre-flight check failed:\n" + "\n".join(f"  - {f}" for f in preflight.failures)
        raise RuntimeError(msg)

    input_path = Path(config.input_path).expanduser().resolve()
    output_path = Path(config.output_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    hash_key = content_hash(input_path)
    pair_cache_dir = artifact_path("pairs_raw", hash_key=hash_key)

    with _progress() as progress:
        # Stage 1: extract
        task = progress.add_task("[cyan]Extracting audio...", total=1)
        wav_path = artifact_path("audio.wav", hash_key=hash_key)
        if config.reuse_cache and wav_path.exists():
            pass
        else:
            extract_wav(input_path, wav_path)
        progress.update(task, completed=1)

        # Stage 2: diarize
        task = progress.add_task("[cyan]Diarizing speakers...", total=1)
        diar_cache = read_json("diarization.json", hash_key=hash_key)
        if config.reuse_cache and diar_cache is not None:
            diarization = [(float(s[0]), float(s[1]), str(s[2])) for s in diar_cache]
        else:
            diarization = diarize(wav_path, hf_token=hf_token)
            write_json("diarization.json", diarization, hash_key=hash_key)
        progress.update(task, completed=1)

        # Stage 3: transcribe
        task = progress.add_task("[cyan]Transcribing...", total=1)
        transcript_cache = read_json("transcript.json", hash_key=hash_key)
        if config.reuse_cache and transcript_cache is not None:
            transcript = Transcript.model_validate(transcript_cache)
        else:
            whisper_segs = transcribe(str(wav_path), model_size=config.whisper_model)
            transcript = assign_speakers(whisper_segs, diarization)
            write_json("transcript.json", transcript.model_dump(), hash_key=hash_key)
        progress.update(task, completed=1)

        # Stage 4: build Q+A pairs
        task = progress.add_task("[cyan]Building Q&A pairs...", total=1)
        labeled = label_speakers(transcript)
        pairs = build_qa_pairs(labeled)
        # Skip pairs with empty questions or very short answers (< 20 chars)
        analyzable = [p for p in pairs if p.question and len(p.answer.strip()) >= 20]
        progress.update(task, completed=1)
        if not analyzable:
            raise RuntimeError(
                "No analyzable Q+A pairs detected. "
                "Check that the audio has two distinct speakers and clear questions."
            )

        # Stage 5: per-pair analysis
        task = progress.add_task(
            "[cyan]Analyzing pairs...", total=len(analyzable)
        )
        pair_analyses: list[PairAnalysis] = []
        for p in analyzable:
            pa = analyze_pair(client, p, cache_dir=pair_cache_dir)
            pair_analyses.append(pa)
            progress.update(task, advance=1)

        # Stage 6: aggregate
        task = progress.add_task("[cyan]Aggregating report...", total=1)
        report = aggregate(client, pair_analyses, config.candidate_background)
        # Attach the full transcript (the aggregator doesn't produce it).
        report = report.model_copy(update={"full_transcript": transcript})
        write_json("final_report.json", report.model_dump(), hash_key=hash_key)
        progress.update(task, completed=1)

    # Stage 7: render
    md = render_markdown(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    return md


def copy_default_cache_to_cwd(cache_root: Path, target: Path) -> None:
    """Helper used by tests to move the cache out of the project root."""
    if cache_root.exists():
        shutil.move(str(cache_root), str(target))


__all__ = ["PipelineConfig", "run_pipeline"]


---

### Task 17: CLI with rich progress and preflight

**Files:**
- Create: `src/audio_interview_analyze/cli.py`

- [ ] **Step 1: Implement the CLI**

Write `src/audio_interview_analyze/cli.py`:

```python
"""Command-line interface for audio-interview-analyze."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from audio_interview_analyze.llm.deepseek import DeepSeekClient
from audio_interview_analyze.pipeline import PipelineConfig, run_pipeline
from audio_interview_analyze.preflight import run_preflight


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio-interview-analyze",
        description="Transcribe a technical interview and produce a Markdown analysis report.",
    )
    p.add_argument("input", type=Path, help="Path to interview audio or video file")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("./report.md"),
        help="Output Markdown path (default: ./report.md)",
    )
    p.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model size (default: large-v3; small/medium are faster downloads)",
    )
    p.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Skip audio extraction, diarization, and transcription if cached",
    )
    p.add_argument(
        "--preflight",
        action="store_true",
        help="Run preflight checks only; do not process the input",
    )
    p.add_argument(
        "--background",
        default="",
        help="Optional candidate background (years of experience, target role, etc.) to include in the aggregator prompt",
    )
    return p


def _print_preflight() -> int:
    result = run_preflight()
    print("Pre-flight check:")
    print(f"  [{'OK' if result.ffmpeg else 'FAIL'}] ffmpeg on PATH")
    print(f"  [{'OK' if result.deepseek_api_key else 'FAIL'}] DEEPSEEK_API_KEY set")
    print(f"  [{'OK' if result.hf_token else 'FAIL'}] HuggingFace token set")
    if result.all_ok:
        print("All checks passed.")
        return 0
    print("Failures:")
    for f in result.failures:
        print(f"  - {f}")
    return 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.preflight:
        return _print_preflight()

    preflight = run_preflight()
    if not preflight.all_ok:
        _print_preflight()
        return 2

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    client = DeepSeekClient(api_key=api_key)
    config = PipelineConfig(
        input_path=args.input,
        output_path=args.out,
        whisper_model=args.model,
        reuse_cache=args.reuse_cache,
        candidate_background=args.background,
    )
    try:
        run_pipeline(config, client=client, hf_token=hf_token)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Report written to: {args.out}")
    print(f"Cache: {Path.cwd() / '.interview-cache'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the CLI is installed**

```bash
uv run audio-interview-analyze --help
```

Expected: prints the help text with the four flags.

- [ ] **Step 3: Run the preflight-only mode**

```bash
uv run audio-interview-analyze --preflight
```

Expected: prints the checklist. Status of items will depend on your local environment; at minimum it should not crash.

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff check src/audio_interview_analyze/cli.py
git add src/audio_interview_analyze/cli.py
git commit -m "feat: add CLI with argparse, rich progress, and preflight"
```

---

### Task 18: Gated end-to-end smoke test

**Files:**
- Create: `tests/test_pipeline_smoke.py`
- Create: `tests/fixtures/silent_5s.wav` (generated, not committed)

- [ ] **Step 1: Generate the 5-second silent WAV fixture**

```bash
uv run python -c "
import wave, struct
with wave.open('tests/fixtures/silent_5s.wav', 'wb') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(b'\x00\x00' * 16000 * 5)
print('wrote tests/fixtures/silent_5s.wav')
"
```

Expected: prints `wrote tests/fixtures/silent_5s.wav`.

- [ ] **Step 2: Write the smoke test**

Write `tests/test_pipeline_smoke.py`:

```python
"""End-to-end smoke test (gated by RUN_SLOW=1).

Runs the full pipeline against a 5-second silent WAV. With no speech, the
pipeline should detect zero analyzable Q+A pairs and raise a clear error
*after* writing the audio.wav cache artifact. The test asserts that:

1. The error is the expected "no Q+A pairs" error.
2. The cache directory was created.
3. No LLM call is made (verified by the fact that we never construct a
   real DeepSeekClient in this test).

Real-audio smoke tests should be run manually by the developer after
setting up the environment.
"""

import os
from pathlib import Path

import pytest

from audio_interview_analyze.cache import cache_root
from audio_interview_analyze.pipeline import PipelineConfig, run_pipeline
from audio_interview_analyze.preflight import run_preflight


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SLOW") != "1",
    reason="set RUN_SLOW=1 to run the end-to-end smoke test",
)


def test_silent_audio_raises_no_qa_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Skip preflight by short-circuiting: the silent fixture doesn't need
    # ffmpeg or DeepSeek for the failure mode we want to test (we fail
    # *before* any LLM call). But run_preflight is called inside the
    # pipeline; we patch the preflight to be a no-op success.
    from audio_interview_analyze import pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_preflight", lambda: type("R", (), {"all_ok": True, "failures": []})())

    # Provide a real API key so the client can be constructed (it won't be
    # called).
    os.environ["DEEPSEEK_API_KEY"] = "fake-key-for-smoke"
    from audio_interview_analyze.llm.deepseek import DeepSeekClient

    client = DeepSeekClient(api_key="fake-key-for-smoke", base_url="http://localhost:1")

    config = PipelineConfig(
        input_path=Path(__file__).parent / "fixtures" / "silent_5s.wav",
        output_path=tmp_path / "report.md",
        whisper_model="tiny",  # smallest model, fastest for the smoke test
        reuse_cache=False,
    )

    with pytest.raises(RuntimeError, match="No analyzable Q+A pairs"):
        run_pipeline(config, client=client, hf_token=None)
```

- [ ] **Step 3: Verify the test is collected but skipped**

```bash
uv run pytest tests/test_pipeline_smoke.py -v
```

Expected: `1 skipped` (the `RUN_SLOW=1` env var is not set).

- [ ] **Step 4: Commit (do NOT run the smoke test in CI yet)**

```bash
git add tests/test_pipeline_smoke.py tests/fixtures/silent_5s.wav
git commit -m "test: add gated end-to-end smoke test with silent fixture"
```

---

### Task 19: README with setup instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the README**

Replace the contents of `README.md` with:

````markdown
# audio-interview-analyze

Analyze technical interview recordings: transcribe with speaker labels, evaluate
each Q&A pair with an LLM, and produce a Chinese Markdown report (overall
evaluation, highlights, weaknesses, per-question analysis, knowledge points).

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- [`ffmpeg`](https://ffmpeg.org/) on your `PATH` (`brew install ffmpeg` on macOS)
- A DeepSeek API key — sign up at <https://platform.deepseek.com/>
- A HuggingFace account + access token — sign up at <https://huggingface.co/>

## Setup

1. **Install dependencies:**

   ```bash
   uv sync
   ```

2. **Accept the pyannote-audio model terms:**

   Visit <https://huggingface.co/pyannote/speaker-diarization-3.1> and accept
   the model terms. Without this step the diarization step will fail with a
   401 / model-not-found error.

3. **Create a `.env` file in the project root:**

   ```bash
   cat > .env <<EOF
   DEEPSEEK_API_KEY=sk-...
   HF_TOKEN=hf_...
   EOF
   ```

   `HF_TOKEN` (or `HUGGINGFACE_TOKEN`) is required to download the
   pyannote-audio model on first run.

4. **Verify the environment:**

   ```bash
   uv run audio-interview-analyze --preflight
   ```

   You should see three green `[OK]` lines.

## Usage

```bash
# Basic: analyze an interview, write report.md to current dir
uv run audio-interview-analyze path/to/interview.mp4

# Custom output path
uv run audio-interview-analyze path/to/interview.mp4 --out ./reports/q1.md

# Reuse cached transcription (fast re-runs of the LLM stage only)
uv run audio-interview-analyze path/to/interview.mp4 --reuse-cache

# Use a smaller Whisper model for faster (less accurate) first run
uv run audio-interview-analyze path/to/interview.mp4 --model small

# Provide candidate background (years of experience, target role)
uv run audio-interview-analyze path/to/interview.mp4 --background "3 年 Java 后端"
```

The first run downloads ~3 GB of Whisper weights and ~100 MB of pyannote
weights; subsequent runs use the local cache.

## What you get

A Markdown report with these sections:

1. **面试总评** — overall score (0-100) + 4 dimension scores
2. **亮点** — consolidated strengths
3. **不足** — consolidated weaknesses
4. **逐题分析** — per-question evaluation (interviewer question + candidate
   answer + evaluation + improvement suggestions)
5. **知识点总结** — deduplicated knowledge points grouped by category
   (数据结构 / 算法 / 系统设计 / 编程语言 / 计算机网络 / 操作系统 / 数据库 /
   其它), each with an explanation and learning directions
6. **完整转写 (附录)** — full labeled transcript

## Architecture

See [`docs/superpowers/specs/2026-06-30-audio-interview-analyzer-design.md`](docs/superpowers/specs/2026-06-30-audio-interview-analyzer-design.md)
for the full design spec.

The pipeline has 7 stages:

```
extract → diarize → transcribe → build_qa_pairs → analyze_pair (LLM) →
aggregate (LLM) → render_markdown
```

All intermediate artifacts (extracted WAV, diarization, transcript, per-pair
JSON) are cached under `./.interview-cache/<content-hash>/` so re-runs are
cheap.

## Development

```bash
# Run unit tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run lint
uv run ruff check .

# Run the end-to-end smoke test (downloads models on first run)
RUN_SLOW=1 uv run pytest tests/test_pipeline_smoke.py -v
```

## Troubleshooting

- **"ffmpeg not found"** — install ffmpeg and ensure it's on your PATH.
- **"DEEPSEEK_API_KEY not set"** — create a `.env` file as shown above.
- **"Failed to load pyannote diarization model"** — accept the model terms
  on HuggingFace and set `HF_TOKEN`.
- **"No analyzable Q+A pairs detected"** — the audio may have only one
  speaker, or the recording quality is too low for diarization. Check the
  cached transcript under `.interview-cache/`.
- **Whisper transcription seems wrong** — try a larger model (`--model
  large-v3` is the default and most accurate), or check that the audio
  language matches what you expect.
````

- [ ] **Step 2: Verify the README renders reasonably**

```bash
uv run python -c "
from pathlib import Path
text = Path('README.md').read_text(encoding='utf-8')
assert '# audio-interview-analyze' in text
assert 'uv run audio-interview-analyze' in text
print('README ok')
"
```

Expected: prints `README ok`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup, usage, and architecture"
```

---

### Task 20: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest -v
```

Expected: all unit tests pass, the smoke test is skipped.

- [ ] **Step 2: Run lint**

```bash
uv run ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 3: Run the CLI in preflight mode**

```bash
uv run audio-interview-analyze --preflight
```

Expected: prints a checklist. Status will vary by environment; no crash.

- [ ] **Step 4: Verify the install entry point works**

```bash
uv run audio-interview-analyze --help
```

Expected: prints the help text.

- [ ] **Step 5: Tag the release**

```bash
git tag v0.1.0
git log --oneline | head -20
```

Expected: a clean linear history of ~20 commits, tagged `v0.1.0`.

---

## Summary of what was built

After completing all 20 tasks you will have:

- **Library code** under `src/audio_interview_analyze/` with one-file-per-concern organization.
- **Unit tests** for data model, cache, Q+A extraction, prompts, renderer, DeepSeek client, per-pair analyzer, aggregator (with summarizer), and preflight.
- **LLM tests** using `respx` mocks — no real API calls, no API key needed in CI.
- **A gated smoke test** that runs the full pipeline (set `RUN_SLOW=1`).
- **A CLI** with `--preflight`, `--reuse-cache`, `--model`, `--out`, `--background` flags and `rich` progress output.
- **A README** with setup, usage, and troubleshooting.
- **A cached pipeline** that re-runs cheaply thanks to the content-hash artifact store.

## Open questions / future work (out of scope for v1)

- Multiple LLM provider implementations (only DeepSeek ships; the interface allows swapping).
- Web UI / Gradio.
- PDF / HTML report formats (Markdown only).
- Three-or-more-speaker diarization (only the binary interviewer / candidate split).
- Spaced-repetition flashcard generation from the extracted knowledge points.
- Streaming token output for the LLM calls (batch analysis only).
