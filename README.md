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

# Specify the interview domain (helps the LLM correct technical terms)
uv run audio-interview-analyze path/to/interview.mp4 --domain "后端 / Go / Kubernetes"

# (Advanced) provide a custom terminology dictionary for known ASR errors
uv run audio-interview-analyze path/to/interview.mp4 --terms-file my-terms.json
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

The pipeline has 8 stages:

```
extract → diarize → transcribe → clean_transcript (multi-stage LLM) →
build_qa_pairs → analyze_pair (LLM) → aggregate (LLM) → render_markdown
```

The `clean_transcript` stage runs a multi-stage proofreading pipeline:

1. **Context-aware chunking** — splits at long pauses while preserving
   speaker turns and Q+A continuity.
2. **LLM correction** — corrects each chunk, with previous corrections as
   reference for consistency.
3. **LLM review** — validates original-vs-corrected and can reverse
   over-corrections.
4. **Global final validation** — reviews the full correction log for
   inconsistencies and supports revising earlier corrections.

Use `--domain` to give the LLM context (default:
"软件工程 / 前端开发 / AI Agent").  If you have a list of known systematic
ASR errors, you can optionally provide a terminology dictionary with
`--terms-file`.

All intermediate artifacts (extracted WAV, diarization, transcript,
cleaned transcript, per-pair JSON) are cached under
`./.interview-cache/<content-hash>/` so re-runs are cheap.

## Custom terminology file format

If you want to give the model extra hints for specific ASR errors, create
a JSON file like `my-terms.json`:

```json
{
  "terms": [
    {"correct": "Vue", "aliases": ["BUD"], "category": "前端框架"},
    {"correct": "前端", "aliases": ["前段"], "category": "岗位"},
    {"correct": "Fiber", "aliases": ["Fibre"], "category": "React"}
  ]
}
```

Aliases are matched both as exact substrings and by pinyin similarity,
so the LLM can use them as strong hints while still making the final
context-aware decision.

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