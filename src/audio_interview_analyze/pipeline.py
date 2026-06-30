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
from audio_interview_analyze.analysis.clean_transcript import clean_transcript
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
    domain: str = "软件工程 / 前端开发 / AI Agent"


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

        # Stage 3.5: clean transcript (LLM). Fixes ASR-misheard technical
        # terms before Q+A extraction. Cached separately so the original
        # transcript is preserved for diff/review.
        task = progress.add_task("[cyan]Cleaning transcript...", total=1)
        cleaned_cache = read_json("cleaned_transcript.json", hash_key=hash_key)
        if config.reuse_cache and cleaned_cache is not None:
            transcript = Transcript.model_validate(cleaned_cache)
        else:
            transcript = clean_transcript(client, transcript, domain=config.domain)
            write_json(
                "cleaned_transcript.json", transcript.model_dump(), hash_key=hash_key
            )
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
            # Per-pair LLM analysis cached at pairs/<n>.json so re-runs
            # with --reuse-cache skip the per-pair DeepSeek call.
            pair_cache_path = artifact_path(
                f"pairs/{p.pair_index}.json", hash_key=hash_key
            )
            cached_pair = (
                read_json(f"pairs/{p.pair_index}.json", hash_key=hash_key)
                if config.reuse_cache and pair_cache_path.exists()
                else None
            )
            if cached_pair is not None:
                pair_analyses.append(PairAnalysis.model_validate(cached_pair))
            else:
                pa = analyze_pair(client, p, cache_dir=pair_cache_dir)
                write_json(
                    f"pairs/{p.pair_index}.json", pa.model_dump(), hash_key=hash_key
                )
                pair_analyses.append(pa)
            progress.update(task, advance=1)

        # Stage 6: aggregate. Cached at final_report.json so re-runs skip
        # the aggregator DeepSeek call when the per-pair inputs are
        # unchanged.
        task = progress.add_task("[cyan]Aggregating report...", total=1)
        final_report_path = artifact_path("final_report.json", hash_key=hash_key)
        cached_report = (
            read_json("final_report.json", hash_key=hash_key)
            if config.reuse_cache and final_report_path.exists()
            else None
        )
        if cached_report is not None:
            report = FinalReport.model_validate(cached_report)
        else:
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
