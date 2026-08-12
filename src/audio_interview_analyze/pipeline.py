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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from audio_interview_analyze.analysis.analyze import analyze_conversation, analyze_pair
from audio_interview_analyze.analysis.clean_transcript import clean_transcript
from audio_interview_analyze.analysis.qa_pairs import (
    build_qa_pairs,
    build_qa_pairs_with_llm,
    label_speakers,
)
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
from audio_interview_analyze.report.model import FinalReport, PairAnalysis, QAPair, Transcript
from audio_interview_analyze.report.render import render_markdown


@dataclass
class PipelineConfig:
    input_path: Path
    output_path: Path
    whisper_model: str = "large-v3"
    whisper_backend: str = "faster-whisper"
    asr_prompt: str = ""
    reuse_cache: bool = False
    candidate_background: str = ""
    domain: str = "软件工程 / 前端开发 / AI Agent"
    terms_path: Path | None = None
    enable_study_guide: bool = True
    study_guide_path: Path | None = None
    # Number of independent LLM calls (per-pair analysis, study-guide
    # batches) to run in parallel. Prompts are identical to sequential
    # execution, so this does not change output quality.
    llm_concurrency: int = 4
    # Number of transcript-cleaning chunks to process in parallel. Each
    # chunk sees a snapshot of prior corrections rather than the full
    # history; set to 1 for the original strictly-sequential behaviour.
    clean_concurrency: int = 4


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

        # Stages 2+3: diarize and transcribe are independent — run them
        # concurrently (diarization typically on MPS GPU, Whisper on CPU,
        # so they barely contend). Speaker assignment happens afterwards.
        diar_task = progress.add_task("[cyan]Diarizing speakers...", total=1)
        tr_task = progress.add_task("[cyan]Transcribing...", total=1)

        transcript_cache = (
            read_json("transcript.json", hash_key=hash_key) if config.reuse_cache else None
        )

        def _run_diarization() -> list[tuple[float, float, str]]:
            diar_cache = read_json("diarization.json", hash_key=hash_key)
            if config.reuse_cache and diar_cache is not None:
                return [(float(s[0]), float(s[1]), str(s[2])) for s in diar_cache]
            d = diarize(wav_path, hf_token=hf_token)
            write_json("diarization.json", d, hash_key=hash_key)
            return d

        def _run_whisper() -> list[dict]:
            return transcribe(
                str(wav_path),
                model_size=config.whisper_model,
                initial_prompt=config.asr_prompt,
                backend=config.whisper_backend,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            diar_future = pool.submit(_run_diarization)
            whisper_future = pool.submit(_run_whisper) if transcript_cache is None else None

            diarization = diar_future.result()
            progress.update(diar_task, completed=1)

            if transcript_cache is not None:
                transcript = Transcript.model_validate(transcript_cache)
            else:
                assert whisper_future is not None
                whisper_segs = whisper_future.result()
                transcript = assign_speakers(whisper_segs, diarization)
                write_json("transcript.json", transcript.model_dump(), hash_key=hash_key)
            progress.update(tr_task, completed=1)

        # Stage 3.5: clean transcript (LLM). Fixes ASR-misheard technical
        # terms before Q+A extraction. Cached separately so the original
        # transcript is preserved for diff/review.
        task = progress.add_task("[cyan]Cleaning transcript...", total=None)
        cleaned_cache = read_json("cleaned_transcript.json", hash_key=hash_key)
        if config.reuse_cache and cleaned_cache is not None:
            transcript = Transcript.model_validate(cleaned_cache)
            progress.update(task, total=1, completed=1)
        else:
            transcript = clean_transcript(
                client,
                transcript,
                domain=config.domain,
                terminology=config.terms_path,
                concurrency=config.clean_concurrency,
                progress_callback=lambda done, total: progress.update(
                    task, completed=done, total=total
                ),
            )
            write_json(
                "cleaned_transcript.json", transcript.model_dump(), hash_key=hash_key
            )

        # Stage 4: build Q+A pairs (rule-based then LLM refinement)
        task = progress.add_task("[cyan]Building Q&A pairs...", total=1)
        labeled = label_speakers(transcript)
        rule_pairs = build_qa_pairs(labeled)
        pairs = build_qa_pairs_with_llm(client, labeled) or rule_pairs
        # Split into questions (LLM-evaluated) and conversations (no LLM).
        # Conversations still appear in the report but in a separate section.
        analyzable = [p for p in pairs if not p.is_conversation and p.question and len(p.answer.strip()) >= 20]
        conversation_pairs = [p for p in pairs if p.is_conversation]
        progress.update(task, completed=1)
        if not analyzable:
            raise RuntimeError(
                "No analyzable Q+A pairs detected. "
                "Check that the audio has two distinct speakers and clear questions."
            )

        # Stage 5: per-pair analysis. Independent LLM calls are executed
        # with a bounded thread pool; prompts are identical to sequential
        # execution, so results are the same. Per-pair results are cached
        # at pairs/<n>.json so re-runs with --reuse-cache skip the calls.
        task = progress.add_task(
            "[cyan]Analyzing pairs...", total=len(analyzable)
        )
        pair_analyses: list[PairAnalysis] = []
        per_pair_ran = False  # tracks whether any pair was newly analyzed
        # Conversation pairs are emitted as-is (no LLM call) and slotted
        # into the report at the right position by pair_index.
        for p in conversation_pairs:
            pair_analyses.append(analyze_conversation(p))

        results: dict[int, PairAnalysis] = {}  # position in `analyzable` -> analysis
        to_analyze: list[tuple[int, QAPair]] = []
        for i, p in enumerate(analyzable):
            pair_cache_path = artifact_path(
                f"pairs/{p.pair_index}.json", hash_key=hash_key
            )
            cached_pair = (
                read_json(f"pairs/{p.pair_index}.json", hash_key=hash_key)
                if config.reuse_cache and pair_cache_path.exists()
                else None
            )
            if cached_pair is not None:
                results[i] = PairAnalysis.model_validate(cached_pair)
                progress.update(task, advance=1)
            else:
                to_analyze.append((i, p))

        if to_analyze:
            workers = max(1, min(config.llm_concurrency, len(to_analyze)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(analyze_pair, client, p, cache_dir=pair_cache_dir): (i, p)
                    for i, p in to_analyze
                }
                for fut in as_completed(futures):
                    i, p = futures[fut]
                    pa = fut.result()
                    write_json(
                        f"pairs/{p.pair_index}.json", pa.model_dump(), hash_key=hash_key
                    )
                    results[i] = pa
                    per_pair_ran = True
                    progress.update(task, advance=1)

        for i in sorted(results):
            pair_analyses.append(results[i])

        # Stage 6: aggregate. Cached at final_report.json, but ONLY reused
        # if the per-pair stage did not run in this invocation. Otherwise
        # a re-run with new cleaned transcript + new per-pair analyses
        # would silently produce a stale aggregated report.
        task = progress.add_task("[cyan]Aggregating report...", total=1)
        final_report_path = artifact_path("final_report.json", hash_key=hash_key)
        cached_report = (
            read_json("final_report.json", hash_key=hash_key)
            if config.reuse_cache and not per_pair_ran and final_report_path.exists()
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

    # Stage 8: study guide (optional, default-on). Generates a per-KP
    # learning guide in a separate Markdown file. Cached separately so
    # re-runs can iterate on the study guide without re-aggregating.
    if config.enable_study_guide:
        from audio_interview_analyze.study.generate import generate_study_guide
        from audio_interview_analyze.study.render import render_study_guide

        study_out = config.study_guide_path or output_path.with_name(
            output_path.stem.replace("report", "study-guide", 1) + ".md"
            if output_path.stem.startswith("report")
            else "study-guide.md"
        )
        # Default to "study-guide.md" next to the report
        if study_out.name == output_path.name:
            study_out = output_path.parent / "study-guide.md"

        study_cache_path = artifact_path("study_guide.json", hash_key=hash_key)
        cached_guide = (
            read_json("study_guide.json", hash_key=hash_key)
            if config.reuse_cache and study_cache_path.exists()
            else None
        )
        if cached_guide is not None:
            guide = cached_guide
        else:
            task = progress.add_task("[cyan]Generating study guide...", total=1)
            guide = generate_study_guide(client, report, concurrency=config.llm_concurrency)
            write_json("study_guide.json", guide, hash_key=hash_key)
            progress.update(task, completed=1)
        study_md = render_study_guide(guide)
        study_out.parent.mkdir(parents=True, exist_ok=True)
        study_out.write_text(study_md, encoding="utf-8")

    return md


def copy_default_cache_to_cwd(cache_root: Path, target: Path) -> None:
    """Helper used by tests to move the cache out of the project root."""
    if cache_root.exists():
        shutil.move(str(cache_root), str(target))


__all__ = ["PipelineConfig", "run_pipeline"]
