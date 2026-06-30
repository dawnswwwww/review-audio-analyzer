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
    p.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Path to interview audio or video file (required unless --preflight)",
    )
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

    if args.input is None:
        parser.error("the following arguments are required: input")

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
