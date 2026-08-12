"""Transcription with speaker alignment.

Run ``faster-whisper`` on the audio, then intersect the word-level
timestamps with the diarization segments to produce ordered
``Utterance(speaker, text, start, end)`` objects.
"""

from __future__ import annotations

from faster_whisper import WhisperModel  # type: ignore[import-untyped]

from audio_interview_analyze.report.model import Transcript, Utterance

DEFAULT_INITIAL_PROMPT = (
    "前端 Vue React Fiber diff算法 事件循环 SSR"
)


def transcribe(
    audio_path: str | bytes,
    *,
    model_size: str = "large-v3",
    device: str = "auto",
    compute_type: str = "default",
    initial_prompt: str | None = DEFAULT_INITIAL_PROMPT,
    backend: str = "faster-whisper",
) -> list[dict]:
    """Run Whisper and return a list of ``{"start", "end", "text", "words"}``.

    Each ``words`` entry is ``{"start", "end", "word"}``. The model is
    loaded once per process; callers should reuse the returned list rather
    than re-running on the same file.

    Args:
        audio_path: Path to the audio file.
        model_size: Whisper model size.
        device: Device to run on.
        compute_type: Computation type for the model (faster-whisper only).
        initial_prompt: Optional prompt to bias Whisper toward the expected
            domain vocabulary. Pass an empty string to disable.
        backend: ``"faster-whisper"`` (default; CPU-optimized ctranslate2) or
            ``"openai-whisper"`` (torch-based; runs on CUDA GPUs, much faster
            on NVIDIA hardware such as the DGX Spark's GB10).
    """
    if backend == "openai-whisper":
        return _transcribe_openai(
            audio_path,
            model_size=model_size,
            device=device,
            initial_prompt=initial_prompt,
        )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    kwargs: dict[str, object] = {
        "vad_filter": True,
        "word_timestamps": True,
    }
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    segments, _info = model.transcribe(audio_path, **kwargs)
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


def _transcribe_openai(
    audio_path: str | bytes,
    *,
    model_size: str,
    device: str,
    initial_prompt: str | None,
) -> list[dict]:
    """Transcribe with the torch-based ``openai-whisper`` backend.

    faster-whisper is built on ctranslate2, which ships CPU-only wheels on
    linux-aarch64, so it cannot use an NVIDIA GPU there. ``openai-whisper``
    runs on PyTorch and supports CUDA, which turns a multi-hour CPU
    transcription into minutes on a GB10/Blackwell GPU. The output shape is
    identical to the faster-whisper path so ``assign_speakers`` consumes it
    unchanged.

    openai-whisper is more hallucination-prone than faster-whisper (which
    runs a VAD filter by default), especially on long silent stretches and
    with automatic language detection. We therefore pin the language to
    Chinese (these interviews are in Chinese), pass a domain anchor prompt,
    disable text conditioning, and raise the no-speech threshold so silent
    segments are dropped instead of hallucinated.
    """
    import torch  # noqa: PLC0415
    import whisper  # type: ignore[import-untyped]  # noqa: PLC0415

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="zh" if not initial_prompt else None,
        initial_prompt=initial_prompt or DEFAULT_INITIAL_PROMPT,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        fp16=(device == "cuda"),
    )
    out: list[dict] = []
    for seg in result["segments"]:
        out.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip(),
                "words": [
                    {"start": float(w["start"]), "end": float(w["end"]), "word": w["word"]}
                    for w in (seg.get("words") or [])
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
