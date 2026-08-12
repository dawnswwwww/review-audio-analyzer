"""Speaker diarization using pyannote-audio.

Returns a list of ``(start, end, speaker_label)`` tuples. Thin wrapper:
exception translation only.
"""

from __future__ import annotations

from pathlib import Path


def _pick_device():
    """Return ``torch.device("mps")`` when Apple-Silicon GPU is usable, else None."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return None


def _run_pipeline(pipeline, audio_path, *, catch: bool = True):
    """Execute the pyannote generator and return the ``DiarizeOutput``.

    With ``catch=True``, any inference error returns ``None`` so the caller
    can retry on another device; otherwise errors propagate.
    """
    try:
        output = pipeline(str(audio_path))
        try:
            return next(output)
        except StopIteration as e:
            return e.value
    except Exception:
        if catch:
            return None
        raise


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
            token=hf_token,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load pyannote diarization model. "
            "Accept the model terms on HuggingFace and set HF_TOKEN. See README. "
            f"Underlying error: {type(e).__name__}: {e}"
        ) from e

    # Prefer Apple-Silicon GPU (MPS) when available — diarization is several
    # times faster than on CPU. Fall back to CPU if MPS init or inference
    # fails for any reason.
    device = _pick_device()
    if device is not None:
        try:
            pipeline.to(device)
        except Exception:
            device = None

    # pyannote 3.x: ``pipeline(audio)`` is a generator that uses the PEP 380
    # ``return value`` idiom — the ``DiarizeOutput`` is delivered as the
    # ``value`` of a ``StopIteration`` raised at the end of the generator.
    # (Older versions returned the Annotation directly.)
    result = _run_pipeline(pipeline, audio_path, catch=device is not None)
    if result is None:
        # GPU inference failed — retry once on CPU.
        import torch

        pipeline.to(torch.device("cpu"))
        result = _run_pipeline(pipeline, audio_path, catch=False)
    annotation = result.speaker_diarization
    segments: list[tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append((float(turn.start), float(turn.end), str(speaker)))
    return segments