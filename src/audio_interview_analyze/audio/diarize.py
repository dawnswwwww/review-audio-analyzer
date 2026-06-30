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
            token=hf_token,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load pyannote diarization model. "
            "Accept the model terms on HuggingFace and set HF_TOKEN. See README. "
            f"Underlying error: {type(e).__name__}: {e}"
        ) from e

    diarization = pipeline(str(audio_path))
    segments: list[tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((float(turn.start), float(turn.end), str(speaker)))
    return segments