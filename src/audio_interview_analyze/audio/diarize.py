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

    # pyannote 3.x: ``pipeline(audio)`` is a generator that yields a single
    # ``DiarizeOutput`` whose ``speaker_diarization`` is the actual
    # ``Annotation``. (Older versions returned the Annotation directly.)
    output = pipeline(str(audio_path))
    result = next(output)
    annotation = result.speaker_diarization
    segments: list[tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append((float(turn.start), float(turn.end), str(speaker)))
    return segments