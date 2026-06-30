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