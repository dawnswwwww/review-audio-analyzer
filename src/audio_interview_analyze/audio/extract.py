"""Extract a 16 kHz mono WAV from any input audio or video file.

Delegates to ``ffmpeg`` via ``pydub``. Caller is expected to have ``ffmpeg``
on PATH (verified by the preflight check).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def extract_wav(
    input_path: Path | str,
    output_path: Path | str,
    *,
    apply_preprocessing: bool = True,
) -> Path:
    """Read ``input_path`` and write a 16 kHz mono WAV to ``output_path``.

    Args:
        input_path: Source audio or video file.
        output_path: Destination WAV file (16 kHz, mono).
        apply_preprocessing: If True, apply a light ffmpeg filter chain
            (highpass, lowpass, loudness normalization, noise reduction)
            before writing the WAV. This usually improves ASR accuracy
            on noisy interview recordings.

    Returns:
        The output path.

    Raises:
        FileNotFoundError: If the input is missing.
        RuntimeError: If ``ffmpeg`` is not on PATH.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`).")
    in_path = Path(input_path)
    out_path = Path(output_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if apply_preprocessing:
        # Conservative chain for speech ASR:
        # - highpass=f=80: remove low-frequency rumble
        # - lowpass=f=8000: keep speech bandwidth, discard high-freq noise
        # - loudnorm: normalize perceived loudness (EBU R128)
        # - afftdn=nf=-25: light spectral noise reduction
        filter_chain = (
            "highpass=f=80,"
            "lowpass=f=8000,"
            "loudnorm=I=-16:TP=-1.5:LRA=11,"
            "afftdn=nf=-25"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(in_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            filter_chain,
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(in_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out_path),
        ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract WAV: {result.stderr.strip() or 'unknown error'}"
        )

    return out_path
