"""Pre-flight environment check.

Verifies that the three external dependencies are present:
1. ``ffmpeg`` on PATH.
2. ``DEEPSEEK_API_KEY`` env var set.
3. HuggingFace token (``HF_TOKEN`` or ``HUGGINGFACE_TOKEN``) set, required
   for the pyannote-audio model.

Returns a structured ``PreflightResult`` so the CLI can render a clear
checklist and link to the relevant README sections.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field


@dataclass
class PreflightResult:
    ffmpeg: bool
    deepseek_api_key: bool
    hf_token: bool
    failures: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return not self.failures


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def check_api_key() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY", "").strip())


def check_hf_token() -> bool:
    return bool(os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACE_TOKEN", "").strip())


def run_preflight() -> PreflightResult:
    ffmpeg_ok = check_ffmpeg()
    api_key_ok = check_api_key()
    hf_ok = check_hf_token()
    failures: list[str] = []
    if not ffmpeg_ok:
        failures.append("ffmpeg not found on PATH (install: `brew install ffmpeg`)")
    if not api_key_ok:
        failures.append("DEEPSEEK_API_KEY not set (see README §Configuration)")
    if not hf_ok:
        failures.append("HuggingFace token not set (set HF_TOKEN; required for pyannote-audio)")
    return PreflightResult(
        ffmpeg=ffmpeg_ok,
        deepseek_api_key=api_key_ok,
        hf_token=hf_ok,
        failures=failures,
    )
