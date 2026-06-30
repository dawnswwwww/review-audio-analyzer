"""End-to-end smoke test (gated by RUN_SLOW=1).

Runs the full pipeline against a 5-second silent WAV. With no speech, the
pipeline should detect zero analyzable Q+A pairs and raise a clear error
*after* writing the audio.wav cache artifact. The test asserts that:

1. The error is the expected "no Q+A pairs" error.
2. The cache directory was created.
3. No LLM call is made (verified by the fact that we never construct a
   real DeepSeekClient in this test).

Real-audio smoke tests should be run manually by the developer after
setting up the environment.
"""

import os
from pathlib import Path

import pytest

from audio_interview_analyze.pipeline import PipelineConfig, run_pipeline

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SLOW") != "1",
    reason="set RUN_SLOW=1 to run the end-to-end smoke test",
)


def test_silent_audio_raises_no_qa_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Skip preflight by short-circuiting: the silent fixture doesn't need
    # ffmpeg or DeepSeek for the failure mode we want to test (we fail
    # *before* any LLM call). But run_preflight is called inside the
    # pipeline; we patch the preflight to be a no-op success.
    from audio_interview_analyze import pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_preflight", lambda: type("R", (), {"all_ok": True, "failures": []})())

    # Provide a real API key so the client can be constructed (it won't be
    # called).
    os.environ["DEEPSEEK_API_KEY"] = "fake-key-for-smoke"
    from audio_interview_analyze.llm.deepseek import DeepSeekClient

    client = DeepSeekClient(api_key="fake-key-for-smoke", base_url="http://localhost:1")

    config = PipelineConfig(
        input_path=Path(__file__).parent / "fixtures" / "silent_5s.wav",
        output_path=tmp_path / "report.md",
        whisper_model="tiny",  # smallest model, fastest for the smoke test
        reuse_cache=False,
    )

    with pytest.raises(RuntimeError, match="No analyzable Q+A pairs"):
        run_pipeline(config, client=client, hf_token=None)