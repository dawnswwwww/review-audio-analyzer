"""Tests for the pre-flight environment check."""

from audio_interview_analyze.preflight import (
    PreflightResult,
    check_api_key,
    check_ffmpeg,
    check_hf_token,
    run_preflight,
)


def test_check_ffmpeg_passes_when_on_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")
    assert check_ffmpeg() is True


def test_check_ffmpeg_fails_when_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert check_ffmpeg() is False


def test_check_api_key_passes_when_set(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    assert check_api_key() is True


def test_check_api_key_fails_when_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert check_api_key() is False


def test_check_hf_token_passes_when_set(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    assert check_hf_token() is True


def test_check_hf_token_passes_when_hugging_face_token_set(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test-token")
    assert check_hf_token() is True


def test_check_hf_token_fails_when_missing(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    assert check_hf_token() is False


def test_run_preflight_returns_structured_result(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("HF_TOKEN", "t")
    result = run_preflight()
    assert isinstance(result, PreflightResult)
    assert result.ffmpeg is True
    assert result.deepseek_api_key is True
    assert result.hf_token is True
    assert result.all_ok is True
    assert result.failures == []


def test_run_preflight_collects_all_failures(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    result = run_preflight()
    assert result.all_ok is False
    assert len(result.failures) == 3
