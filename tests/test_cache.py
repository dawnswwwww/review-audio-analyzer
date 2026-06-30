"""Tests for the content-hash cache."""

from audio_interview_analyze.cache import (
    artifact_path,
    cache_root,
    content_hash,
    read_json,
    write_json,
)


def test_content_hash_is_deterministic(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello world")
    b.write_bytes(b"hello world")
    assert content_hash(a) == content_hash(b)


def test_content_hash_changes_with_content(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"world")
    assert content_hash(a) != content_hash(b)


def test_cache_root_under_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cache_root() == tmp_path / ".interview-cache"


def test_artifact_path_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = "abc123"
    assert artifact_path("transcript.json", hash_key=h) == tmp_path / ".interview-cache" / h / "transcript.json"


def test_write_then_read_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = {"score": 80, "highlights": ["a", "b"]}
    write_json("transcript.json", data, hash_key="hash1")
    assert read_json("transcript.json", hash_key="hash1") == data


def test_write_creates_parent_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_json("nested/file.json", {"x": 1}, hash_key="hash2")
    assert (tmp_path / ".interview-cache" / "hash2" / "nested" / "file.json").exists()


def test_read_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert read_json("missing.json", hash_key="hash3") is None