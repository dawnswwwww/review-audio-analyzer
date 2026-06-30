"""Content-hash based artifact cache.

All intermediate artifacts (extracted audio, diarization, transcript,
per-pair analyses) are stored under ``./.interview-cache/<content-hash>/``
so the analysis can be re-run cheaply.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_DIRNAME = ".interview-cache"


def content_hash(path: Path | str) -> str:
    """Return a SHA-256 hex digest of the file at ``path``."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_root(base: Path | None = None) -> Path:
    """Return the cache root directory, creating it if necessary."""
    base = base or Path.cwd()
    root = base / CACHE_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def artifact_path(relative: str, *, hash_key: str, base: Path | None = None) -> Path:
    """Return the absolute path for a cached artifact under ``<hash>/<relative>``."""
    return cache_root(base) / hash_key / relative


def write_json(relative: str, data: Any, *, hash_key: str, base: Path | None = None) -> Path:
    """Write ``data`` as JSON to the cache, creating parent dirs as needed.

    Returns the path written.
    """
    path = artifact_path(relative, hash_key=hash_key, base=base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_json(relative: str, *, hash_key: str, base: Path | None = None) -> Any | None:
    """Read a JSON artifact from the cache, or return ``None`` if missing."""
    path = artifact_path(relative, hash_key=hash_key, base=base)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))