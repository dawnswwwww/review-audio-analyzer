"""Shared test fixtures."""

import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Return a temporary cache directory for the test."""
    cache = tmp_path / ".interview-cache"
    cache.mkdir()
    return cache
