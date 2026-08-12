# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Status

This repository is in its initial stage. The `main.py` entry point is a stub, `README.md` is empty, and `pyproject.toml` declares no dependencies yet. The project name (`audio-interview-analyze`) suggests the eventual goal is audio interview analysis, but no source structure, modules, or tests exist yet — do not assume an architecture that isn't there.

## Package Management & Tooling: uv

This project uses [uv](https://docs.astral.sh/uv/) for Python package management and script execution. The Python interpreter is pinned to **3.12** via `.python-version`, and `pyproject.toml` requires `>=3.12`.

### Common Commands

All commands should be run via `uv` from the project root. uv implicitly uses the project's virtual environment (`.venv/`, gitignored), so do not activate it manually.

| Purpose | Command |
|---|---|
| Run the entry point | `uv run main.py` |
| Run a module/script | `uv run python <path>` |
| Add a runtime dependency | `uv add <package>` |
| Add a dev dependency | `uv add --dev <package>` |
| Remove a dependency | `uv remove <package>` |
| Sync the lockfile to the venv | `uv sync` |
| Update a specific package | `uv lock --upgrade-package <package>` |
| Pin/change Python version | `uv python pin 3.12` (writes `.python-version`) |
| Run a tool via uvx (one-off) | `uvx <tool>` (e.g. `uvx ruff check .`) |
| Format / lint (when added) | `uv run ruff format .` / `uv run ruff check .` |
| Run tests (when added) | `uv run pytest` (single test: `uv run pytest tests/test_x.py::test_name -v`) |

### Key Conventions

- **Do not use `pip` or `python` directly** for installing packages or running scripts — always go through `uv` so the project lockfile (`uv.lock`, which uv will create on first `uv add`) and `.venv` stay in sync.
- **Dependency declarations belong in `pyproject.toml`**, not in ad-hoc install commands.
- The entry point is `main.py` at the repo root (no `src/` layout yet). Reorganize the package structure inside `pyproject.toml` if/when a `src/` directory is introduced.

## Layout

```
.
├── main.py            # Entry point (currently a stub)
├── pyproject.toml     # Project metadata & dependencies
├── .python-version    # Pinned interpreter (3.12)
├── .gitignore         # Ignores __pycache__, .venv, build artifacts
└── README.md          # Empty
```

There is no `src/`, no `tests/`, and no tooling config (ruff, mypy, pytest) yet. When adding them, prefer declaring them as dev dependencies in `pyproject.toml` and running them through `uv run`.
