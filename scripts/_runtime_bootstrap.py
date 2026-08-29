"""Thin runtime import helpers for CLI entrypoints."""

from __future__ import annotations

import importlib


def load_module(module_name: str, *, context: str):
    """Import a runtime module lazily with a CLI-friendly error."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or module_name
        raise SystemExit(
            f"{context} requires runtime dependency '{missing}'. "
            "Install the repo runtime dependencies and rerun the command."
        ) from exc
