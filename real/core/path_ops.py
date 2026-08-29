"""Small path helpers for portable display and local repo path recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_repo_root(repo_root: Optional[Path]) -> Path:
    return Path(repo_root).resolve() if repo_root is not None else _default_repo_root()


def _path_str(path: Path) -> str:
    return path.as_posix()


def repo_relative_path(path: Path, repo_root: Optional[Path] = None) -> Path | None:
    """Return a path relative to the repo root when the path lives under it."""
    root = _normalize_repo_root(repo_root)
    p = Path(path)
    candidates = [p] if p.is_absolute() else [root / p, p]
    for candidate in candidates:
        try:
            rel = candidate.resolve().relative_to(root)
        except ValueError:
            continue
        return rel
    return None


def portable_path_str(path: Path, repo_root: Optional[Path] = None) -> str:
    """Prefer a repo-relative display path when possible."""
    rel = repo_relative_path(path, repo_root=repo_root)
    if rel is not None:
        return _path_str(rel)
    return _path_str(Path(path))


def basename_or_relpath(path: Path, repo_root: Optional[Path] = None) -> str:
    """Prefer a repo-relative path, otherwise fall back to the basename."""
    rel = repo_relative_path(path, repo_root=repo_root)
    if rel is not None:
        return _path_str(rel)
    p = Path(path)
    return p.name or _path_str(p)


def maybe_localize_outputs_path(path: Path, repo_root: Optional[Path] = None) -> Path:
    """Rewrite foreign `/outputs/...` paths to the local repo when possible."""
    root = _normalize_repo_root(repo_root)
    p = Path(path)
    if p.exists():
        return p
    if not p.is_absolute():
        candidate = root / p
        if candidate.exists():
            return candidate
    marker = "/outputs/"
    normalized = str(p).replace("\\", "/")
    if marker in normalized:
        rel = normalized.split(marker, 1)[1]
        candidate = root / "outputs" / Path(rel)
        if candidate.exists():
            return candidate
    return p
