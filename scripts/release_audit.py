#!/usr/bin/env python3
"""Fail closed on obvious violations of the bounded public-release surface."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024

REQUIRED_PATHS = {
    "AGENTS.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "REAL_CONCEPT.md",
    "RELEASE_MANIFEST.json",
    "RELEASE_STATUS.md",
    "SECURITY.md",
    "THIRD_PARTY.md",
    "docs/ARCHITECTURE.md",
    "docs/EVIDENCE.md",
    "docs/PROVENANCE.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RESEARCH_CONTRACT.md",
    "examples/structural_contact_demo.py",
    "train_real_v1_3.py",
}

BLOCKED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".ipynb",
    ".key",
    ".p12",
    ".pem",
    ".pfx",
    ".pt",
    ".pth",
    ".safetensors",
}
BLOCKED_DIR_NAMES = {"checkpoints", "data", "outputs", "results", "wandb"}
BLOCKED_BASENAMES = {".env", "credentials", "credentials.json", "secrets", "secrets.json"}

TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".gitignore",
    ".gitattributes",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

CONTENT_PATTERNS = {
    "unix_home_path": re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/"),
    "windows_user_path": re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\s]+", re.IGNORECASE),
    "colab_drive_path": re.compile(re.escape("/content/drive/" + "MyDrive/"), re.IGNORECASE),
    "private_source_username": re.compile("kiff" + "klipp", re.IGNORECASE),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github_token": re.compile(r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}"),
    "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "openai_style_key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "private_key_header": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _iter_public_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.is_file() or path.is_symlink():
            files.append(path)
    return sorted(files, key=_relative)


def audit() -> list[str]:
    failures: list[str] = []
    public_files = _iter_public_files()
    present = {_relative(path) for path in public_files}

    for required in sorted(REQUIRED_PATHS - present):
        failures.append(f"missing required path: {required}")

    for path in public_files:
        rel = _relative(path)
        if path.is_symlink():
            failures.append(f"symlink is not allowed in public seed: {rel}")
            continue
        if path.name in BLOCKED_BASENAMES or path.suffix.lower() in BLOCKED_SUFFIXES:
            failures.append(f"blocked artifact type: {rel}")
        if any(part in BLOCKED_DIR_NAMES for part in Path(rel).parts):
            failures.append(f"blocked artifact directory: {rel}")
        size = path.stat().st_size
        if size > MAX_PUBLIC_FILE_BYTES:
            failures.append(f"file exceeds 5 MiB public limit: {rel} ({size} bytes)")

        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            "LICENSE",
            "NOTICE",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"expected text file is not UTF-8: {rel}")
            continue
        for label, pattern in CONTENT_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label} detected in {rel}")

        if path.suffix.lower() == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                failures.append(f"invalid JSON in {rel}: {exc}")

        if path.suffix.lower() == ".md":
            for raw_target in MARKDOWN_LINK.findall(text):
                target = raw_target.strip().split("#", 1)[0]
                if not target or "://" in target or target.startswith(("mailto:", "#")):
                    continue
                resolved = (path.parent / target).resolve()
                try:
                    resolved.relative_to(ROOT)
                except ValueError:
                    failures.append(f"markdown link escapes repository in {rel}: {raw_target}")
                    continue
                if not resolved.exists():
                    failures.append(f"broken local markdown link in {rel}: {raw_target}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output")
    args = parser.parse_args()

    failures = audit()
    payload = {"passed": not failures, "failure_count": len(failures), "failures": failures}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif failures:
        print("PUBLIC RELEASE AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
    else:
        print("PUBLIC RELEASE AUDIT: PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
