#!/usr/bin/env python3
"""Write or verify the deterministic REAL public-release file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "RELEASE_MANIFEST.json"
SOURCE_COMMIT = "1847a309a0c20b5b88b3186516ff9ea4efa714a5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST_PATH:
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        rows.append({"path": rel, "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    rows.sort(key=lambda row: row["path"])
    return {
        "schema_version": "real.public-release-manifest.v1",
        "release_version": "0.1.0",
        "release_date": "2026-08-29",
        "source_commit": SOURCE_COMMIT,
        "file_count": len(rows),
        "files": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Write RELEASE_MANIFEST.json")
    mode.add_argument("--check", action="store_true", help="Verify RELEASE_MANIFEST.json")
    args = parser.parse_args()

    current = build_manifest()
    if args.write:
        MANIFEST_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST_PATH.name} with {current['file_count']} files")
        return

    if not MANIFEST_PATH.exists():
        print(f"missing {MANIFEST_PATH.name}")
        sys.exit(1)
    recorded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if recorded != current:
        print(f"{MANIFEST_PATH.name} is stale; run with --write")
        sys.exit(1)
    print(f"{MANIFEST_PATH.name}: PASS ({current['file_count']} files)")


if __name__ == "__main__":
    main()
