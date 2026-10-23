"""JSON caches shared by concurrent jobs: saves merge with what is on disk and replace the file atomically, so two
processes writing the same cache neither lose each other's entries nor leave a half-written file."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        return {}


def merge_save(path: Path, entries: dict, indent: int | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {**load_json(path), **entries}
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(merged, f, indent=indent, ensure_ascii=False)
    os.replace(tmp, path)
    return merged
