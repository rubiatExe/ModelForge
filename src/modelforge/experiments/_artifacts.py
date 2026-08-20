"""Small safety helpers shared by experiment command-line runners."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from modelforge.datasets import atomic_write_bytes


def require_available_outputs(paths: Iterable[Path], *, overwrite: bool) -> None:
    """Validate all destinations before an experiment performs expensive work."""

    destinations = tuple(Path(path) for path in paths)
    if len(destinations) != len(set(destinations)):
        raise ValueError("experiment output paths must be distinct")
    if overwrite:
        return
    existing = [path for path in destinations if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite experiment artifact: {rendered}")


def canonical_sha256(value: Any) -> str:
    """Hash one JSON-compatible value without relying on mapping insertion order."""

    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_svg_artifact(path: Path, svg: str) -> None:
    """Atomically persist a standalone, script-free SVG document."""

    if "<script" in svg.casefold() or "javascript:" in svg.casefold():
        raise ValueError("unsafe active content is forbidden in SVG artifacts")
    atomic_write_bytes(path, (svg.rstrip() + "\n").encode("utf-8"))
