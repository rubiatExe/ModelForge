"""Deterministic JSON artifact helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

ArtifactModel = TypeVar("ArtifactModel", bound=BaseModel)


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def artifact_json(value: Any) -> str:
    """Canonical, finite JSON with a trailing newline for clean diffs."""

    return (
        json.dumps(
            _json_value(value),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def write_json_artifact(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as sink:
            sink.write(artifact_json(value))
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json_artifact(path: Path, model_type: type[ArtifactModel]) -> ArtifactModel:
    """Load one strict Pydantic artifact with path-aware errors."""

    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"), strict=True)
    except ValueError as exc:
        raise ValueError(f"invalid {model_type.__name__} artifact at {path}: {exc}") from exc
