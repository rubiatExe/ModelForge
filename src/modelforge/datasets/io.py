"""Bounded, line-aware JSONL and atomic artifact I/O."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from modelforge.datasets.fingerprint import canonical_json

ModelT = TypeVar("ModelT", bound=BaseModel)

DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_LINE_BYTES = 64 * 1024


class DatasetIOError(ValueError):
    pass


class JSONLLineError(DatasetIOError):
    def __init__(self, path: Path, line_number: int, message: str) -> None:
        self.path = path
        self.line_number = line_number
        self.message = message
        super().__init__(f"{path}:{line_number}: {message}")


class DigestMismatchError(DatasetIOError):
    pass


def _reject_nonstandard_number(value: str) -> None:
    raise ValueError(f"non-standard JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def _require_regular_file(path: Path, *, max_file_bytes: int) -> None:
    if path.is_symlink():
        raise DatasetIOError(f"refusing to read symlink: {path}")
    if not path.is_file():
        raise DatasetIOError(f"not a regular file: {path}")
    size = path.stat().st_size
    if size > max_file_bytes:
        raise DatasetIOError(f"file exceeds {max_file_bytes} bytes: {path}")


def read_jsonl(
    path: str | Path,
    model_type: type[ModelT],
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
) -> list[ModelT]:
    """Read strict UTF-8 JSONL, preserving the exact failing line number."""

    source = Path(path)
    _require_regular_file(source, max_file_bytes=max_file_bytes)
    records: list[ModelT] = []
    with source.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if len(raw_line) > max_line_bytes:
                raise JSONLLineError(source, line_number, f"line exceeds {max_line_bytes} bytes")
            if not raw_line.endswith(b"\n"):
                raise JSONLLineError(source, line_number, "line must end with a newline")
            payload = raw_line[:-1]
            if payload.endswith(b"\r"):
                payload = payload[:-1]
            if not payload.strip():
                raise JSONLLineError(source, line_number, "blank lines are forbidden")
            try:
                text = payload.decode("utf-8", errors="strict")
                value = json.loads(
                    text,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_nonstandard_number,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
                raise JSONLLineError(source, line_number, f"invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise JSONLLineError(source, line_number, "JSON value must be an object")
            try:
                records.append(model_type.model_validate(value))
            except ValidationError as exc:
                raise JSONLLineError(
                    source,
                    line_number,
                    f"schema validation failed: {exc.errors(include_url=False)}",
                ) from exc
    if not records:
        raise DatasetIOError(f"JSONL file is empty: {source}")
    return records


def model_jsonl_bytes(records: Iterable[BaseModel]) -> bytes:
    lines = [canonical_json(record.model_dump(mode="json")) for record in records]
    if not lines:
        raise DatasetIOError("refusing to write an empty JSONL artifact")
    return ("\n".join(lines) + "\n").encode("utf-8")


def model_json_bytes(record: BaseModel) -> bytes:
    return (canonical_json(record.model_dump(mode="json")) + "\n").encode("utf-8")


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Commit one file with fsync + same-directory atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_jsonl_atomic(path: str | Path, records: Iterable[BaseModel]) -> None:
    atomic_write_bytes(path, model_jsonl_bytes(records))


def write_json_atomic(path: str | Path, record: BaseModel) -> None:
    atomic_write_bytes(path, model_json_bytes(record))


def read_json_model(
    path: str | Path,
    model_type: type[ModelT],
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> ModelT:
    source = Path(path)
    _require_regular_file(source, max_file_bytes=max_file_bytes)
    try:
        payload = source.read_text(encoding="utf-8")
        value: Any = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise DatasetIOError(f"invalid JSON artifact {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetIOError(f"JSON artifact must contain an object: {source}")
    try:
        return model_type.model_validate(value)
    except ValidationError as exc:
        raise DatasetIOError(
            f"schema validation failed for {source}: {exc.errors(include_url=False)}"
        ) from exc


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    _require_regular_file(source, max_file_bytes=2**63 - 1)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise DigestMismatchError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
