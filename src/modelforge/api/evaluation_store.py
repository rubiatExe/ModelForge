"""Bounded in-memory job state for the development modular monolith."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from modelforge.api.contracts import EvaluationRunRequest, EvaluationRunStatus


class EvaluationStore:
    def __init__(self, *, max_runs: int = 1_000) -> None:
        self._runs: dict[UUID, EvaluationRunStatus] = {}
        self._order: list[UUID] = []
        self._max_runs = max_runs
        self._lock = threading.Lock()

    def create(self, request: EvaluationRunRequest) -> EvaluationRunStatus:
        run = EvaluationRunStatus(
            run_id=uuid4(), status="queued", model_id=request.model_id, split=request.split
        )
        with self._lock:
            self._runs[run.run_id] = run
            self._order.append(run.run_id)
            if len(self._order) > self._max_runs:
                oldest = self._order.pop(0)
                self._runs.pop(oldest, None)
        return run

    def get(self, run_id: UUID) -> EvaluationRunStatus | None:
        with self._lock:
            return self._runs.get(run_id)

    def mark_running(self, run_id: UUID) -> EvaluationRunStatus:
        return self._replace(run_id, status="running")

    def complete(self, run_id: UUID, report: dict[str, Any]) -> EvaluationRunStatus:
        return self._replace(
            run_id,
            status="complete",
            completed_at=datetime.now(UTC),
            report=report,
            error_code=None,
        )

    def fail(self, run_id: UUID, error_code: str) -> EvaluationRunStatus:
        return self._replace(
            run_id,
            status="failed",
            completed_at=datetime.now(UTC),
            report=None,
            error_code=error_code,
        )

    def _replace(self, run_id: UUID, **changes: Any) -> EvaluationRunStatus:
        with self._lock:
            current = self._runs.get(run_id)
            if current is None:
                raise KeyError(run_id)
            updated = current.model_copy(update=changes)
            self._runs[run_id] = updated
            return updated
