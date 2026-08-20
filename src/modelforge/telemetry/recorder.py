"""Privacy-safe operational telemetry for a single-process v1 service."""

from __future__ import annotations

import math
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TelemetryEvent(BaseModel):
    """Operational fields only; ticket text, evidence, and employee data are absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_id: str = Field(min_length=8, max_length=64)
    model_selected: str = Field(min_length=1, max_length=200)
    status: Literal["ok", "error"]
    fallback: bool = False
    route_reason: str | None = Field(default=None, max_length=120)
    routing_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0.0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    schema_failure: bool = False
    error_code: str | None = Field(default=None, max_length=80)


class TelemetrySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_count: int
    error_rate: float
    schema_failure_rate: float
    fallback_rate: float
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    estimated_cost_usd: float
    model_counts: dict[str, int]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


class TelemetryRecorder:
    """In-memory summary plus optional locked JSONL development sink.

    The file backend is intentionally documented as single-process. Production
    deployments should replace this interface with a durable telemetry sink.
    """

    def __init__(self, path: Path | None = None, *, memory_limit: int = 10_000) -> None:
        self.path = path
        self._events: deque[TelemetryEvent] = deque(maxlen=memory_limit)
        self._lock = threading.Lock()

    def record(self, event: TelemetryEvent) -> None:
        encoded = event.model_dump_json()
        with self._lock:
            self._events.append(event)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(encoded + "\n")

    def events(self) -> tuple[TelemetryEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def summary(self) -> TelemetrySummary:
        events = self.events()
        count = len(events)
        if not count:
            return TelemetrySummary(
                request_count=0,
                error_rate=0.0,
                schema_failure_rate=0.0,
                fallback_rate=0.0,
                average_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                estimated_cost_usd=0.0,
                model_counts={},
            )
        latencies = [event.latency_ms for event in events]
        model_counts: dict[str, int] = {}
        for event in events:
            model_counts[event.model_selected] = model_counts.get(event.model_selected, 0) + 1
        return TelemetrySummary(
            request_count=count,
            error_rate=sum(event.status == "error" for event in events) / count,
            schema_failure_rate=sum(event.schema_failure for event in events) / count,
            fallback_rate=sum(event.fallback for event in events) / count,
            average_latency_ms=sum(latencies) / count,
            p50_latency_ms=_percentile(latencies, 0.50),
            p95_latency_ms=_percentile(latencies, 0.95),
            estimated_cost_usd=sum(event.estimated_cost_usd for event in events),
            model_counts=model_counts,
        )

    @classmethod
    def read_jsonl(cls, path: Path) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    events.append(TelemetryEvent.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"invalid telemetry at {path}:{line_number}") from exc
        return events
