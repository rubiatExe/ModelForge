from pathlib import Path

from modelforge.telemetry import TelemetryEvent, TelemetryRecorder


def test_telemetry_summary_and_privacy_contract(tmp_path: Path) -> None:
    recorder = TelemetryRecorder(tmp_path / "events.jsonl")
    recorder.record(
        TelemetryEvent(
            request_id="request-0001",
            model_selected="small-v1",
            status="ok",
            latency_ms=10,
            estimated_cost_usd=0.001,
        )
    )
    recorder.record(
        TelemetryEvent(
            request_id="request-0002",
            model_selected="frontier-v1",
            status="error",
            fallback=True,
            latency_ms=30,
            estimated_cost_usd=0.004,
        )
    )

    summary = recorder.summary()
    assert summary.request_count == 2
    assert summary.error_rate == 0.5
    assert summary.fallback_rate == 0.5
    assert summary.p95_latency_ms == 30
    assert summary.estimated_cost_usd == 0.005

    raw = (tmp_path / "events.jsonl").read_text()
    assert "ticket" not in raw
    assert len(TelemetryRecorder.read_jsonl(tmp_path / "events.jsonl")) == 2
