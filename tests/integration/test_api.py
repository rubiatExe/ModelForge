import time
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from modelforge.api.app import create_app
from modelforge.config import Settings
from modelforge.models.heuristic import HeuristicTriageModel
from modelforge.routing import RoutingPolicy


class FrontierFixture(HeuristicTriageModel):
    model_id = "frontier-fixture-v1"
    role = "frontier"


def make_policy(threshold: float) -> RoutingPolicy:
    return RoutingPolicy(
        policy_id="api-fixture-policy",
        version="1",
        status="uncalibrated",
        threshold=threshold,
        small_model_id="heuristic-demo-v1",
        frontier_model_id="frontier-fixture-v1",
        dataset_version="fixture",
        created_at=datetime(2026, 8, 20, tzinfo=UTC),
    )


def ticket_payload(body: str | None = None) -> dict:
    return {
        "ticket_id": "TKT-API-1",
        "subject": "SSO redirect loop after Okta",
        "body": body or "Okta shows success but an SSO redirect loop affects multiple users.",
        "employee_department": "Sales",
        "submitted_at": "2026-08-20T12:00:00Z",
    }


def test_api_model_to_validated_output_and_safe_telemetry() -> None:
    app = create_app(
        settings=Settings(environment="test", telemetry_path=None),
        policy=make_policy(0.0),
    )
    with TestClient(app) as client:
        response = client.post("/v1/triage", json=ticket_payload())
        assert response.status_code == 200
        assert set(response.json()) == {
            "issue_type",
            "severity",
            "routing_team",
            "affected_scope",
            "recommended_action",
            "evidence",
            "confidence",
        }
        assert response.headers["X-ModelForge-Model"] == "heuristic-demo-v1"
        summary = client.get("/v1/telemetry").json()
        assert summary["request_count"] == 1
        assert summary["model_counts"] == {"heuristic-demo-v1": 1}


def test_low_confidence_fallback_selects_frontier() -> None:
    app = create_app(
        settings=Settings(environment="test", telemetry_path=None),
        frontier_model=FrontierFixture(),
        policy=make_policy(0.99),
    )
    with TestClient(app) as client:
        response = client.post("/v1/triage", json=ticket_payload())
        assert response.status_code == 200
        assert response.headers["X-ModelForge-Model"] == "frontier-fixture-v1"
        assert response.headers["X-ModelForge-Route"] == "low_confidence"


def test_unconfigured_required_fallback_returns_503() -> None:
    app = create_app(
        settings=Settings(environment="test", telemetry_path=None),
        policy=make_policy(0.99),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/triage", json=ticket_payload())
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "FRONTIER_UNAVAILABLE"


def test_validation_error_does_not_reflect_ticket_body() -> None:
    app = create_app(
        settings=Settings(environment="test", telemetry_path=None),
        policy=make_policy(0.0),
    )
    payload = ticket_payload("secret-personal-value")
    payload["submitted_at"] = "not-a-date"
    with TestClient(app) as client:
        response = client.post("/v1/triage", json=payload)
        assert response.status_code == 422
        assert "secret-personal-value" not in response.text


def test_request_size_limit_rejects_before_validation() -> None:
    app = create_app(
        settings=Settings(environment="test", telemetry_path=None, max_request_bytes=1_024),
        policy=make_policy(0.0),
    )
    with TestClient(app) as client:
        response = client.post("/v1/triage", json=ticket_payload("x" * 2_000))
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_evaluation_run_completes_through_api() -> None:
    app = create_app(
        settings=Settings(environment="test", telemetry_path=None),
        policy=make_policy(0.0),
    )
    with TestClient(app) as client:
        started = client.post(
            "/v1/evaluations/run",
            json={"model_id": "heuristic-demo-v1", "split": "validation", "limit": 5},
        )
        assert started.status_code == 202
        run_id = started.json()["run_id"]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = client.get(f"/v1/evaluations/{run_id}")
            if status.json()["status"] in {"complete", "failed"}:
                break
            time.sleep(0.01)
        assert status.json()["status"] == "complete"
        assert status.json()["report"]["aggregate"]["total_cases"] == 5


def test_locked_test_evaluation_requires_explicit_runtime_opt_in() -> None:
    app = create_app(
        settings=Settings(environment="test", telemetry_path=None),
        policy=make_policy(0.0),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/evaluations/run",
            json={"model_id": "heuristic-demo-v1", "split": "test", "limit": 1},
        )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "TEST_EVALUATION_DISABLED"
