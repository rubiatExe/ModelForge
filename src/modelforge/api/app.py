"""FastAPI application factory and keyless development service."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from modelforge import __version__
from modelforge.api.contracts import (
    EvaluationRunRequest,
    EvaluationRunStatus,
    HealthResponse,
    ModelDescriptor,
)
from modelforge.api.errors import error_response, install_exception_handlers
from modelforge.api.evaluation_store import EvaluationStore
from modelforge.api.evaluator import LocalEvaluationExecutor
from modelforge.api.middleware import RequestSizeLimitMiddleware
from modelforge.api.registry import ModelRegistry
from modelforge.config import Settings
from modelforge.errors import ConfigurationError, ModelInferenceError
from modelforge.logging import configure_logging
from modelforge.models.base import ModelAdapterError, TriageModel
from modelforge.models.heuristic import HeuristicTriageModel
from modelforge.routing import RoutingPolicy, TriageRouter, load_routing_policy
from modelforge.schemas import TicketInput, TriageResult
from modelforge.telemetry import TelemetryEvent, TelemetryRecorder, TelemetrySummary


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _development_policy() -> RoutingPolicy:
    return RoutingPolicy.model_validate(
        {
            "policy_id": "iam-routing-v1-dev",
            "version": "1.0.0",
            "status": "uncalibrated",
            "threshold": 0.85,
            "small_model_id": "heuristic-demo-v1",
            "frontier_model_id": "unconfigured-frontier",
            "dataset_version": "iam_ticket_triage@1.0.0",
            "validation_run_id": None,
            "created_at": "2026-08-20T00:00:00Z",
            "notes": "Embedded development fallback; not measured evidence.",
        }
    )


def _policy_from_settings(settings: Settings) -> RoutingPolicy:
    path = settings.resolve_path(settings.routing_policy_path)
    if path is not None and path.exists():
        policy = load_routing_policy(path)
    elif settings.environment == "production":
        raise ConfigurationError("production routing policy artifact is missing")
    else:
        policy = _development_policy()
    if settings.environment == "production" and policy.status != "calibrated":
        raise ConfigurationError("production requires a calibrated routing policy")
    return policy


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    for model in (app.state.small_model, app.state.frontier_model):
        if model is None:
            continue
        close = getattr(model, "aclose", None) or getattr(model, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


def create_app(
    *,
    settings: Settings | None = None,
    small_model: TriageModel | None = None,
    frontier_model: TriageModel | None = None,
    policy: RoutingPolicy | None = None,
    evaluation_executor: Any | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env(project_root=_source_root())
    if small_model is None:
        if settings.runtime_backend == "heuristic":
            small_model = HeuristicTriageModel()
        else:
            if not settings.small_model_revision:
                raise ConfigurationError(
                    "Hugging Face runtime requires MODELFORGE_SMALL_MODEL_REVISION"
                )
            from modelforge.models.huggingface import (
                HuggingFaceModelConfig,
                HuggingFaceTriageModel,
            )
            small_model = HuggingFaceTriageModel(
                HuggingFaceModelConfig(
                    model_name_or_path=settings.small_model_name,
                    revision=settings.small_model_revision,
                    adapter_name_or_path=(
                        str(settings.resolve_path(settings.adapter_path))
                        if settings.adapter_path
                        else None
                    ),
                    adapter_revision=settings.adapter_revision,
                    serving_id="small-iam-triage-v1",
                    device=settings.device,
                    local_files_only=settings.local_files_only,
                )
            )
    if settings.frontier_configured and frontier_model is None:
        from modelforge.models.frontier import (
            FrontierTriageModel,
            OpenAICompatibleConfig,
            OpenAICompatibleFrontierProvider,
        )
        assert settings.frontier_base_url is not None
        assert settings.frontier_model is not None
        assert settings.frontier_api_key is not None
        provider = OpenAICompatibleFrontierProvider(
            OpenAICompatibleConfig(
                base_url=settings.frontier_base_url,
                api_key=settings.frontier_api_key,
                model=settings.frontier_model,
                timeout_seconds=settings.frontier_timeout_seconds,
                input_usd_per_million=settings.frontier_input_usd_per_million,
                output_usd_per_million=settings.frontier_output_usd_per_million,
                pricing_source=settings.frontier_pricing_source,
                pricing_as_of=settings.frontier_pricing_as_of,
            )
        )
        frontier_model = FrontierTriageModel(
            provider,
            model_id="frontier-iam-triage-v1",
            default_timeout_s=settings.frontier_timeout_seconds,
        )
    policy = policy or _policy_from_settings(settings)
    if policy.status == "calibrated":
        if policy.small_model_id != small_model.model_id:
            raise ConfigurationError("routing policy small-model identity does not match runtime")
        if frontier_model is None or policy.frontier_model_id != frontier_model.model_id:
            raise ConfigurationError("routing policy frontier identity does not match runtime")
    registry = ModelRegistry(tuple(model for model in (small_model, frontier_model) if model))
    router = TriageRouter(
        small_model=small_model,
        frontier_model=frontier_model,
        policy=policy,
        per_model_timeout_s=settings.frontier_timeout_seconds,
    )
    telemetry = TelemetryRecorder(settings.resolve_path(settings.telemetry_path))
    evaluation_store = EvaluationStore()

    app = FastAPI(
        title="ModelForge IAM Triage",
        version=__version__,
        description="Evidence-first selective routing for enterprise IAM support tickets.",
        lifespan=_lifespan,
    )
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    install_exception_handlers(app)
    app.state.settings = settings
    app.state.registry = registry
    app.state.router = router
    app.state.telemetry = telemetry
    app.state.evaluation_store = evaluation_store
    app.state.evaluation_executor = evaluation_executor or LocalEvaluationExecutor(
        registry=registry,
        dataset_root=settings.resolve_path(settings.dataset_root) or settings.dataset_root,
    )
    app.state.evaluation_semaphore = asyncio.Semaphore(settings.max_concurrent_evaluations)
    app.state.evaluation_tasks = set()
    app.state.small_model = small_model
    app.state.frontier_model = frontier_model

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if policy.status == "calibrated" else "degraded",
            version=__version__,
            small_model_ready=registry.get(small_model.model_id) is not None,
            frontier_configured=frontier_model is not None,
            routing_policy_status=policy.status,
        )

    @app.get("/v1/models", response_model=list[ModelDescriptor], tags=["models"])
    async def models() -> list[ModelDescriptor]:
        return registry.descriptors()

    @app.post("/v1/triage", response_model=TriageResult, tags=["triage"])
    async def triage(ticket: TicketInput, response: Response) -> TriageResult:
        request_id = uuid4().hex
        started = time.perf_counter()
        try:
            routed = await router.triage(ticket)
        except Exception as exc:
            telemetry.record(
                TelemetryEvent(
                    request_id=request_id,
                    model_selected=small_model.model_id,
                    status="error",
                    fallback=False,
                    route_reason="routing_error",
                    latency_ms=(time.perf_counter() - started) * 1_000,
                    error_code=getattr(exc, "code", type(exc).__name__)[:80],
                )
            )
            raise
        response.headers["X-Request-ID"] = request_id
        response.headers["X-ModelForge-Model"] = routed.prediction.model_id
        response.headers["X-ModelForge-Route"] = routed.reason
        response.headers["X-ModelForge-Policy"] = routed.policy_id
        telemetry.record(
            TelemetryEvent(
                request_id=request_id,
                model_selected=routed.prediction.model_id,
                status="ok",
                fallback=routed.used_frontier,
                route_reason=routed.reason,
                routing_confidence=(
                    routed.confidence_assessment.score
                    if routed.confidence_assessment is not None
                    else None
                ),
                input_tokens=routed.total_input_tokens,
                output_tokens=routed.total_output_tokens,
                latency_ms=routed.total_latency_ms,
                estimated_cost_usd=routed.total_estimated_cost_usd,
            )
        )
        return routed.prediction.result

    @app.post("/v1/models/{model_id:path}/triage", response_model=TriageResult, tags=["triage"])
    async def model_triage(model_id: str, ticket: TicketInput, response: Response) -> TriageResult | JSONResponse:
        model = registry.get(model_id)
        if model is None:
            return error_response(404, "MODEL_NOT_FOUND", "requested model is not registered")
        request_id = uuid4().hex
        started = time.perf_counter()
        try:
            prediction = await model.triage(ticket, timeout_s=settings.frontier_timeout_seconds)
        except ModelAdapterError as exc:
            telemetry.record(
                TelemetryEvent(
                    request_id=request_id,
                    model_selected=model.model_id,
                    status="error",
                    latency_ms=(time.perf_counter() - started) * 1_000,
                    schema_failure=exc.code == "model_output_invalid",
                    error_code=exc.code,
                )
            )
            raise ModelInferenceError("the requested model could not produce a valid result") from exc
        response.headers["X-Request-ID"] = request_id
        response.headers["X-ModelForge-Model"] = prediction.model_id
        telemetry.record(
            TelemetryEvent(
                request_id=request_id,
                model_selected=prediction.model_id,
                status="ok",
                input_tokens=prediction.input_tokens,
                output_tokens=prediction.output_tokens,
                latency_ms=prediction.latency_ms,
                estimated_cost_usd=prediction.estimated_cost_usd,
            )
        )
        return prediction.result

    @app.get("/v1/telemetry", response_model=TelemetrySummary, tags=["operations"])
    async def telemetry_summary() -> TelemetrySummary:
        return telemetry.summary()

    @app.post(
        "/v1/evaluations/run",
        response_model=EvaluationRunStatus,
        status_code=202,
        tags=["evaluations"],
    )
    async def start_evaluation(request: EvaluationRunRequest) -> EvaluationRunStatus | JSONResponse:
        if registry.get(request.model_id) is None:
            return error_response(404, "MODEL_NOT_FOUND", "requested model is not registered")
        if request.split == "test" and not settings.allow_test_evaluation:
            return error_response(
                403,
                "TEST_EVALUATION_DISABLED",
                "locked-test evaluation requires an explicit runtime opt-in",
            )
        run = evaluation_store.create(request)

        async def execute() -> None:
            evaluation_store.mark_running(run.run_id)
            try:
                async with app.state.evaluation_semaphore:
                    executor = app.state.evaluation_executor
                    if hasattr(executor, "run"):
                        report = await executor.run(request)
                    else:
                        report = await executor(request)
                evaluation_store.complete(run.run_id, report)
            except Exception:
                evaluation_store.fail(run.run_id, "EVALUATION_FAILED")

        task = asyncio.create_task(execute())
        app.state.evaluation_tasks.add(task)
        task.add_done_callback(app.state.evaluation_tasks.discard)
        return run

    @app.get(
        "/v1/evaluations/{run_id}",
        response_model=EvaluationRunStatus,
        tags=["evaluations"],
    )
    async def get_evaluation(run_id: UUID) -> EvaluationRunStatus | JSONResponse:
        run = evaluation_store.get(run_id)
        if run is None:
            return error_response(404, "EVALUATION_NOT_FOUND", "evaluation run was not found")
        return run

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = Settings.from_env(project_root=_source_root())
    configure_logging(settings.log_level)
    uvicorn.run(
        "modelforge.api.app:app",
        host=os.environ.get("MODELFORGE_HOST", "127.0.0.1"),
        port=int(os.environ.get("MODELFORGE_PORT", "8000")),
        reload=False,
    )
