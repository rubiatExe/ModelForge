"""Safe exception mapping that never echoes ticket bodies or provider details."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from modelforge.api.contracts import ErrorDetail, ErrorEnvelope
from modelforge.errors import (
    ConfigurationError,
    FrontierUnavailableError,
    ModelForgeError,
    ModelInferenceError,
)

logger = logging.getLogger(__name__)


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: list[dict] | None = None,
) -> JSONResponse:
    body = ErrorEnvelope(error=ErrorDetail(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's full errors include the rejected input. Omit it so a ticket
        # body does not get reflected into logs, tracing, or client error stores.
        details = [
            {
                "location": [str(part) for part in error.get("loc", ())],
                "message": error.get("msg", "invalid value"),
                "type": error.get("type", "validation_error"),
            }
            for error in exc.errors()
        ]
        return error_response(422, "VALIDATION_ERROR", "request validation failed", details)

    @app.exception_handler(FrontierUnavailableError)
    async def frontier_unavailable(_: Request, exc: FrontierUnavailableError) -> JSONResponse:
        logger.warning("frontier escalation unavailable", extra={"error_code": "FRONTIER_UNAVAILABLE"})
        return error_response(503, "FRONTIER_UNAVAILABLE", str(exc))

    @app.exception_handler(ModelInferenceError)
    async def inference_error(_: Request, exc: ModelInferenceError) -> JSONResponse:
        logger.warning("model inference failed", extra={"error_code": "MODEL_INFERENCE_FAILED"})
        return error_response(502, "MODEL_INFERENCE_FAILED", str(exc))

    @app.exception_handler(ConfigurationError)
    async def configuration_error(_: Request, exc: ConfigurationError) -> JSONResponse:
        logger.error("runtime configuration error", extra={"error_code": "CONFIGURATION_ERROR"})
        return error_response(503, "CONFIGURATION_ERROR", str(exc))

    @app.exception_handler(ModelForgeError)
    async def expected_error(_: Request, exc: ModelForgeError) -> JSONResponse:
        logger.warning("ModelForge request failed", extra={"error_code": "MODELFORGE_ERROR"})
        return error_response(400, "MODELFORGE_ERROR", str(exc))

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled request error", extra={"error_code": "INTERNAL_ERROR"})
        return error_response(500, "INTERNAL_ERROR", "an unexpected error occurred")
