"""Runtime configuration loaded explicitly at application construction time."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


def _optional_path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value and value.strip() else None


class Settings(BaseModel):
    """Configuration with safe defaults and no required import-time secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    max_request_bytes: int = Field(default=16_384, ge=1_024, le=1_048_576)
    max_concurrent_evaluations: int = Field(default=1, ge=1, le=16)
    allow_test_evaluation: bool = False

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    routing_policy_path: Path | None = Path("experiments/results/routing_policy_v1.json")
    telemetry_path: Path | None = Path("telemetry/events.jsonl")
    dataset_root: Path = Path("data/iam_triage_v1")

    runtime_backend: Literal["heuristic", "huggingface"] = "heuristic"
    small_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    small_model_revision: str | None = None
    adapter_path: Path | None = None
    adapter_revision: str | None = None
    device: str = "auto"
    local_files_only: bool = False

    frontier_base_url: str | None = None
    frontier_model: str | None = None
    frontier_api_key: SecretStr | None = None
    frontier_input_usd_per_million: float = Field(default=0.0, ge=0.0)
    frontier_output_usd_per_million: float = Field(default=0.0, ge=0.0)
    frontier_pricing_source: str | None = Field(default=None, max_length=1_000)
    frontier_pricing_as_of: date | None = None
    frontier_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("unsupported log level")
        return normalized

    def resolve_path(self, value: Path | None) -> Path | None:
        if value is None or value.is_absolute():
            return value
        return self.project_root / value

    @property
    def frontier_configured(self) -> bool:
        return bool(
            self.frontier_base_url
            and self.frontier_model
            and self.frontier_api_key
            and self.frontier_api_key.get_secret_value()
        )

    @classmethod
    def from_env(cls, *, project_root: Path | None = None) -> Settings:
        env = os.environ
        return cls(
            environment=env.get("MODELFORGE_ENV", "development"),
            log_level=env.get("MODELFORGE_LOG_LEVEL", "INFO"),
            max_request_bytes=int(env.get("MODELFORGE_MAX_REQUEST_BYTES", "16384")),
            max_concurrent_evaluations=int(
                env.get("MODELFORGE_MAX_CONCURRENT_EVALUATIONS", "1")
            ),
            allow_test_evaluation=env.get(
                "MODELFORGE_ALLOW_TEST_EVALUATION",
                "false",
            ).casefold()
            in {"1", "true", "yes"},
            project_root=project_root or Path.cwd(),
            routing_policy_path=_optional_path(env.get("MODELFORGE_ROUTING_POLICY")),
            telemetry_path=_optional_path(env.get("MODELFORGE_TELEMETRY_PATH")),
            dataset_root=Path(env.get("MODELFORGE_DATASET_ROOT", "data/iam_triage_v1")),
            runtime_backend=env.get("MODELFORGE_RUNTIME_BACKEND", "heuristic"),
            small_model_name=env.get(
                "MODELFORGE_SMALL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
            ),
            small_model_revision=env.get("MODELFORGE_SMALL_MODEL_REVISION") or None,
            adapter_path=_optional_path(env.get("MODELFORGE_ADAPTER_PATH")),
            adapter_revision=env.get("MODELFORGE_ADAPTER_REVISION") or None,
            device=env.get("MODELFORGE_DEVICE", "auto"),
            local_files_only=env.get("MODELFORGE_LOCAL_FILES_ONLY", "false").casefold()
            in {"1", "true", "yes"},
            frontier_base_url=env.get("MODELFORGE_FRONTIER_BASE_URL") or None,
            frontier_model=env.get("MODELFORGE_FRONTIER_MODEL") or None,
            frontier_api_key=(
                SecretStr(env["MODELFORGE_FRONTIER_API_KEY"])
                if env.get("MODELFORGE_FRONTIER_API_KEY")
                else None
            ),
            frontier_input_usd_per_million=float(
                env.get("MODELFORGE_FRONTIER_INPUT_USD_PER_MILLION", "0")
            ),
            frontier_output_usd_per_million=float(
                env.get("MODELFORGE_FRONTIER_OUTPUT_USD_PER_MILLION", "0")
            ),
            frontier_pricing_source=(
                env.get("MODELFORGE_FRONTIER_PRICING_SOURCE") or None
            ),
            frontier_pricing_as_of=(
                env.get("MODELFORGE_FRONTIER_PRICING_AS_OF") or None
            ),
            frontier_timeout_seconds=float(
                env.get("MODELFORGE_FRONTIER_TIMEOUT_SECONDS", "30")
            ),
        )
