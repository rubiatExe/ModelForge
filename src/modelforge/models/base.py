"""Typed, provider-independent model contracts.

The contract deliberately contains operational metadata but no ticket payload.
Callers may keep ``raw_output`` in memory for debugging/evaluation, but Pydantic
excludes it from normal serialization so it cannot drift into telemetry by
accident.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import ConfigDict, Field

from modelforge.errors import ModelForgeError
from modelforge.schemas import TicketInput, TriageResult
from modelforge.schemas._base import StrictBaseModel

ModelRole = Literal["small", "frontier", "classical", "demo"]
SafeMetadataValue = str | int | float | bool | None


class ModelPrediction(StrictBaseModel):
    """A validated triage result plus comparable inference metadata."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    result: TriageResult
    model_id: str = Field(min_length=1, max_length=300)
    model_version: str = Field(min_length=1, max_length=300)
    role: ModelRole
    prompt_version: str | None = Field(default=None, max_length=100)
    latency_ms: float = Field(ge=0.0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    label_logprob: float | None = Field(default=None, ge=0.0, le=1.0)
    sample_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    provider_request_id: str | None = Field(default=None, max_length=300)
    raw_output: str | None = Field(default=None, repr=False, exclude=True, max_length=50_000)
    metadata: dict[str, SafeMetadataValue] = Field(default_factory=dict)

    @property
    def output(self) -> TriageResult:
        """Compatibility/readability alias for the validated result."""

        return self.result


@runtime_checkable
class TriageModel(Protocol):
    """The single asynchronous boundary used by API, routing, and evaluation."""

    @property
    def model_id(self) -> str: ...

    @property
    def role(self) -> ModelRole: ...

    async def triage(
        self,
        ticket: TicketInput,
        *,
        timeout_s: float | None = None,
    ) -> ModelPrediction: ...


class ModelAdapterError(ModelForgeError):
    """Base class for safe, typed adapter failures."""

    code = "model_adapter_error"
    retryable = False

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class ModelConfigurationError(ModelAdapterError):
    code = "model_configuration_error"


class ModelDependencyError(ModelConfigurationError):
    code = "model_dependency_error"


class ModelLoadError(ModelAdapterError):
    code = "model_load_error"


class ModelInferenceError(ModelAdapterError):
    code = "model_inference_error"


class ModelTimeoutError(ModelAdapterError):
    code = "model_timeout"
    retryable = True


class ModelProviderError(ModelAdapterError):
    code = "model_provider_error"
    retryable = True


class ModelOutputError(ModelAdapterError):
    code = "model_output_invalid"

    def __init__(
        self,
        message: str,
        *,
        raw_output: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        estimated_cost_usd: float = 0.0,
        model_version: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        # Kept only in memory for evaluators that distinguish schema-invalid
        # output from provider failure. Exception text/repr never includes it.
        super().__init__(message, cause=cause)
        self.raw_output = raw_output
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.estimated_cost_usd = estimated_cost_usd
        self.model_version = model_version
