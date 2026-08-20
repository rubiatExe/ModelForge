"""Provider-neutral frontier inference and an OpenAI-compatible HTTP adapter."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Protocol, runtime_checkable

from pydantic import Field, SecretStr, field_validator, model_validator

from modelforge.models.base import (
    ModelConfigurationError,
    ModelDependencyError,
    ModelOutputError,
    ModelPrediction,
    ModelProviderError,
    ModelRole,
    ModelTimeoutError,
)
from modelforge.models.prompting import (
    PromptDefinition,
    load_iam_prompt,
    parse_triage_json,
    render_iam_messages,
    require_grounded_evidence,
)
from modelforge.schemas import TicketInput
from modelforge.schemas._base import StrictBaseModel


class FrontierCompletion(StrictBaseModel):
    content: str = Field(min_length=1, max_length=50_000)
    model_version: str = Field(min_length=1, max_length=300)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    provider_request_id: str | None = Field(default=None, max_length=300)


@runtime_checkable
class FrontierProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        timeout_s: float,
    ) -> FrontierCompletion: ...


class OpenAICompatibleConfig(StrictBaseModel):
    provider_id: str = Field(default="openai-compatible", min_length=1, max_length=100)
    base_url: str = Field(min_length=8, max_length=2_000)
    api_key: SecretStr
    model: str = Field(min_length=1, max_length=300)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    input_usd_per_million: float = Field(default=0.0, ge=0.0)
    output_usd_per_million: float = Field(default=0.0, ge=0.0)
    pricing_source: str | None = Field(default=None, min_length=1, max_length=1_000)
    pricing_as_of: date | None = None
    json_mode: bool = True

    @field_validator("base_url")
    @classmethod
    def require_http_url(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("base_url must use http or https")
        return value.rstrip("/")

    @model_validator(mode="after")
    def priced_requests_require_provenance(self) -> OpenAICompatibleConfig:
        has_pricing = self.input_usd_per_million > 0 or self.output_usd_per_million > 0
        if has_pricing and (self.pricing_source is None or self.pricing_as_of is None):
            raise ValueError(
                "non-zero frontier pricing requires pricing_source and pricing_as_of"
            )
        if (self.pricing_source is None) != (self.pricing_as_of is None):
            raise ValueError("pricing_source and pricing_as_of must be provided together")
        return self

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> OpenAICompatibleConfig:
        """Read provider settings explicitly at construction time, never import time."""

        env = environ if environ is not None else os.environ
        missing = [
            name
            for name in (
                "MODELFORGE_FRONTIER_BASE_URL",
                "MODELFORGE_FRONTIER_API_KEY",
                "MODELFORGE_FRONTIER_MODEL",
            )
            if not env.get(name)
        ]
        if missing:
            raise ModelConfigurationError(
                "frontier provider is not configured; missing " + ", ".join(missing)
            )
        try:
            return cls(
                provider_id=env.get("MODELFORGE_FRONTIER_PROVIDER_ID", "openai-compatible"),
                base_url=env["MODELFORGE_FRONTIER_BASE_URL"],
                api_key=SecretStr(env["MODELFORGE_FRONTIER_API_KEY"]),
                model=env["MODELFORGE_FRONTIER_MODEL"],
                timeout_seconds=float(env.get("MODELFORGE_FRONTIER_TIMEOUT_SECONDS", "30")),
                input_usd_per_million=float(
                    env.get("MODELFORGE_FRONTIER_INPUT_USD_PER_MILLION", "0")
                ),
            output_usd_per_million=float(
                env.get("MODELFORGE_FRONTIER_OUTPUT_USD_PER_MILLION", "0")
            ),
            pricing_source=env.get("MODELFORGE_FRONTIER_PRICING_SOURCE") or None,
            pricing_as_of=env.get("MODELFORGE_FRONTIER_PRICING_AS_OF") or None,
            json_mode=env.get("MODELFORGE_FRONTIER_JSON_MODE", "true").casefold()
                not in {"0", "false", "no"},
            )
        except (TypeError, ValueError) as exc:
            raise ModelConfigurationError("frontier provider environment is invalid", cause=exc) from exc


class AsyncHttpClient(Protocol):
    async def post(self, url: str, **kwargs: Any) -> Any: ...

    async def aclose(self) -> None: ...


def _usage_value(payload: Mapping[str, Any], name: str) -> int | None:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelProviderError("provider returned invalid token usage metadata")
    return value


class OpenAICompatibleFrontierProvider:
    """Minimal async `/chat/completions` client with injection and deadlines."""

    def __init__(
        self,
        config: OpenAICompatibleConfig | None = None,
        *,
        client: AsyncHttpClient | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config or OpenAICompatibleConfig.from_env(environ)
        self.provider_id = self.config.provider_id
        self._client = client
        self._owns_client = client is None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> AsyncHttpClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                try:
                    import httpx
                except ImportError as exc:
                    raise ModelDependencyError(
                        "OpenAI-compatible inference requires httpx",
                        cause=exc,
                    ) from exc
                self._client = httpx.AsyncClient()
        return self._client

    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        timeout_s: float,
    ) -> FrontierCompletion:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "temperature": 0,
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        try:
            async with asyncio.timeout(timeout_s):
                response = await client.post(
                    f"{self.config.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout_s,
                )
        except TimeoutError as exc:
            raise ModelTimeoutError("frontier provider deadline exceeded", cause=exc) from exc
        except ModelDependencyError:
            raise
        except Exception as exc:
            raise ModelProviderError("frontier provider request failed", cause=exc) from exc

        status = getattr(response, "status_code", None)
        if not isinstance(status, int):
            raise ModelProviderError("frontier provider returned an invalid HTTP response")
        if status >= 400:
            raise ModelProviderError(f"frontier provider returned HTTP {status}")
        try:
            body = response.json()
        except Exception as exc:
            raise ModelProviderError("frontier provider returned invalid JSON", cause=exc) from exc
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelProviderError("frontier provider response is missing completion content", cause=exc) from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelProviderError("frontier provider returned empty completion content")
        usage = body.get("usage") or {}
        if not isinstance(usage, Mapping):
            raise ModelProviderError("frontier provider returned invalid usage metadata")
        input_tokens = _usage_value(usage, "prompt_tokens")
        output_tokens = _usage_value(usage, "completion_tokens")
        cost = (
            ((input_tokens or 0) * self.config.input_usd_per_million)
            + ((output_tokens or 0) * self.config.output_usd_per_million)
        ) / 1_000_000
        response_headers = getattr(response, "headers", {})
        header_request_id = (
            response_headers.get("x-request-id") if isinstance(response_headers, Mapping) else None
        )
        request_id = body.get("id") or header_request_id
        model_version = body.get("model") or self.config.model
        if not isinstance(model_version, str) or not model_version:
            raise ModelProviderError("frontier provider returned an invalid model identity")
        return FrontierCompletion(
            content=content,
            model_version=model_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            provider_request_id=str(request_id) if request_id is not None else None,
        )

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()


class FrontierTriageModel:
    """Validate a provider-neutral completion through the common model boundary."""

    role: ModelRole = "frontier"

    def __init__(
        self,
        provider: FrontierProvider,
        *,
        model_id: str = "frontier-iam-triage-v1",
        prompt: PromptDefinition | None = None,
        default_timeout_s: float = 30.0,
    ) -> None:
        if default_timeout_s <= 0:
            raise ModelConfigurationError("default_timeout_s must be positive")
        self.provider = provider
        self.model_id = model_id
        self.prompt = prompt or load_iam_prompt()
        self.default_timeout_s = default_timeout_s

    async def triage(
        self,
        ticket: TicketInput,
        *,
        timeout_s: float | None = None,
    ) -> ModelPrediction:
        deadline = timeout_s if timeout_s is not None else self.default_timeout_s
        if deadline <= 0:
            raise ValueError("timeout_s must be positive")
        messages = render_iam_messages(ticket, prompt=self.prompt)
        started = time.perf_counter()
        completion = await self.provider.complete(messages, timeout_s=deadline)
        try:
            result = parse_triage_json(completion.content)
            require_grounded_evidence(ticket, result, raw_output=completion.content)
        except ModelOutputError as exc:
            raise ModelOutputError(
                str(exc),
                raw_output=exc.raw_output or completion.content,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                estimated_cost_usd=completion.estimated_cost_usd,
                model_version=completion.model_version,
                cause=exc,
            ) from exc
        return ModelPrediction(
            result=result,
            model_id=self.model_id,
            model_version=completion.model_version,
            role=self.role,
            prompt_version=self.prompt.version,
            latency_ms=(time.perf_counter() - started) * 1_000,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            estimated_cost_usd=completion.estimated_cost_usd,
            provider_request_id=completion.provider_request_id,
            raw_output=completion.content,
            metadata={
                "provider": self.provider.provider_id,
                "prompt_sha256": self.prompt.sha256,
                "confidence_kind": "self_reported",
            },
        )

    async def predict(
        self,
        ticket: TicketInput,
        *,
        timeout_s: float | None = None,
    ) -> ModelPrediction:
        return await self.triage(ticket, timeout_s=timeout_s)

    async def aclose(self) -> None:
        """Close an owned async provider client when supported."""

        close = getattr(self.provider, "aclose", None)
        if callable(close):
            await close()
