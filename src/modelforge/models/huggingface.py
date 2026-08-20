"""Lazy Hugging Face base/PEFT adapter inference behind the common contract."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from modelforge.models.base import (
    ModelConfigurationError,
    ModelDependencyError,
    ModelInferenceError,
    ModelLoadError,
    ModelOutputError,
    ModelPrediction,
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


class HuggingFaceModelConfig(StrictBaseModel):
    model_name_or_path: str = Field(min_length=1, max_length=1_000)
    revision: str = Field(min_length=1, max_length=200)
    adapter_name_or_path: str | None = Field(default=None, max_length=1_000)
    adapter_revision: str | None = Field(default=None, max_length=200)
    serving_id: str = Field(default="small-qwen-lora-v1", min_length=1, max_length=300)
    device: str = Field(default="auto", min_length=1, max_length=50)
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
    max_input_tokens: int = Field(default=2_048, ge=64, le=32_768)
    max_new_tokens: int = Field(default=512, ge=16, le=4_096)
    local_files_only: bool = False
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def adapter_revision_requires_adapter(self) -> HuggingFaceModelConfig:
        if self.adapter_revision and not self.adapter_name_or_path:
            raise ValueError("adapter_revision requires adapter_name_or_path")
        return self


class HuggingFaceGeneration(StrictBaseModel):
    content: str = Field(min_length=1, max_length=50_000)
    model_version: str = Field(min_length=1, max_length=1_000)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    label_logprob: float | None = Field(default=None, ge=0.0, le=1.0)


class HuggingFaceBackend(Protocol):
    async def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        timeout_s: float,
    ) -> HuggingFaceGeneration: ...


def _require_huggingface(adapter: bool) -> tuple[Any, Any, Any, Any | None]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ModelDependencyError(
            "Hugging Face inference requires the 'train' extra: pip install -e '.[train]'",
            cause=exc,
        ) from exc
    PeftModel: Any | None = None
    if adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ModelDependencyError(
                "loading a LoRA adapter requires PEFT: pip install -e '.[train]'",
                cause=exc,
            ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, PeftModel


def _device_and_dtype(torch: Any, config: HuggingFaceModelConfig) -> tuple[str, Any]:
    requested = config.device
    if requested == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ModelConfigurationError("CUDA was requested but is unavailable")
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ModelConfigurationError("MPS was requested but is unavailable")

    precision = config.precision
    if precision == "auto":
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            precision = "bf16"
        elif device.startswith("cuda") or device == "mps":
            precision = "fp16"
        else:
            precision = "fp32"
    if precision in {"fp16", "bf16"} and device == "cpu":
        raise ModelConfigurationError(f"{precision} inference is not enabled on CPU")
    if precision == "bf16" and device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise ModelConfigurationError("BF16 was requested on a CUDA device without BF16 support")
    return device, {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


class TransformersBackend:
    """Serialized lazy generation for a frozen base and optional PEFT adapter."""

    def __init__(self, config: HuggingFaceModelConfig) -> None:
        self.config = config
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self._model_version: str | None = None
        # A dedicated single worker is a resource-safety boundary. If an async
        # deadline expires, native generation cannot be killed, but subsequent
        # calls remain queued behind it instead of racing the same model.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="modelforge-hf")
        self._load_future: asyncio.Future[None] | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            if self._load_future is None:
                loop = asyncio.get_running_loop()
                self._load_future = asyncio.ensure_future(
                    loop.run_in_executor(self._executor, self._load_sync)
                )
            load_future = self._load_future
        # Shield keeps one canonical load alive after a caller deadline. A
        # later request awaits the same future rather than loading twice.
        await asyncio.shield(load_future)

    def _load_sync(self) -> None:
        torch, AutoModel, AutoTokenizer, PeftModel = _require_huggingface(
            self.config.adapter_name_or_path is not None
        )
        device, dtype = _device_and_dtype(torch, self.config)
        common = {
            "revision": self.config.revision,
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": self.config.trust_remote_code,
        }
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, **common)
            model = AutoModel.from_pretrained(
                self.config.model_name_or_path,
                torch_dtype=dtype,
                **common,
            )
            if self.config.adapter_name_or_path:
                adapter_kwargs = {
                    "revision": self.config.adapter_revision,
                    "is_trainable": False,
                }
                adapter_kwargs = {key: value for key, value in adapter_kwargs.items() if value is not None}
                model = PeftModel.from_pretrained(
                    model,
                    self.config.adapter_name_or_path,
                    **adapter_kwargs,
                )
            model.to(device)
            model.eval()
        except Exception as exc:
            raise ModelLoadError("failed to load Hugging Face model artifacts", cause=exc) from exc
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
        base_version = resolved or self.config.revision
        adapter_version = self.config.adapter_revision or self.config.adapter_name_or_path
        self._model_version = (
            f"{base_version}+adapter:{Path(adapter_version).name}"
            if adapter_version
            else str(base_version)
        )
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _generate_sync(self, messages: Sequence[Mapping[str, str]]) -> HuggingFaceGeneration:
        tokenizer, model, torch, device = (
            self._tokenizer,
            self._model,
            self._torch,
            self._device,
        )
        try:
            prompt = tokenizer.apply_chat_template(
                [dict(item) for item in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_input_tokens,
                add_special_tokens=False,
            )
            if hasattr(encoded, "to"):
                encoded = encoded.to(device)
            else:
                encoded = {key: value.to(device) for key, value in encoded.items()}
            input_count = int(encoded["input_ids"].shape[-1])
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0][input_count:]
            content = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            output_count = int(new_tokens.shape[-1])
        except Exception as exc:
            raise ModelInferenceError("Hugging Face generation failed", cause=exc) from exc
        if not content:
            raise ModelInferenceError("Hugging Face model returned an empty completion")
        return HuggingFaceGeneration(
            content=content,
            model_version=self._model_version or self.config.revision,
            input_tokens=input_count,
            output_tokens=output_count,
        )

    async def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        timeout_s: float,
    ) -> HuggingFaceGeneration:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        try:
            async with asyncio.timeout(timeout_s):
                await self._ensure_loaded()
                async with self._inference_lock:
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        self._executor,
                        self._generate_sync,
                        messages,
                    )
        except TimeoutError as exc:
            # Python cannot stop an already-running native generation thread.
            # The dedicated single-worker executor keeps later work queued.
            raise ModelTimeoutError("Hugging Face inference deadline exceeded", cause=exc) from exc

    def close(self) -> None:
        """Stop accepting queued work; running native work finishes in-place."""

        self._executor.shutdown(wait=False, cancel_futures=True)


class HuggingFaceTriageModel:
    """Base or adapter-backed small model with strict output validation."""

    role: ModelRole = "small"

    def __init__(
        self,
        config: HuggingFaceModelConfig,
        *,
        backend: HuggingFaceBackend | None = None,
        prompt: PromptDefinition | None = None,
        default_timeout_s: float = 60.0,
    ) -> None:
        if default_timeout_s <= 0:
            raise ModelConfigurationError("default_timeout_s must be positive")
        self.config = config
        self.model_id = config.serving_id
        self.backend = backend or TransformersBackend(config)
        self.prompt = prompt or load_iam_prompt()
        self.default_timeout_s = default_timeout_s

    async def triage(
        self,
        ticket: TicketInput,
        *,
        timeout_s: float | None = None,
    ) -> ModelPrediction:
        deadline = timeout_s if timeout_s is not None else self.default_timeout_s
        messages = render_iam_messages(ticket, prompt=self.prompt)
        started = time.perf_counter()
        generation = await self.backend.generate(messages, timeout_s=deadline)
        try:
            result = parse_triage_json(generation.content)
            require_grounded_evidence(ticket, result, raw_output=generation.content)
        except ModelOutputError as exc:
            raise ModelOutputError(
                str(exc),
                raw_output=exc.raw_output or generation.content,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
                model_version=generation.model_version,
                cause=exc,
            ) from exc
        return ModelPrediction(
            result=result,
            model_id=self.model_id,
            model_version=generation.model_version,
            role=self.role,
            prompt_version=self.prompt.version,
            latency_ms=(time.perf_counter() - started) * 1_000,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            estimated_cost_usd=0.0,
            label_logprob=generation.label_logprob,
            raw_output=generation.content,
            metadata={
                "adapter_loaded": self.config.adapter_name_or_path is not None,
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

    def close(self) -> None:
        """Release an owned native backend executor when the service stops."""

        close = getattr(self.backend, "close", None)
        if callable(close):
            close()
