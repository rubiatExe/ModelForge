"""Strict YAML configuration for reproducible LoRA experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from modelforge.models.base import ModelConfigurationError, ModelDependencyError
from modelforge.schemas._base import StrictBaseModel


class LoraParameters(StrictBaseModel):
    rank: int = Field(default=16, ge=1, le=512)
    alpha: int = Field(default=32, ge=1, le=2_048)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )


class LoraTrainingConfig(StrictBaseModel):
    config_version: Literal["1.0"] = "1.0"
    experiment_name: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    model_name_or_path: str = Field(min_length=1, max_length=1_000)
    model_revision: str = Field(min_length=1, max_length=200)
    tokenizer_name_or_path: str | None = Field(default=None, max_length=1_000)
    tokenizer_revision: str | None = Field(default=None, max_length=200)
    trust_remote_code: bool = False
    local_files_only: bool = False
    require_resolved_revision: bool = True

    train_data: Path
    validation_data: Path
    prompt_path: Path = Path("prompts/iam_triage_v1.txt")
    output_root: Path = Path("experiments/runs")

    seed: int = Field(default=17, ge=0, le=2**32 - 1)
    max_length: int = Field(default=1_024, ge=128, le=32_768)
    minimum_response_tokens: int = Field(default=16, ge=1, le=1_024)
    training_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Neutral self-report target; never calibration evidence.",
    )

    epochs: float = Field(default=3.0, gt=0.0, le=100.0)
    optimizer: Literal["adamw_torch"] = "adamw_torch"
    learning_rate: float = Field(default=2e-4, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    train_batch_size: int = Field(default=4, ge=1, le=1_024)
    validation_batch_size: int = Field(default=4, ge=1, le=1_024)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=1_024)
    gradient_checkpointing: bool = False
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    logging_steps: int = Field(default=10, ge=1)
    save_total_limit: int = Field(default=2, ge=1, le=20)
    early_stopping_patience: int = Field(default=2, ge=0, le=20)
    dataloader_num_workers: int = Field(default=0, ge=0, le=64)

    device: str = Field(default="auto", min_length=1, max_length=50)
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
    deterministic_algorithms: bool = True
    lora: LoraParameters = Field(default_factory=LoraParameters)

    @model_validator(mode="after")
    def validate_paths_and_lengths(self) -> LoraTrainingConfig:
        if self.train_data == self.validation_data:
            raise ValueError("train_data and validation_data must be different files")
        if self.minimum_response_tokens >= self.max_length:
            raise ValueError("minimum_response_tokens must be smaller than max_length")
        return self

    @property
    def run_directory(self) -> Path:
        return self.output_root / self.experiment_name

    @property
    def content_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_training_config(
    path: Path,
    *,
    project_root: Path | None = None,
) -> LoraTrainingConfig:
    """Load YAML safely and resolve artifact paths against an explicit root."""

    try:
        import yaml
    except ImportError as exc:
        raise ModelDependencyError("YAML training configs require PyYAML", cause=exc) from exc
    config_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ModelConfigurationError("training config is unavailable", cause=exc) from exc
    except yaml.YAMLError as exc:
        raise ModelConfigurationError("training config is invalid YAML", cause=exc) from exc
    if not isinstance(raw, dict):
        raise ModelConfigurationError("training config must be a YAML object")
    try:
        config = LoraTrainingConfig.model_validate(raw)
    except Exception as exc:
        raise ModelConfigurationError("training config failed validation", cause=exc) from exc

    root = (project_root or Path.cwd()).resolve()

    def resolve(value: Path) -> Path:
        return value if value.is_absolute() else root / value

    return config.model_copy(
        update={
            "train_data": resolve(config.train_data),
            "validation_data": resolve(config.validation_data),
            "prompt_path": resolve(config.prompt_path),
            "output_root": resolve(config.output_root),
        }
    )
