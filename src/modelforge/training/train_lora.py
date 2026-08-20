"""Readable, reproducible LoRA supervised fine-tuning entry point."""

from __future__ import annotations

import argparse
import inspect
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from modelforge.datasets import read_jsonl
from modelforge.models.base import (
    ModelConfigurationError,
    ModelDependencyError,
    ModelLoadError,
)
from modelforge.models.prompting import load_iam_prompt, render_iam_messages
from modelforge.schemas import DatasetSplit, LabeledExample, TriageResult
from modelforge.training.config import LoraTrainingConfig, load_training_config
from modelforge.training.manifest import (
    ExperimentManifest,
    ParameterCounts,
    artifact_hashes,
    collect_git_metadata,
    collect_hardware_metadata,
    collect_package_versions,
    detect_overfitting,
    extract_loss_history,
    sha256_file,
    utc_now,
    write_manifest_once,
)
from modelforge.training.runtime import choose_precision, set_deterministic_seeds
from modelforge.training.tokenization import ResponseOnlyDataCollator, tokenize_response_only


class _FeatureDataset:
    def __init__(self, rows: Sequence[dict[str, list[int]]]) -> None:
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self._rows[index]


def _require_training_stack() -> tuple[Any, ...]:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ModelDependencyError(
            "LoRA training requires the optional stack: pip install -e '.[train]'",
            cause=exc,
        ) from exc
    peft = (LoraConfig, TaskType, get_peft_model)
    transformers = (
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    return (torch, *peft, *transformers)


def _require_split(path: Path, split: DatasetSplit) -> list[LabeledExample]:
    try:
        examples = read_jsonl(path, LabeledExample)
    except Exception as exc:
        raise ModelConfigurationError(f"failed to load {split.value} training artifact", cause=exc) from exc
    wrong = [item.example_id for item in examples if item.split is not split]
    if wrong:
        raise ModelConfigurationError(
            f"{path} contains records outside the {split.value} split: {wrong[:3]}"
        )
    return examples


def supervised_messages(
    example: LabeledExample,
    *,
    prompt: Any,
    confidence: float,
) -> list[dict[str, str]]:
    """Build the one conversation that tokenization will encode exactly once."""

    result = TriageResult(
        **example.expected.model_dump(),
        confidence=confidence,
    )
    messages = render_iam_messages(example.ticket, prompt=prompt)
    assistant = json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return [*messages, {"role": "assistant", "content": assistant}]


def tokenize_examples(
    examples: Sequence[LabeledExample],
    *,
    tokenizer: Any,
    config: LoraTrainingConfig,
    prompt: Any,
) -> tuple[list[dict[str, list[int]]], int]:
    rows: list[dict[str, list[int]]] = []
    truncated = 0
    for example in examples:
        encoded = tokenize_response_only(
            tokenizer,
            supervised_messages(
                example,
                prompt=prompt,
                confidence=config.training_confidence,
            ),
            max_length=config.max_length,
            minimum_response_tokens=config.minimum_response_tokens,
        )
        rows.append(encoded.as_features())
        truncated += int(encoded.was_truncated)
    return rows, truncated


def count_parameters(model: Any) -> ParameterCounts:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return ParameterCounts(
        trainable=trainable,
        total=total,
        trainable_fraction=(trainable / total) if total else 0.0,
    )


def _resolved_revision(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _is_commit_like(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{7,64}", value))


def _require_resolved_identity(
    requested: str,
    resolved: str | None,
    *,
    artifact: str,
    required: bool,
) -> str:
    identity = resolved or requested
    if required and not _is_commit_like(identity):
        raise ModelConfigurationError(
            f"{artifact} revision {requested!r} did not resolve to an immutable commit hash"
        )
    return identity


def _training_arguments(
    TrainingArguments: Any,
    *,
    config: LoraTrainingConfig,
    checkpoint_dir: Path,
    precision: Any,
) -> Any:
    kwargs: dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "run_name": config.experiment_name,
        "seed": config.seed,
        "data_seed": config.seed,
        "num_train_epochs": config.epochs,
        "optim": config.optimizer,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.validation_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "max_grad_norm": config.max_grad_norm,
        "logging_steps": config.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": precision.bf16,
        "fp16": precision.fp16,
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": config.dataloader_num_workers,
        "dataloader_pin_memory": precision.device.startswith("cuda"),
        "save_safetensors": True,
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if "use_cpu" in parameters and precision.device == "cpu":
        kwargs["use_cpu"] = True
    return TrainingArguments(**kwargs)


def _json_scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float | int | str | bool | None]:
    output: dict[str, float | int | str | bool | None] = {}
    for key, value in metrics.items():
        if value is None or isinstance(value, (str, bool, int, float)):
            output[str(key)] = value
        elif hasattr(value, "item"):
            converted = value.item()
            if isinstance(converted, (str, bool, int, float)):
                output[str(key)] = converted
    return output


def run_lora_training(
    config: LoraTrainingConfig,
    *,
    project_root: Path | None = None,
) -> ExperimentManifest:
    """Run one immutable LoRA experiment with validation and evidence capture."""

    run_dir = config.run_directory
    if run_dir.exists():
        raise ModelConfigurationError(
            f"experiment directory already exists; choose a new experiment_name: {run_dir}"
        )
    if not config.train_data.is_file() or not config.validation_data.is_file():
        raise ModelConfigurationError("train and validation artifacts must exist before training")
    prompt = load_iam_prompt(config.prompt_path)
    train_examples = _require_split(config.train_data, DatasetSplit.TRAIN)
    validation_examples = _require_split(config.validation_data, DatasetSplit.VALIDATION)

    (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    ) = _require_training_stack()
    precision = choose_precision(
        torch,
        requested_device=config.device,
        requested_precision=config.precision,
    )
    set_deterministic_seeds(
        config.seed,
        torch=torch,
        deterministic_algorithms=config.deterministic_algorithms,
    )

    tokenizer_name = config.tokenizer_name_or_path or config.model_name_or_path
    tokenizer_revision = config.tokenizer_revision or config.model_revision
    common = {
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=tokenizer_revision,
            use_fast=True,
            **common,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            revision=config.model_revision,
            torch_dtype=precision.model_dtype,
            **common,
        )
    except Exception as exc:
        raise ModelLoadError("failed to load base training artifacts", cause=exc) from exc
    if not getattr(tokenizer, "is_fast", False):
        raise ModelConfigurationError("response-only masking requires a fast tokenizer")
    if tokenizer.eos_token_id is None:
        raise ModelConfigurationError("tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_revision = _require_resolved_identity(
        config.model_revision,
        _resolved_revision(getattr(base_model.config, "_commit_hash", None)),
        artifact="model",
        required=config.require_resolved_revision,
    )
    tokenizer_commit = None
    if isinstance(getattr(tokenizer, "init_kwargs", None), Mapping):
        tokenizer_commit = tokenizer.init_kwargs.get("_commit_hash")
    tokenizer_resolved = _require_resolved_identity(
        tokenizer_revision,
        _resolved_revision(tokenizer_commit),
        artifact="tokenizer",
        required=config.require_resolved_revision,
    )

    train_rows, train_truncated = tokenize_examples(
        train_examples,
        tokenizer=tokenizer,
        config=config,
        prompt=prompt,
    )
    validation_rows, validation_truncated = tokenize_examples(
        validation_examples,
        tokenizer=tokenizer,
        config=config,
        prompt=prompt,
    )
    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(config.lora.target_modules),
    )
    model = get_peft_model(base_model, lora_config)
    if config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.config.use_cache = False
    parameter_counts = count_parameters(model)
    if parameter_counts.trainable == 0:
        raise ModelConfigurationError("LoRA configuration selected no trainable parameters")

    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoints"
    adapter_dir = run_dir / "adapter"
    config_snapshot = run_dir / "config.json"
    config_snapshot.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    training_args = _training_arguments(
        TrainingArguments,
        config=config,
        checkpoint_dir=checkpoint_dir,
        precision=precision,
    )
    callbacks = []
    if config.early_stopping_patience:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_FeatureDataset(train_rows),
        eval_dataset=_FeatureDataset(validation_rows),
        data_collator=ResponseOnlyDataCollator(tokenizer, torch_module=torch),
        callbacks=callbacks,
    )
    train_result = trainer.train()
    final_validation = trainer.evaluate()
    if config.gradient_checkpointing:
        model.config.use_cache = True
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()

    losses = extract_loss_history(trainer.state.log_history)
    overfitting = detect_overfitting(losses)
    manifest_path = run_dir / "manifest.json"
    root = (project_root or Path.cwd()).resolve()
    metrics = _json_scalar_metrics(
        {**getattr(train_result, "metrics", {}), **final_validation}
    )
    manifest = ExperimentManifest(
        experiment_name=config.experiment_name,
        completed_at=utc_now(),
        config=config.model_dump(mode="json"),
        config_sha256=config.content_hash,
        dataset_sha256={
            "train": sha256_file(config.train_data),
            "validation": sha256_file(config.validation_data),
        },
        prompt_version=prompt.version,
        prompt_sha256=prompt.sha256,
        requested_model_revision=config.model_revision,
        resolved_model_revision=model_revision,
        requested_tokenizer_revision=tokenizer_revision,
        resolved_tokenizer_revision=tokenizer_resolved,
        seed=config.seed,
        parameter_counts=parameter_counts,
        hardware=collect_hardware_metadata(
            torch,
            device=precision.device,
            precision=precision.precision,
        ),
        git=collect_git_metadata(root),
        packages=collect_package_versions(),
        losses=losses,
        overfitting=overfitting,
        trainer_metrics=metrics,
        artifacts_sha256=artifact_hashes(run_dir, exclude=(manifest_path,)),
        truncated_training_examples=train_truncated,
        truncated_validation_examples=validation_truncated,
    )
    write_manifest_once(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = load_training_config(args.config, project_root=args.project_root)
    manifest = run_lora_training(config, project_root=args.project_root)
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
