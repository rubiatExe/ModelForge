"""Deterministic seed and precision/device decisions for training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

from modelforge.models.base import ModelConfigurationError


@dataclass(frozen=True, slots=True)
class PrecisionPlan:
    device: str
    precision: Literal["fp32", "fp16", "bf16"]
    model_dtype: Any
    fp16: bool
    bf16: bool


def choose_precision(
    torch: Any,
    *,
    requested_device: str,
    requested_precision: Literal["auto", "fp32", "fp16", "bf16"],
) -> PrecisionPlan:
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested_device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ModelConfigurationError("CUDA was requested for training but is unavailable")
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ModelConfigurationError("MPS was requested for training but is unavailable")

    precision = requested_precision
    if precision == "auto":
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            precision = "bf16"
        elif device.startswith("cuda"):
            precision = "fp16"
        else:
            # Trainer mixed precision has uneven CPU/MPS support; fp32 is the
            # conservative portable default for the reference experiment.
            precision = "fp32"
    if precision in {"fp16", "bf16"} and not device.startswith("cuda"):
        raise ModelConfigurationError(f"{precision} training requires a compatible CUDA device")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ModelConfigurationError("BF16 training was requested but is unsupported")
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]
    return PrecisionPlan(
        device=device,
        precision=precision,
        model_dtype=dtype,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
    )


def set_deterministic_seeds(
    seed: int,
    *,
    torch: Any,
    deterministic_algorithms: bool,
) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

