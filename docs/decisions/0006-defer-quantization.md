# ADR 0006: Defer quantization until the fine-tuned model clears quality gates

## Problem

Quantization may reduce model size and latency, but it introduces another model
artifact and another source of quality change. ModelForge does not yet have a
completed, measured LoRA run to optimize.

## Options considered

- Add FP16, INT8, and INT4 conversion before the first controlled training run
- Copy the reference repository's GGUF commands and reported numbers
- Defer quantization until the unquantized adapter clears validation and locked-test gates

## Decision

Defer quantization in v1's current evidence state. First measure the pinned base
model and LoRA adapter. Only then compare FP16, INT8, and INT4 on the same locked
cases, serving contract, load profile, hardware, and evaluator version.

## Reason

Optimizing an unproven model would add complexity without answering the central
quality question. The supplied quantization reference omits required corpora,
tool revisions, hashes, and comparable routed-system measurements, so its
numbers cannot be reused as evidence.

## Evidence

The classical and label-quality experiments are reproducible, but no local
open-weight weights or adapter artifact is currently present. A quantization
result would therefore be either fabricated or disconnected from ModelForge's
actual training run.

## Tradeoffs

The first deployable adapter may consume more memory and run more slowly than a
later quantized form. Deferral keeps the initial experiment interpretable and
prevents artifact proliferation.

## What would change the decision

A completed LoRA run that clears the quality gate and has a reproducible
unquantized serving benchmark. At that point, quantization becomes the next
bounded experiment rather than speculative infrastructure.
