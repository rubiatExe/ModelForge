# ADR 0002: LoRA rather than full fine-tuning

## Problem

The narrow IAM task needs model adaptation while keeping training, rollback,
storage, and explanation manageable for a v1 project.

## Options considered

- Full-parameter supervised fine-tuning
- LoRA/PEFT supervised fine-tuning
- Prompt-only adaptation
- Preference optimization or RLHF

## Decision

Use supervised LoRA via PEFT, with completion-only loss and an immutable base
model revision. Do not add RLHF, DPO, PPO, or a reward model.

## Reason

LoRA reduces trainable parameters and produces a small, swappable adapter while
still testing the hypothesis that task-specific adaptation improves a small
model. It makes rollback and experiment comparison straightforward.

## Evidence

Both model-training references use LoRA, but their runs omit important
reproducibility and validation controls. ModelForge retains the method and
rewrites the data, masking, validation, checkpoint, and manifest paths.

## Tradeoffs

LoRA capacity may be insufficient, and adapter serving adds a small amount of
operational complexity. Full tuning could eventually improve quality.

## What would change the decision

Repeated controlled LoRA experiments that plateau below required slice floors
while full fine-tuning is feasible and demonstrably improves the same locked
evaluation without unacceptable cost.
