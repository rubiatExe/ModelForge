# ADR 0001: Initial small-model candidate

## Problem

ModelForge needs a small open-weight causal model for a first controlled LoRA
experiment. The candidate must be small enough for accessible hardware, support
structured instruction following, and work with standard Transformers/PEFT.

## Options considered

- Qwen2.5 0.5B Instruct
- Qwen2.5 1.5B Instruct
- A larger 3B+ instruct model
- An encoder classifier only

## Decision

Use `Qwen/Qwen2.5-0.5B-Instruct` as the initial configurable candidate. Keep the
classical encoder baseline and do not assume the Qwen candidate will win.

## Reason

The 0.5B model keeps the first experiment feasible on modest CPU/MPS hardware
and is already compatible with the reference workflow's chat template and LoRA
target modules. A small first experiment makes pipeline errors cheaper to find.

## Evidence

The supplied training references target Qwen2.5 0.5B/1.5B and use standard
Transformers plus PEFT. ModelForge will accept the model only on measured
untouched-test quality, not on the reference projects' unreproducible reports.

## Tradeoffs

The model may be too small for robust evidence extraction or adversarial
instructions. A larger model may offer a better quality/cost frontier.

## What would change the decision

A controlled validation run showing the 0.5B candidate cannot clear slice
floors, or a similarly deployable candidate with materially better measured
quality at acceptable latency and memory.
