# ADR 0004: Deterministic-first evaluation

## Problem

Most IAM triage fields have closed labels and exact schema requirements, while
evidence relevance and unsupported implications retain limited semantic
judgment.

## Options considered

- LLM-as-a-judge for the whole response
- Deterministic metrics only
- Deterministic metrics plus a narrow, validated judge and human review

## Decision

Use deterministic validation and classification metrics for every closed or
programmatically checkable property. Restrict the optional LLM judge to evidence
relevance and unsupported claims, and validate it against human judgments.

## Reason

Exact labels, schema validity, and literal evidence grounding do not benefit
from a probabilistic judge. A narrow judge preserves semantic analysis without
letting provider noise decide regression gates.

## Evidence

The evaluation reference uses a single uncalibrated judge even for format and
exact properties. The useful-loop reference shows that naive lexical rubrics
are also gameable. Both motivate multiple explicit evaluator classes.

## Tradeoffs

Literal evidence checks miss paraphrase quality, while judge validation adds a
small manual-review burden.

## What would change the decision

A new output field whose correctness has no defensible deterministic metric,
paired with evidence that a versioned judge agrees with qualified humans at an
acceptable false-acceptance rate.
