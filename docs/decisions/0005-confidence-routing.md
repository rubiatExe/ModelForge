# ADR 0005: Calibrated composite routing signal

## Problem

The small model must escalate uncertain tickets, but its self-reported
`confidence` is not a calibrated probability and token logprobs may be absent or
misleading.

## Options considered

- Route only on self-reported confidence
- Route only on minimum token probability
- Route on deterministic ambiguity rules
- Use a visible composite of available signals and calibrate its threshold

## Decision

Compute an inspectable composite from schema/evidence validity, ambiguity
heuristics, optional label logprob, repeated-sample agreement, and self-report.
Missing signals do not masquerade as measured probabilities. Sweep thresholds
on validation data and lock the selected policy before untouched-test scoring.

## Reason

Any single signal can be confidently wrong. Keeping components visible allows
ablation, calibration measurement, and safe failure behavior. Cached paired
outputs make the business tradeoff reproducible.

## Evidence

The distillation reference multiplies structure, a regex rule, and minimum
token probability, but its implementation includes key tokens, depends on an
unstable serving schema, and calibrates on final eval data. The architectural
idea is useful; the particular score is not.

## Tradeoffs

A hand-designed composite may be conservative and can drift as ticket mix
changes. Repeated sampling increases latency. A learned calibrator needs more
held-out data.

## What would change the decision

Enough representative calibration data to train and validate a secondary
calibrator or selective classifier that produces materially better risk/
coverage behavior and remains stable under drift monitoring.
