# ADR 0007: Keep the routing threshold uncalibrated until paired model evidence exists

## Problem

The service needs a development threshold to exercise fallback behavior, but a
production threshold can only be selected from paired small/frontier outputs on
an isolated calibration population.

## Options considered

- Declare the development value of 0.85 production-ready
- Tune against the untouched test set
- Keep a visibly uncalibrated development policy and require a validation sweep

## Decision

Keep `routing_policy_v1.json` and the embedded development policy explicitly
`uncalibrated`. Sweep 0.60, 0.70, 0.80, 0.85, 0.90, and 0.95 on cached validation
outputs, select a policy from stated quality/cost constraints, then evaluate
that frozen policy once on the locked test set. Production startup rejects an
uncalibrated policy.

## Reason

No provider credentials or paired model artifacts are available in the current
environment. Choosing a threshold now would turn a harness constant into a
false empirical claim. Test-set tuning would also violate the isolation rule.

## Evidence

The routing evaluator accounts for sequential small-then-frontier latency,
population weights, cost, frontier share, and overconfident errors. Unit tests
exercise that logic, but fixtures are not a substitute for measured model
outputs.

## Tradeoffs

Development routing can be demonstrated, but the service intentionally reports
degraded health and cannot claim a validated quality/cost operating point.

## What would change the decision

Comparable full validation artifacts from the selected LoRA and frontier
models, with immutable model, prompt, dataset, pricing, and evaluator identities.
