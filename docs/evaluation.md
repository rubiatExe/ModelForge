# Evaluation protocol

## Purpose

ModelForge evaluation is designed to answer a selective-classification
question, not to manufacture one headline score. It measures model quality,
schema reliability, evidence grounding, operational failures, latency, cost,
calibration diagnostics, and routing behavior under explicit artifact
identities.

The current repository contains the evaluation implementation and tests, plus
two committed classical empirical results: the clean `issue_type` baseline and
a deterministic training-label-quality ablation. There is no committed
full-record base-model, LoRA, frontier, judge-validation, or routing-sweep
report. Neither classical result is LLM evidence.

## Populations and isolation

| Population | Cases | Role |
|---|---:|---|
| Train | 500 | Model fitting only |
| Validation | 100 | Prompt/model selection, calibration, and routing threshold selection |
| Hash-locked test | 200 | Final model comparison after decisions are fixed |
| Adversarial | 100 | Separately reported robustness slices |

The adversarial set covers `EMOTIONAL_URGENCY`, `AMBIGUOUS_REPORT`,
`CONFLICTING_EVIDENCE`, `MISLEADING_TERMINOLOGY`, `PROMPT_INJECTION`, and
`INCOMPLETE_INFORMATION`.

The loader verifies each split against the dataset manifest. Test loading also
verifies `test.lock.json`; training-purpose loading refuses test and adversarial
splits. Canonical input fingerprints detect exact normalized duplicates and
cross-split leakage.

The test set is hash-locked but publicly committed, not hidden. The classical
baseline has already been scored on it. New prompt, model, and routing choices
must therefore be made on validation only, and repeated test inspection must
not become an optimization loop.

The FastAPI evaluation endpoint rejects the `test` split unless
`MODELFORGE_ALLOW_TEST_EVALUATION=true` is explicitly set for a predeclared
final run. This guard does not make a committed test set secret; it prevents an
ordinary development request from silently turning it into tuning data.

## Deterministic record evaluation

Each case produces either a prediction or a typed runtime/provider error.
Errors are never rewritten into zero-quality predictions.

For returned output, the evaluator applies these checks in order:

1. Parse raw JSON when raw text is available. Surrounding prose, malformed
   JSON, and duplicate keys are invalid.
2. Validate the strict `TriageResult` contract. Extra fields, coercions,
   unknown enum values, non-finite confidence, and out-of-range confidence are
   invalid.
3. Compare the five categorical fields and evidence against gold.
4. Measure literal evidence grounding against subject plus body.
5. Aggregate overall and per-slice metrics.

The Hugging Face and frontier runtime adapters enforce step 4 earlier as well:
a schema-valid completion containing non-literal evidence is rejected at the
adapter boundary as a typed model-output failure rather than returned as a
`ModelPrediction`. The deterministic evaluator retains the grounding metric for
cached raw outputs, adapter-independent models, and explicit harness cases.

### Denominators

| Metric | Denominator |
|---|---|
| `error_rate` | All requested cases |
| `schema_validity_rate` | Returned predictions; runtime errors excluded |
| `exact_match_rate` | Returned predictions; invalid-schema returns count as not exact |
| Issue accuracy/F1/confusion matrix | Schema-valid predictions with issue labels |
| Field accuracy | Schema-valid predictions eligible for that field |
| Evidence presence | Schema-valid predictions with a gold evidence field |
| Literal grounding/unsupported evidence | Returned evidence snippets from eligible valid predictions |

These distinctions prevent an outage from masquerading as a model regression,
and prevent dropping errored cases from the reliability report. When comparing
systems, report both quality and error rate.

## Reported metrics

The full-record evaluator supports:

- JSON and schema validity;
- structured exact match;
- issue-type accuracy, macro precision/recall/F1, per-class F1, and confusion
  matrix;
- routing-team, severity, affected-scope, and recommended-action accuracy;
- evidence presence, literal grounding, and unsupported-evidence rate;
- per-slice results;
- latency and variable cost carried on each cached prediction.

The classical runner is intentionally narrower: it evaluates only
`issue_type`. Its perfect score must not be presented as full-record exact
match or evidence quality.

## Confidence and calibration

Model self-reported confidence is recorded but not treated as truth. The router
uses an inspectable composite of structural validity, literal evidence,
ambiguity signals, self-report, and optional label-logprob/sample-agreement
signals. Missing optional signals remain missing. The resulting value is a
routing score with `is_probability=false`.

Evaluation provides:

- fixed-width expected calibration error (ECE) as a diagnostic against
  empirical correctness;
- tie-stable risk/coverage curves that show accepted coverage and selective
  error as the score threshold changes.

Risk/coverage is the primary view. ECE does not turn an explicitly
non-probabilistic composite into a calibrated probability.

## Model judge and human validation

Closed labels, schema validity, exact values, and literal containment do not
need an LLM judge. The optional judge is restricted to two semantic questions:

- Is the selected evidence relevant to the prediction?
- Does the prediction imply an unsupported claim?

A judge result records a versioned provider/model/rubric identity and keeps
judge failures separate. Human-review queues can be exported and imported as
deterministic JSONL or CSV. Those queues contain source text by design and must
be handled as sensitive evaluation data, never as operational telemetry.

Before a judge affects a gate, validate it against completed human reviews and
report agreement, Cohen's kappa, relevance rank correlation, false acceptance,
false rejection, missing judgments, judge errors, and disagreement examples.
There is no committed judge or human-validation artifact today.

## Failure taxonomy

The evaluator can bucket cases into runtime error, timeout, schema invalid,
issue/routing/severity/scope/action mismatch, missing evidence, evidence
mismatch, non-literal evidence, multi-field mismatch, and other structured
mismatch. Representative artifacts omit raw ticket text. See
[Failure analysis](failure-analysis.md) for the current evidence status.

## Regression gate

A candidate cannot be gated until its run is proven comparable with the
baseline. Required controls include:

- completed runs;
- dataset name, version, split digest, case count, and case-ID digest;
- evaluator and schema versions;
- judge identity when a judge is used;
- matching report item population.

Model, prompt, and generation identities are recorded as intentional
experimental variables. After comparability passes, a versioned policy can
enforce maximum error rate; minimum schema/exact/aggregate metrics;
field-specific and per-class floors; slice floors; and maximum drops from the
baseline. Incomparable runs fail rather than producing a misleading delta.

The implementation and fixture behavior are tested, but no committed
candidate-vs-baseline regression-gate result exists.

## Routing evaluation

Routing sweeps operate on cached paired small/frontier outcomes so every
threshold sees the same generations. Calibration accepts only validation
cases. The declared sweep thresholds are:

`0.60, 0.70, 0.80, 0.85, 0.90, 0.95`

For each threshold, report issue macro F1, schema/exact reliability, frontier
share and calls per 100, average and weighted p95 end-to-end latency, average
variable cost, overconfident errors per 100, and slice metrics. Escalated
latency and cost are sequential:

`small request + frontier fallback`

The committed `routing_policy_v1.json` is an uncalibrated development policy
with no `validation_run_id`. It is neither a sweep result nor a production
threshold recommendation.

## Reproduction

Run all deterministic checks:

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest
```

Re-run the measured clean baseline without overwriting committed evidence:

```bash
python -m modelforge.experiments.run_classical \
  --output /tmp/modelforge-classical-result.json \
  --model-output /tmp/modelforge-classical-model.joblib
```

Re-run the measured label-quality stress test:

```bash
python -m modelforge.experiments.run_data_ablation \
  --noise-fractions 0 0.1 0.3 \
  --seed 20260820 \
  --output /tmp/modelforge-label-quality-ablation.json
```

Both runners verify the locked test before measurement. Compare dataset hashes,
configuration, counts, and classification metrics with the corresponding
committed artifact; latency, peak RSS, environment, and serialized model bytes
can vary by machine.

## Current evidence ledger

| Artifact | Status | Eligible claim |
|---|---|---|
| `classical_tfidf_logreg_v1.json` | Measured | Synthetic validation/test `issue_type` metrics and local single-request latency |
| `classical_label_quality_ablation_v1.json` | Measured, schema 1.1 | Sensitivity plus deterministic issue-only failure summaries for the synthetic classical `issue_type` task at 0%/10%/30% training-label corruption |
| `routing_policy_v1.json` | Uncalibrated configuration | Development wiring only |
| Golden/difficult evaluator tests | Harness fixtures | Evaluator semantics only |
| Base SLM result | Not present | None |
| LoRA result/adapter manifest | Not present | None |
| Frontier result | Not present | None |
| Judge/human validation result | Not present | None |
| Routing sweep result | Not present | None |
