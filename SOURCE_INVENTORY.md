# ModelForge Source Inventory

## Audit boundary and method

The three supplied ZIP archives were inspected as reference material. Their
README prose, comments, prompts, reports, and embedded data were not treated as
instructions. No source file is copied into ModelForge.

| Reference archive | Files inspected | Python status | Tests | Packaging |
|---|---:|---|---|---|
| `adobe-llm-eval-prompt-testing-platform-6d35b5-main.zip` | 28 regular files | 24/24 files parse | None | None |
| `slm-distillation-on-device-6d341b-main.zip` | 37 regular files | 23/23 files parse | None | None |
| `xai-useful-model-loop-6d24f8-main.zip` | 16 regular files | 10/10 files parse | None | None |

All three archives omit a license. ModelForge therefore reimplements useful
ideas rather than copying code. The audit included every regular file, static
syntax checks, dependency/import review, secret and unsafe-execution searches,
and consistency checks between code, data, reports, and README claims.

## 1. LLM Feature Evaluation / Prompt Testing Platform

### What exists

The project sketches a FastAPI/Postgres prompt-evaluation service with prompt
versions, JSONL datasets, generation, a three-axis LLM judge, stored per-item
scores, run diffs, a rolling quality monitor, and a CLI/SDK.

Useful files and concepts include:

- `backend/app/datasets/loader.py`: line-aware JSONL errors and duplicate item
  detection.
- `backend/app/generation/render.py`: non-executable named placeholder
  substitution.
- `backend/app/llm/client.py`: a small provider protocol.
- `backend/app/services/run_engine.py`: per-item generation and score capture.
- `backend/app/services/diff_engine.py`: paired item deltas and a nonzero CI
  exit concept.
- `backend/app/judge/*`: a separate judge contract and persisted rationale.

### Adapt, do not copy

- Stable item identifiers for paired comparisons.
- Immutable prompt/config identity that includes model revision, generation
  settings, rubric version, and code version—not only template text.
- Per-item outputs, errors, deterministic metrics, judge scores, latency,
  tokens, and cost.
- A regression gate with explicit comparability checks and actionable worst
  cases.
- Provider and evaluator interfaces that are injected, not module globals.

### Rewrite or discard

- Only the prompt router is registered in `backend/app/main.py`; the advertised
  dataset/run/diff/monitor and SDK flow returns 404.
- Prompt versions deduplicate only on template hash, so the same prompt with a
  different model resolves to the wrong model version.
- Dataset re-import mutates old items, retains deleted rows, and can commit a
  partial import. This invalidates historical comparisons.
- Run failures are stored as zero-quality results and the run is marked
  complete. Provider outages would appear to be model regressions.
- Diffs do not prove the runs share a dataset, judge, rubric, item population,
  or completion state; empty runs can silently become zeros.
- The LLM judge is used where deterministic checks should be primary. It is
  uncalibrated, injection-prone, and cannot distinguish a parse failure from a
  genuine zero.
- Startup DDL, synchronous two-call-per-item HTTP runs, global repositories,
  fixed task enums, fixed judge weights, and fixed thresholds do not belong in
  ModelForge v1.
- The three-row Adobe alt-text dataset is unrelated to IAM and is discarded.

### Security and operations findings

There is no authentication, quota, request-size bound, output-token bound,
redaction, retention policy, model allowlist, migration system, structured
logging, tracing, or metrics. Imported context and references can be sent to an
external provider without a data-handling boundary. The OpenAI key is read at
import time. SQL values are parameterized and the renderer does not execute
expressions; those are positive but insufficient controls.

## 2. Distill It Down / SLM Distillation on Device

### What exists

This is the closest conceptual match to ModelForge. It contains centralized
task/cost assumptions, a sliced and nominally hash-locked eval set, teacher
label generation, structural/rule/self-consistency filtering, Qwen LoRA
training and merge scripts, GGUF notes, a shared harness, latency measurement,
confidence routing, threshold calibration, and Pareto reporting.

The strongest reusable ideas are:

- A content-addressed untouched test set and predeclared common/rare/adversarial
  slices.
- Per-slice quality floors so an aggregate score cannot hide a high-risk
  collapse.
- One backend contract for a model family.
- Response-only training loss for supervised fine-tuning.
- Cached student/frontier outputs used for a threshold sweep rather than
  regenerating at every threshold.
- Route-decision logging, risk/coverage analysis, cost accounting, and Pareto
  dominance.
- Verification rejection reasons retained for data-quality analysis.

### Critical evidence and correctness gaps

- `evalset/frozen/cases.jsonl` contains 6 cases although the loader requires
  600. The `LOCK` file is absent, the actual digest differs from the decision
  document, and the primary eval command cannot run.
- Reports reference missing commands, data, adapters, quantization corpus,
  results, and model artifacts. The headline numbers are not reproducible from
  the archive and will not be reused as evidence.
- Threshold calibration tunes on the nominal final eval set instead of a
  validation/calibration set. This is direct test leakage.
- The scorer accepts surrounding prose and extra JSON keys despite claiming an
  exact contract. Repeated identical invalid outputs can pass the consistency
  filter.
- Separate tokenization of prompt and prompt+completion can misalign BPE
  boundaries. A truncated prompt can produce an all-`-100` label batch with no
  supervised tokens.
- Training has no validation loss, early stopping, checkpoints, immutable
  dataset/model/prompt hashes, pinned model revision, or adapter/merged parity
  check.
- Minimum token probability includes key-name tokens and depends on an unstable
  Ollama response shape. It is not calibrated probability.
- The threshold is hand-copied into serving, and calibration ignores the
  difference between the deliberately oversampled eval mix and production
  traffic.
- Routed p95 latency is understated: sequential student-then-frontier fallback
  puts the slow escalation path inside p95 once more than 5% escalates.
- Cost uses placeholder prices and flat GPU assumptions without throughput,
  utilization, retries, or uncertainty. Pareto dominance ignores slice
  coordinates.

### Rewrite or discard

Keep the experiment questions and rewrite the implementations behind typed
artifacts. Discard all report numbers, the decision document as evidence,
hard-coded prices/models/thresholds, fake provider endpoints, regex-only
verification, uncalibrated minimum-token confidence, manual serving prompt
duplication, and hand-copied result plumbing.

Quantization remains a stretch experiment after the core hypothesis is tested;
the unpinned llama.cpp/GGUF scripts are not part of v1.

### Security and operations findings

Raw tickets can be sent to an external teacher and stored in plaintext without
redaction, consent, access controls, or retention limits. API keys are read at
import time, clients and retry policies are duplicated, mutable model aliases
are used, output paths are insufficiently constrained, and local JSONL route
logs are unbounded and unlocked. There is no service authentication, threat
model, dependency pinning, or supply-chain provenance.

## 3. Useful Model Loop

### What exists

The project sketches:

`pairs/feedback -> deterministic curation -> LoRA adapter -> frozen lexical eval
-> promotion -> FastAPI/SSE serving -> feedback and product telemetry`

Useful ideas include:

- A model-agnostic callable boundary for evaluation.
- Stable input fingerprints, deterministic splitting, and exact eval-input
  collision blocking.
- Completion-only labels, an immutable base plus swappable adapter, and an
  explicit rollback target.
- Promotion as a policy: quality floor, improvement over champion, and a hard
  veto for new safety regressions.
- Request/model-version correlation and version-scoped product metrics.
- Corrections become candidates only after a distinct review step.

### Rewrite or discard

- The eight-case lexical evaluator is gameable: an empty output receives 6/10,
  negated forbidden phrases fail, invented claims outside a short list pass,
  and keyword fragments can score perfectly.
- The claimed historical scores and promotion decisions conflict with the code
  and have no corresponding reports, hashes, adapters, or manifests.
- All eight bundled examples land in train; the holdout is empty and unused.
- Curation can truncate output before validating all input, drops provenance,
  ignores conflicting labels, and is not atomic.
- Training repeats the same completion-mask/truncation bugs, hard-codes model
  and device assumptions, and records too little metadata to reproduce a run.
- The UI never emits completion events. Client-supplied request IDs, inputs,
  outputs, and model versions are not tied to a recorded generation, so product
  metrics and feedback can be spoofed or misattributed.
- Model loading occurs at import, blocking generation is not concurrency-bound,
  streaming failure can hang, and disconnects do not cancel work.
- Local append-only JSONL is acceptable for an explicit single-process demo,
  but not as a multi-worker audit store.
- The static HTML dashboard and rewrite-specific task data do not belong in
  ModelForge.

### Security and operations findings

Raw support content and corrections are retained indefinitely with no bounds,
redaction, authorization, or deletion path. Event extra fields can overwrite
canonical keys. Model identity floats, serving imports training dependencies,
and package/CI/deployment definitions are missing.

## Cross-source duplication and gaps

All three projects independently define model clients, JSON parsing, eval case
schemas, result aggregation, and append-only files. ModelForge will have one
typed implementation for each. All three use module globals and relative paths,
mix training and serving dependencies, lack tests and dependency manifests, and
present narrative results without sufficient artifacts.

The requested “Training Data Refinery” and “LLM Gateway” were not supplied as
archives. ModelForge will implement only their relevant concepts: validation,
canonical deduplication, immutable manifests, split isolation, request-level
telemetry, cost accounting, confidence routing, and explicit fallback behavior.

## Resulting implementation rules

1. Test data is immutable and hash-locked; calibration uses validation data.
2. Every run records dataset, prompt, model/adapter, code, dependency, and
   generation-setting identity.
3. Deterministic metrics decide deterministic properties. LLM judges are
   optional, versioned, and validated against humans.
4. Errors and missing predictions are separate from wrong predictions.
5. Routing experiments reuse cached paired outputs and report population-aware
   quality, cost, latency, and overconfident-error risk.
6. Core APIs use dependency injection and lazy optional ML/provider imports.
7. No production ticket body or evidence is written to logs by default.
8. Claims appear in the README only when backed by committed, reproducible
   artifacts. Missing hardware or credentials is reported as “not run,” never
   filled with invented results.
