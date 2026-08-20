# ModelForge Implementation Plan

## Product question

Can a LoRA-tuned open-weight model handle most enterprise IAM ticket triage at
an acceptable quality floor, escalating uncertain cases to a frontier model at
materially lower cost?

ModelForge v1 is a modular monolith built to answer that question. It is not a
generic ML platform, agent framework, or enterprise gateway.

## Scoped architecture

```text
versioned JSONL + manifest + test lock
                  |
                  v
       train / validation / test
          |          |       |
       TF-IDF     base SLM  frontier
       baseline      |       baseline
                     v
                  LoRA SLM
          \          |          /
           deterministic evaluation
            + optional judge/human
                     |
              cached predictions
                     |
             threshold sweep/gate
                     |
       FastAPI -> router -> SLM/frontier
                     |
              privacy-safe telemetry
```

Python protocols and Pydantic schemas are the seams. Heavy training libraries
and frontier credentials are optional extras and are loaded only by the
corresponding adapter. The default repository remains testable on a CPU without
a provider key.

## Phase 0: analysis and decisions

- Commit `SOURCE_INVENTORY.md` before implementation.
- Record the plan in this file.
- Add lightweight ADRs for the initial model candidate, LoRA, dataset design,
  deterministic-first evaluation, and confidence routing.
- Reimplement concepts cleanly because the references have no license.

Exit check: inventory covers every supplied archive and identifies reuse,
rewrite, discard, security, dependency, assumption, and test findings.

## Phase 1: domain, dataset, and classical baseline

Build:

- Strict Pydantic input/output schemas and closed IAM enums.
- A deterministic, synthetic seed benchmark with exactly 500 train, 100
  validation, 200 untouched test, and 100 adversarial examples.
- Provenance and review fields on every example plus a versioned manifest with
  distributions and SHA-256 digests.
- Canonical Unicode/whitespace normalization, exact duplicate detection,
  conflicting-label detection, split leakage checks, and atomic writes.
- A test-set lock checked by every test evaluation.
- `docs/LABELING_GUIDE.md` with scope/impact-based severity and evidence rules.
- A TF-IDF + logistic-regression baseline. Categorical fields beyond
  `issue_type` may be measured, but issue type is the required classical task.

Exit checks:

- Dataset generation is deterministic and idempotent.
- Training loaders reject test/adversarial data.
- Manifest counts and hashes match files.
- Schema, preprocessing, leakage, and baseline tests pass.
- A real local classical result is stored with its config and runtime metadata.

## Phase 2: model adapters and fine-tuning

Build:

- One async `TriageModel` interface returning validated output plus latency,
  token, cost, and uncertainty metadata.
- A versioned structured prompt that treats ticket text as untrusted data.
- A Hugging Face base-model adapter and the same adapter with an optional PEFT
  checkpoint.
- A provider-neutral frontier interface with one OpenAI-compatible HTTP adapter
  and injected fakes for tests.
- A readable LoRA training path with response-only loss constructed from one
  tokenized chat sequence, validation loss, checkpoints, seeded runs, trainable
  parameter counts, hardware/precision selection, and overfitting warnings.
- An immutable run manifest containing config hash, dataset hashes, model and
  tokenizer revisions, prompt hash, code SHA, packages, losses, and artifacts.

Initial controlled candidate: `Qwen/Qwen2.5-0.5B-Instruct`, chosen as a small
CPU/MPS-accessible reference candidate—not assumed to be the winner. The
experiment config can change the model without changing training code.

Exit checks:

- Prompt/completion masking has supervised tokens and preserves sequence
  alignment under truncation.
- Optional ML imports fail with actionable install guidance.
- Base, adapter, and frontier outputs cross the same strict schema boundary.
- Fake-provider fallback behavior is covered end to end.

## Phase 3: evaluation and regression gates

Build deterministic evaluators for:

- Raw JSON and Pydantic schema validity.
- Exact structured-record match.
- Issue-type accuracy, precision, recall, macro/per-class F1, and confusion
  matrix.
- Routing-team, severity, scope, and action accuracy.
- Evidence presence and unsupported-evidence rate.
- Calibration (ECE and risk/coverage) without calling self-confidence a
  probability.

Build additional mechanisms for:

- Versioned model-based judging only for evidence relevance/unsupported claims.
- Human-review export/import and a gold/human/judge/model comparison table.
- Judge validation: agreement, Cohen's kappa, rank correlation, false
  acceptance, false rejection, and disagreement examples.
- A failure taxonomy and representative example report.
- A regression gate that first proves run comparability, then enforces schema,
  aggregate, per-class, and adversarial-slice floors.

Exit checks:

- Fixed golden and difficult cases catch known regressions.
- Provider errors remain errors, not zero-scored model answers.
- CLI exits nonzero on a policy failure.

## Phase 4: routing, economics, and latency

Build:

- A composite uncertainty signal containing separately visible structural,
  evidence, ambiguity, optional logprob, and self-report components.
- Explicit fail-closed escalation. If the frontier is unavailable, the response
  is marked untrusted internally and the API reports service unavailability;
  it never silently labels the SLM answer as trusted.
- Cached paired student/frontier predictions.
- Sweeps at `0.60, 0.70, 0.80, 0.85, 0.90, 0.95`.
- For every threshold: macro F1, frontier share/calls per 100, average and p95
  end-to-end latency (including sequential fallback), average cost,
  overconfident wrong answers, and slice metrics.
- A quality-vs-cost artifact and decision-ready threshold recommendation.

Exit checks:

- Thresholds are loaded from a versioned policy artifact, not copied into code.
- Calibration uses validation, and final metrics use the untouched locked test.
- Pricing source/date and workload assumptions are explicit.

## Phase 5: API and telemetry

Expose:

- `POST /v1/triage`
- `POST /v1/models/{model}/triage`
- `POST /v1/evaluations/run`
- `GET /v1/evaluations/{run_id}`
- `GET /v1/models`
- `GET /health`

Add:

- Lifespan-based lazy initialization, dependency injection, typed errors, and
  bounded asynchronous evaluation work.
- Request size and field length limits, provider deadlines, and a model
  allowlist.
- Structured logs and append-only single-process demo telemetry containing only
  request IDs and operational fields—not ticket identity, text, or evidence.
- Request count, selected model, fallback, input/output tokens, average/p50/p95
  latency, error/schema-failure rate, routing rate, and estimated cost.

The local JSONL telemetry store is explicitly a development backend. A durable
production sink can implement the same interface without changing routing.

Exit checks:

- API -> model -> validated response, evaluation run, and fallback integration
  tests pass.
- Oversized requests and unconfigured escalation are safe and explicit.
- No secrets or ticket bodies appear in logs or committed artifacts.

## Phase 6: evidence and documentation

- Run the complete test suite and the classical experiment locally.
- Run lightweight deterministic routing fixtures to validate reporting, clearly
  labeled as harness fixtures rather than model evidence.
- Run base/frontier/LoRA experiments only when model artifacts, suitable
  hardware, and provider credentials are available. Never fabricate these
  cells.
- Generate machine-readable JSON plus Markdown/SVG summaries from artifacts.
- Document architecture, dataset, evaluation, experiments, failure analysis,
  model card, decisions, interview questions, limitations, and what failed.
- Add a CI software-test job and a separate artifact-based evaluation gate.

## Definition-of-done accounting

The repository can implement and verify all software paths without external
credentials. Claims that require a downloaded model, a completed fine-tune, or
paid frontier calls become complete only after their immutable run artifacts
exist. README status will distinguish:

- **measured**: produced by a committed command and artifact;
- **harness fixture**: deterministic plumbing verification, not model evidence;
- **not run**: blocked by absent model/hardware/credentials.

This distinction preserves the core purpose of ModelForge: defensible evidence
rather than impressive-looking invented numbers.
