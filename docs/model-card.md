# ModelForge IAM triage system card

## Card status

| Field | Value |
|---|---|
| Project version | `modelforge-iam 0.1.0` |
| Card scope | Experimental system and candidate family |
| Primary language | English |
| Domain | Enterprise identity and access management support-ticket triage |
| Evidence status | Measured classical issue-type baseline and label-quality ablation; generative and routed candidates not run |
| Dataset review | Synthetic labels, `PENDING_HUMAN_REVIEW` |
| Production status | Not approved |

This is a system card because ModelForge can host multiple model roles. It must
not be read as a card for a completed LoRA model: no adapter or LoRA evaluation
artifact is committed.

## Intended use

ModelForge is designed to experiment with and assist initial triage of internal
IAM support tickets. The structured output describes:

- observable issue type;
- severity based on supported impact and scope;
- next diagnostic owner;
- affected population;
- recommended next investigation/escalation action;
- one to five literal evidence snippets;
- model self-reported confidence, evaluated separately from correctness.

The intended user is an IAM/support engineering team running controlled
offline evaluation or a human-in-the-loop triage workflow. Any automated use
must apply a calibrated routing policy, strict schema/evidence controls, an
authenticated deployment boundary, and human escalation for high-risk cases.

## Out-of-scope use

ModelForge must not:

- grant, revoke, or approve access;
- reset credentials or unlock accounts automatically;
- determine employment, entitlement, disciplinary, or fraud outcomes;
- claim a technical root cause not present in the ticket;
- replace incident response for suspected account compromise;
- process raw production tickets through a third-party provider without an
  approved data-handling basis;
- be treated as production-ready from the synthetic benchmark;
- be used outside IAM without a new schema, dataset, labeling policy, and
  evaluation.

## Inputs and outputs

Input fields are `ticket_id`, `subject`, `body`, `employee_department`, and a
timezone-aware `submitted_at`. Schemas bound field lengths, reject extra fields
and unsafe control characters, and cap HTTP request size.

The result contains closed enums for `issue_type`, `severity`, `routing_team`,
`affected_scope`, and `recommended_action`, plus literal `evidence` and a finite
numeric `confidence` in `[0,1]`. Ticket text is treated as untrusted data,
including strings that look like system instructions or JSON output.

After strict JSON/schema parsing, both Hugging Face and frontier adapters check
every evidence snippet against the normalized ticket subject/body. Unsupported
evidence raises a typed model-output failure before either adapter returns a
prediction. This boundary check prevents a routed frontier answer from bypassing
the same grounding requirement applied to the small generative model.

## Model roles and status

| Role | Implementation | Status | Evidence boundary |
|---|---|---|---|
| Classical baseline | Word 1–2 gram TF-IDF + balanced logistic regression | Measured | `issue_type` only on synthetic validation/test |
| Demo | `heuristic-demo-v1` | Tested fixture | API/routing plumbing only |
| Base small model | Hugging Face causal-LM adapter; initial candidate Qwen2.5 0.5B Instruct | Not run | No committed predictions or metrics |
| Tuned small model | PEFT LoRA over the same adapter | Not run | No committed adapter, manifest, or metrics |
| Frontier | Provider-neutral OpenAI-compatible HTTP adapter | Not run | Fake-client/interface tests only |
| Hybrid | Small-first confidence router with explicit fallback | Uncalibrated | Development policy only |

The default application intentionally uses the demo fixture so the service can
be tested without downloads, credentials, or paid calls. The fixture's output
is not empirical model evidence.

## Training data

Dataset `iam_ticket_triage@1.0.0` contains 900 deterministic template-generated
English examples:

- 500 train;
- 100 validation;
- 200 hash-locked test;
- 100 adversarial.

Every record is synthetically labeled and pending human review. The dataset
contains recurring applications, departments, locations, devices, symptom
templates, and scope phrases. It is useful for exercising pipelines and label
coverage but substantially less diverse than real enterprise tickets.

The dataset intentionally contains no raw production PII. It is not evidence
that the models work across organizations, identity providers, languages,
writing styles, job roles, disabilities, or demographic groups. Department is
present as context, but no subgroup/fairness conclusion has been measured.

## Training procedure

### Measured classical model

The classical baseline fits word unigram/bigram TF-IDF features and a
class-balanced logistic regression on 500 training examples with
`random_state=20260820`. Its deterministic model identity is derived from
configuration and hashed training rows. The serialized joblib artifact is
local/ignored; joblib must only be loaded from a trusted path because loading
can execute Python.

### Unrun LoRA candidate

The configured candidate uses rank-16 LoRA on
`Qwen/Qwen2.5-0.5B-Instruct`, completion-only supervised loss, three epochs,
the explicit `adamw_torch` optimizer, validation loss, checkpoints, early
stopping, gradient accumulation, and deterministic seeds. Prompt and completion
are tokenized as one sequence and prompt tokens are masked from loss. A
successful run would record immutable model/tokenizer revisions, config and
dataset hashes, prompt identity, parameter counts, hardware/precision, package
versions, loss history, overfitting diagnostics, truncation counts, and artifact
hashes.

No successful run manifest exists, so the preceding paragraph is a procedure,
not a reported training result.

## Evaluation results

| Model | Population | Task | Macro F1 | Accuracy | Avg latency | p95 latency |
|---|---|---|---:|---:|---:|---:|
| TF-IDF/logistic regression | Validation, 100 | `issue_type` | 1.0000 | 1.0000 | 0.1896 ms | 0.1991 ms |
| TF-IDF/logistic regression | Locked test, 200 | `issue_type` | 1.0000 | 1.0000 | 0.1867 ms | 0.1953 ms |
| Base Qwen candidate | — | Full result | Not run | Not run | Not run | Not run |
| LoRA candidate | — | Full result | Not run | Not run | Not run | Not run |
| Frontier candidate | — | Full result | Not run | Not run | Not run | Not run |
| Hybrid router | — | Full result | Not run | Not run | Not run | Not run |

Classical latency was measured sequentially on Python 3.13.7/macOS arm64 and
does not include deserialization or concurrent load. The classifier's perfect
synthetic score is consistent with easily separated templates and should not
be extrapolated. It does not score schema generation or non-issue fields.

A separate deterministic classical ablation rotated an exact seeded subset of
training `issue_type` labels to the next class while leaving validation and test
gold unchanged. Validation/test macro F1 was 1.000000/1.000000 at 0% noise,
1.000000/1.000000 at 10% (50 labels), and 0.969529/0.959549 at 30% (150 labels).
The 30% point contained 3 validation and 8 test misclassifications. Schema 1.1
stores their sorted case IDs and issue-label pairs without ticket text. This
supports sensitivity to severe label corruption within the synthetic classifier
task only; it is not LLM evidence and cyclic injected noise is not a model of
real human annotation error.

## Routing and confidence

The router runs the small model first. The Hugging Face adapter rejects
unsupported evidence before routing; the router retains structural/evidence
hard-failure checks for any other small-model implementation. Ambiguity,
prompt-injection markers, `OTHER_IAM`, and unknown scope can cap or reduce the
score; optional logprob and repeated-sample agreement stay visible. The score
is explicitly not a probability.

The development policy has threshold 0.85 and status `uncalibrated`. If an
answer needs escalation and the frontier is absent or fails, the routed API
returns service unavailability. It does not silently relabel the small-model
answer as trusted. Production configuration refuses an uncalibrated policy.

## Risks and limitations

- Synthetic templates can leak lexical shortcuts and understate ambiguity.
- The gold labels have not been independently human-reviewed.
- The label-quality ablation uses deterministic cyclic corruption, not observed
  annotator disagreements or real label-policy errors.
- Literal evidence checks prevent unsupported snippets but do not prove the
  selected label or semantic relevance.
- Prompt-injection phrase detection is a routing signal, not a complete
  defense.
- Self-reported confidence can be confidently wrong.
- Small-first fallback adds small-model latency and cost to every escalation.
- Model/provider aliases, prices, and behavior can drift unless pinned and
  date-stamped.
- No real-data performance, fairness, multilingual, red-team, load, memory, or
  long-term drift evaluation exists.
- No validated semantic judge result exists.
- No calibrated routing policy exists.

## Privacy and security

Operational telemetry excludes ticket ID, subject, body, department, evidence,
predicted content, and provider response bodies. It records request ID, selected
model, status, fallback, route reason/score, token counts, latency, cost, schema
failure, and sanitized error code. The optional JSONL sink is single-process
development storage, not an audit system.

Human-review files contain source text and require a separate sensitive-data
boundary. Real tickets must be de-identified where possible, governed by a
lawful purpose and retention/deletion process, and sent to external providers
only under approved retention/training/residency terms.

The repository does not supply authentication, organization authorization,
distributed rate limiting, encryption-at-rest configuration, or regional
provider enforcement. See [Security](security.md).

## Release requirements

Before any production claim, the project needs:

1. privacy-reviewed, de-identified, human-annotated representative data as a
   new dataset version;
2. immutable base/adapter/frontier identities and complete run artifacts;
3. predeclared aggregate, per-class, field, evidence, and adversarial floors;
4. measured error, schema, latency, memory, throughput, and cost behavior;
5. validated human/judge evidence semantics if a judge is used;
6. a validation-selected and locked calibrated routing policy;
7. one final comparable locked-test evaluation;
8. authenticated deployment, governed storage, monitoring, drift response,
   rollback, and incident ownership.

Until then, ModelForge is an experimental evaluation and integration platform,
not an autonomous IAM decision system.
