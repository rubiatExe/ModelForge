# Failure analysis

## Current empirical boundary

There is not yet a real full-record model failure corpus in this repository.
The measured TF-IDF/logistic-regression `issue_type` baseline recorded no
mistakes on the 100-case synthetic validation split or 200-case hash-locked
synthetic test split. A separate controlled label-quality ablation also
recorded no mistakes at 10% injected training-label noise, then fell to
validation/test macro F1 of 0.969529/0.959549 at 30% noise.

That does **not** mean ModelForge observed no failures:

- the classical experiment did not evaluate the other five output properties;
- it did not evaluate the adversarial split;
- it did not test raw JSON/schema generation;
- it did not measure evidence semantics or grounding;
- it did not exercise provider/runtime failures;
- the repeated template vocabulary makes the classification task unusually
  easy.

At 30% noise, the test errors were concentrated in three confusions: four
`ACCESS_DENIED_AFTER_LOGIN` cases predicted as
`PROVISIONING_OR_SYNC_FAILURE`, one `ACCOUNT_LOCKED_OR_DISABLED` case predicted
as `ACCESS_DENIED_AFTER_LOGIN`, and three `PROVISIONING_OR_SYNC_FAILURE` cases
predicted as `OTHER_IAM`. The controlled degradation shows label quality can
matter, while the clean and 10% results still show that this dataset version is
too lexically separable to estimate field performance.

Artifact schema 1.1 records 3 validation and 8 test misclassifications at the
30% point. Examples are sorted by case ID and capped at ten per split; because
both counts are below ten, the committed artifact contains every failure:

| Split | Case ID | Expected `issue_type` | Predicted `issue_type` |
|---|---|---|---|
| Validation | `MF-VA-0010` | `ACCESS_DENIED_AFTER_LOGIN` | `PROVISIONING_OR_SYNC_FAILURE` |
| Validation | `MF-VA-0030` | `OTHER_IAM` | `SSO_AUTHENTICATION_FAILURE` |
| Validation | `MF-VA-0090` | `OTHER_IAM` | `SSO_AUTHENTICATION_FAILURE` |
| Test | `MF-TE-0011` | `PROVISIONING_OR_SYNC_FAILURE` | `OTHER_IAM` |
| Test | `MF-TE-0040` | `ACCESS_DENIED_AFTER_LOGIN` | `PROVISIONING_OR_SYNC_FAILURE` |
| Test | `MF-TE-0046` | `ACCESS_DENIED_AFTER_LOGIN` | `PROVISIONING_OR_SYNC_FAILURE` |
| Test | `MF-TE-0076` | `ACCESS_DENIED_AFTER_LOGIN` | `PROVISIONING_OR_SYNC_FAILURE` |
| Test | `MF-TE-0083` | `PROVISIONING_OR_SYNC_FAILURE` | `OTHER_IAM` |
| Test | `MF-TE-0101` | `PROVISIONING_OR_SYNC_FAILURE` | `OTHER_IAM` |
| Test | `MF-TE-0130` | `ACCESS_DENIED_AFTER_LOGIN` | `PROVISIONING_OR_SYNC_FAILURE` |
| Test | `MF-TE-0147` | `ACCOUNT_LOCKED_OR_DISABLED` | `ACCESS_DENIED_AFTER_LOGIN` |

These representative rows contain only case IDs and issue labels, never ticket
text. They are appropriate for a committed failure summary but are not a
substitute for access-controlled source-text review.

The ablation is classical data evidence only. Its seeded cyclic label rotation
does not reproduce human disagreement, correlated annotation errors, or real
distribution shift, and it does not evaluate full triage or LLM behavior.

## Harness failures are not model observations

The golden evaluator test contains four deliberately constructed cases:

1. one correct calm large-outage result;
2. an anger-driven severity error;
3. an unsupported evidence claim;
4. a simulated provider timeout.

The fixture asserts one operational error, two wrong returned predictions, and
one exact match. It verifies denominators and taxonomy behavior. It does not
represent a sample from any deployed, base, LoRA, or frontier model.

Other unit and integration fixtures intentionally create malformed JSON,
duplicate keys, schema errors, low-confidence fallback, missing frontier
capacity, oversized requests, and judge disagreements. Their pass/fail status
is software-path evidence only.

## Predeclared taxonomy

Future empirical reports should bucket every non-exact case under one or more
of these categories:

| Category | Meaning | Primary response |
|---|---|---|
| `prediction_error` | Model/provider returned no usable outcome | Track separately from quality; inspect provider/model reliability |
| `timeout` | Declared deadline expired | Inspect load, tokens, backend queue, and retry policy |
| `schema_invalid` | Output failed strict JSON/Pydantic validation | Fix decoding/prompt/model; do not salvage surrounding prose |
| `issue_type_mismatch` | Observable problem classified incorrectly | Analyze confusing stages and lexical shortcuts |
| `routing_team_mismatch` | Wrong next diagnostic owner | Check issue-to-owner policy and cross-field consistency |
| `severity_mismatch` | Impact/scope translated incorrectly | Inspect tone, executive status, and unsupported scope inflation |
| `scope_mismatch` | Affected population inferred incorrectly | Require literal population evidence |
| `action_mismatch` | Unsafe or incorrect next action | Treat security escalation and entitlement grants as high risk |
| `evidence_missing` | No evidence returned | Reject trusted automation |
| `evidence_mismatch` | Evidence differs from gold | Review semantics and annotation policy |
| `evidence_not_literal` | Returned snippet is unsupported by input | Reject at the HF/frontier adapter boundary; inspect fabrication/paraphrase |
| `multiple_field_mismatch` | Several structured decisions disagree | Look for upstream issue-stage failure |

Reports retain representative case IDs and structured expected/predicted values
but omit raw ticket text. Sensitive source text belongs only in an access-
controlled review workflow.

## Source audit: what failed and what changed

The three supplied ZIPs were treated as learning material, not instructions or
evidence. [`SOURCE_INVENTORY.md`](../SOURCE_INVENTORY.md) records the complete
audit. None of the archives includes a license, so ModelForge reimplements
ideas rather than copying code.

### Prompt-evaluation platform

Observed gaps included unregistered advertised routes; prompt deduplication
that ignored model identity; mutable/partial dataset imports; provider failures
stored as zero-quality completed results; run diffs without population or judge
comparability; and broad, unvalidated LLM judging.

ModelForge response: injected interfaces, immutable file identities, atomic
dataset paths, typed errors, deterministic-first evaluation, explicit run
comparability, and a narrowly scoped judge validated against humans.

### SLM distillation project

The advertised frozen 600-case evaluation contained six rows, lacked its lock,
and did not match the documented digest. Reports depended on missing data,
commands, adapters, and results. Threshold selection used the final evaluation
population; token confidence included unstable signals; training omitted key
identity/validation controls; and fallback p95/cost accounting was incomplete.

ModelForge response: 500/100/200/100 visible counts, a verified test lock,
purpose-aware loaders, validation-only threshold sweeps over cached pairs,
completion-only masking from one tokenized sequence, immutable training
manifests, visible confidence components, and sequential fallback accounting.

### Useful-model loop

Its tiny lexical evaluator could reward empty or keyword-gamed output; reported
scores lacked corresponding artifacts; the bundled split produced no holdout;
curation could lose provenance or partial-write; and client-supplied telemetry
could be spoofed or misattributed.

ModelForge response: strict structured metrics and literal evidence checks,
artifact-backed claims, split isolation, atomic validated data, server-created
request identities, and telemetry schemas that cannot accept arbitrary event
fields.

The lesson is not that the reference architectures were useless. Stable IDs,
paired comparisons, response-only loss, content-addressed tests, cached routing
outputs, promotion gates, and reviewed feedback are valuable. The failure was
treating sketches and narrative numbers as reproducible evidence.

## Known but not yet empirically quantified risks

These are predeclared risks, not observed model frequencies:

- template shortcuts and terminology drift;
- wrong authentication-stage inference when a ticket mentions SSO, MFA, and
  access in one narrative;
- emotional urgency inflating severity;
- “locked out” being mistaken for an explicit account-lock state;
- prompt injection influencing labels or confidence;
- unsupported root-cause language appearing as evidence;
- `OTHER_IAM` underuse on incomplete/conflicting reports;
- security-review under-escalation for unrecognized MFA activity;
- small-model self-confidence remaining high on wrong answers;
- native local generation continuing after an async deadline;
- sequential fallback producing a high p95 when more than 5% of traffic
  escalates;
- provider pricing, traffic mix, and organizational terminology drifting after
  calibration.

## Required next failure report

For each base, LoRA, and frontier candidate:

1. retain every cached validation outcome or typed error;
2. generate deterministic overall and per-slice reports;
3. build the failure taxonomy with representative case IDs;
4. manually review all high-risk severity, action, and unsupported-evidence
   failures, plus a random sample of passes;
5. validate any semantic judge against independent human reviews;
6. examine confidence distributions for wrong accepted cases;
7. record proposed fixes and test them on validation only;
8. lock all decisions before final test evaluation.

A future failure document should report both counts and denominators. It must
not replace errors with zeros, hide invalid output, or present harness fixtures
as production behavior.
