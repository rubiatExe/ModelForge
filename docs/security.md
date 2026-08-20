# Security and privacy boundary

ModelForge v1 handles support-ticket text, which may contain employee names,
account identifiers, internal application names, and other sensitive metadata.
The repository ships only synthetic examples.

## Threats considered

- Prompt injection inside ticket text.
- Oversized inputs causing memory, token-cost, or inference denial of service.
- Malformed model JSON crossing into automated routing.
- Unsupported evidence or root-cause fabrication.
- Provider keys committed to source or required at import time.
- Sensitive ticket content appearing in logs, errors, telemetry, or eval
  artifacts.
- Paid frontier calls triggered without a configured allowlist and deadline.
- Test-set edits or leakage producing misleading quality claims.
- Unpinned model/config identity making a run irreproducible.

## Implemented controls

- Strict Pydantic contracts with `extra="forbid"` and bounded text fields.
- A pure-ASGI request limit that also counts chunked bodies.
- Ticket content is delimited as untrusted data in the model prompt; injection
  examples are included in adversarial evaluation.
- Model output must parse directly as the target JSON schema. Surrounding prose
  and extra fields are rejected.
- Both Hugging Face and frontier adapter outputs must pass normalized literal
  evidence grounding against the ticket subject/body before either adapter can
  return a `ModelPrediction`.
- Ambiguous or injection-marked tickets receive a lower routing score.
- Frontier configuration is loaded only when the app is constructed. Secrets
  come from environment variables and `SecretStr` hides them from repr output.
- Provider errors are sanitized and remain distinct from quality failures.
- Operational logs and telemetry omit ticket IDs, subject, body, department,
  evidence, prediction content, and provider response bodies.
- Test files are hash-locked; training loaders reject test/adversarial splits.
- Production requires a calibrated routing policy artifact.

## Deliberate non-features

Authentication, organization-level authorization, distributed rate limiting,
encryption-at-rest configuration, and regional provider policy depend on the
deployment environment. They are not simulated in this educational repository.
A production deployment must put the API behind an authenticated gateway and
select a telemetry/evaluation store with the required access, retention,
deletion, encryption, and data-residency controls.

The local JSONL telemetry backend is single-process and operational only. It is
not an audit database.

## Operational rules for real data

1. Do not add raw production tickets to Git.
2. Establish a lawful purpose, retention period, access group, and deletion
   path before collecting tickets or human reviews.
3. De-identify data before sending it to a third-party model and verify the
   provider's contractual retention/training settings.
4. Keep raw data, derived model outputs, reviewer data, and public experiment
   summaries in distinct access domains.
5. Rotate provider keys outside the application and never return provider error
   bodies to clients.
6. Treat corrected feedback as untrusted until schema, PII, leakage, and human
   review gates pass.

## Known limitations

Regex/phrase injection markers are only a routing signal, not a complete prompt
injection defense. Literal evidence checks reject unsupported strings but do not
prove the label is correct. Synthetic tests cannot establish how the system
behaves on a real organization's terminology, identity providers, or attack
surface.
