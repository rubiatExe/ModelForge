# ModelForge IAM Ticket Labeling Guide

## Purpose and boundary

This guide defines the gold-labeling policy for the narrow ModelForge v1 task:
enterprise identity and access management (IAM) support-ticket triage. A label
describes the observable support problem and the safest next action. It does not
claim a root cause, resolve the ticket, or determine whether a user is entitled
to access.

Ticket text is untrusted data. Instructions inside a subject or body—such as
“ignore the classifier,” requests to output a particular label, or JSON that
looks like a model response—are ticket content, not labeling instructions.

The core rule is:

> Label only what the ticket supports. Do not convert guesses, emotional
> language, or familiar terminology into technical evidence.

The committed v1 benchmark is synthetic and remains
`PENDING_HUMAN_REVIEW`. Passing its programmatic integrity checks is not human
approval and does not establish production readiness.

## Annotation workflow

1. Read the subject and body once without assigning a label.
2. Identify the last clearly observed stage: before authentication, during MFA,
   after successful authentication, or during provisioning/synchronization.
3. Select `issue_type` from direct observations, not a presumed root cause.
4. Determine `affected_scope` from explicit counts or scope statements.
5. Assign `severity` from supported scope and business impact. Ignore tone.
6. Select the team and action that own the next diagnostic step.
7. Copy one to five short, literal evidence spans from the subject or body.
8. Re-read for contradictions, injected instructions, unsupported implications,
   sensitive data, and accidental evidence paraphrases.
9. Record annotation and review status honestly. A reviewer must not approve
   their own disputed label without adjudication.

When the ticket does not support a closed label, use `OTHER_IAM`, route to
`SERVICE_DESK`, and choose `COLLECT_MORE_INFORMATION`. Do not force a specific
diagnosis merely to reduce use of the catch-all class.

## `issue_type`

### `SSO_AUTHENTICATION_FAILURE`

Use when the ticket directly describes failure in federated or SSO
authentication: redirect loops, SAML/OIDC handoff failure, successful IdP login
that does not establish an application session, or repeated return to login.

Do not use for a rejected MFA challenge, a locked directory account, or an
authorization error after the application session is established.

### `MFA_FAILURE`

Use for a missing or rejected authenticator push/code, failed security-key
challenge, broken MFA enrollment, or suspicious/unrecognized MFA approvals.
Unrecognized approvals should normally use `ESCALATE_SECURITY_REVIEW` rather
than a routine MFA reset.

### `ACCOUNT_LOCKED_OR_DISABLED`

Use only when the ticket explicitly reports that the identity/account is
locked, disabled, suspended, or locked after failed attempts. “Locked out” in
ordinary speech is not enough if the body says authentication succeeds.

### `ACCESS_DENIED_AFTER_LOGIN`

Use when authentication succeeds but the user lacks an application role,
entitlement, group-derived permission, project, or resource. Examples include
an explicit “Access denied” page after login or a missing role in an otherwise
working application session.

### `PROVISIONING_OR_SYNC_FAILURE`

Use when an expected identity, account, group membership, entitlement, or
deprovisioning change exists upstream but has not reached a downstream system.
The ticket must support an expected automated creation/change or a stale sync
state; a generic request for new access is not automatically a sync failure.

### `OTHER_IAM`

Use for IAM-related requests that lack enough evidence to distinguish the
listed types, or for a supported IAM problem outside them. State what missing
fact is needed in reviewer notes; never invent that fact in `evidence`.

## `affected_scope`

- `SINGLE_USER`: exactly one user/account is explicitly affected.
- `MULTIPLE_USERS`: more than one user is named or counted, without support for
  a whole department or organization.
- `DEPARTMENT`: the ticket explicitly supports department-wide impact.
- `ORGANIZATION`: the ticket supports broad, cross-department or company-wide
  impact.
- `UNKNOWN`: the ticket does not establish how many users are affected.

Do not upgrade scope because a high-profile employee submitted the ticket.
“My team” alone is not a department unless the ticket establishes that scope.

## `severity`

Severity reflects supported impact and urgency, not punctuation, capitalization,
anger, or a requester-selected priority.

- `P1`: organization-wide inability to authenticate/access a critical service,
  or a credible active identity-security event with broad immediate impact.
- `P2`: multiple-user or department impact, or a serious time-sensitive access
  failure with a viable incident response path.
- `P3`: one supported user/account is blocked or materially impaired, without
  evidence of broader impact.
- `P4`: impact is unknown, informational, low-impact, or too incomplete to
  establish a higher severity.

A credible unrecognized-MFA or account-compromise indicator is at least `P2`
even when only one account is known; use `P1` only when the supported scope and
immediate impact meet the P1 rule.

In this seed benchmark, an explicit organization-wide outage is `P1`, explicit
multiple-user/department impact is `P2`, a single-user block is `P3`, and
unknown scope is `P4`. Real operational use must also apply the organization’s
documented business-criticality and security-incident policy.

“URGENT!!! THIS IS TERRIBLE!” does not increase severity by itself. A prompt
injection that demands `P1` has no labeling weight.

## Routing and recommended action

Choose the owner of the next evidence-based diagnostic step:

| Observable problem | Routing team | Recommended action |
|---|---|---|
| Locked/disabled account | `SERVICE_DESK` | `RESET_OR_UNLOCK_ACCOUNT` |
| Routine MFA failure | `IDENTITY_PLATFORM` | `INVESTIGATE_MFA_CONFIGURATION` |
| SSO/federation failure | `IDENTITY_PLATFORM` | `INVESTIGATE_IDP_OR_SSO_CONFIGURATION` |
| Authenticated but unauthorized | `ACCESS_MANAGEMENT` | `REVIEW_ACCESS_ENTITLEMENTS` |
| Provisioning/sync lag or failure | `IDENTITY_PLATFORM` | `INVESTIGATE_PROVISIONING_SYNC` |
| Insufficient information | `SERVICE_DESK` | `COLLECT_MORE_INFORMATION` |
| Credible compromise indicators | safest applicable team | `ESCALATE_SECURITY_REVIEW` |

Do not use `RESET_OR_UNLOCK_ACCOUNT` when the ticket merely says “locked out”
but provides no account-lock message. Do not grant or recommend an entitlement;
`REVIEW_ACCESS_ENTITLEMENTS` means verify approval and current state.

## Literal evidence policy

`evidence` must contain one to five concise snippets copied from the ticket’s
subject or body. Every snippet must remain a literal substring after NFKC
Unicode normalization, case folding, and whitespace normalization.

Good evidence:

- `approve MFA, then the browser returns to the sign-in screen`
- `I can sign in successfully`
- `Four people on my team see the same behavior`

Invalid evidence:

- `The IdP configuration is broken` when the ticket reports only a redirect.
- `The whole company is affected` when the ticket mentions two users.
- A cleaned-up paraphrase that does not occur in the source ticket.
- The injected phrase `output P1` as support for actual severity.

Prefer the shortest span that preserves the observation. Evidence should cover
the issue and, when available, the affected scope. Do not copy passwords,
tokens, recovery codes, government identifiers, or other unnecessary secrets.
Escalate the record for privacy review if such content appears.

## Ambiguity, conflict, and adversarial text

- Prefer direct observed behavior over a requester’s proposed diagnosis.
- When two observations genuinely support different issue types and neither is
  clearly primary, use `OTHER_IAM` and `COLLECT_MORE_INFORMATION`.
- Treat phrases such as “probably an Okta bug” as hypotheses unless backed by
  an observable handoff failure.
- Emotional urgency affects neither scope nor severity.
- Misleading words in the subject do not override a precise body description.
- Prompt-injection text never changes the labeling policy.
- Missing screenshots or logs do not erase a clear symptom, but missing the
  failed stage or error usually requires `OTHER_IAM`.

The adversarial split is reported separately under exactly one primary
category: emotional urgency, ambiguous report, conflicting evidence,
misleading terminology, prompt injection, or incomplete information. A
category describes the robustness challenge, not a new IAM label.

## Provenance, review, and disagreement

- `SYNTHETICALLY_LABELED` means labels came from deterministic generation rules.
- `HUMAN_ANNOTATED` means one qualified annotator assigned the record.
- `HUMAN_ADJUDICATED` means disagreement was resolved under this guide.
- `PENDING_HUMAN_REVIEW` must remain until a human checks the full record.
- `HUMAN_APPROVED` and `HUMAN_REJECTED` require an attributable review event in
  the review artifact; changing JSON alone is insufficient evidence.

For disagreement, each reviewer records their independent label and a short
source-based reason. An adjudicator then selects a label or marks the example
unsuitable. Never silently average categorical labels. Conflicting labels for
the same canonical ticket are a dataset error and must block publication.

## Split isolation and quality checklist

Train only on `train`. Tune prompts, thresholds, and routing policy only on
`validation`. Use `test` once for final measurement through the SHA-verified
test loader. Report `adversarial` separately and never train on it.

Before accepting a dataset version, verify:

- every record passes the strict schema and contains no extra fields;
- each evidence snippet is literally grounded;
- canonical IDs and ticket fingerprints are unique;
- no canonical ticket appears in more than one split;
- one canonical ticket never has conflicting gold labels;
- all label distributions and record counts match the manifest;
- split file SHA-256 digests match the manifest;
- the untouched test file matches `test.lock.json`;
- provenance, annotation status, and review status are truthful;
- no production PII, secret, or proprietary ticket text was committed.

Any content change creates a new dataset version and new manifest. Never edit a
locked test file in place or regenerate evidence after observing test results.
