# ADR 0003: Dataset size, provenance, and split isolation

## Problem

ModelForge needs enough labeled IAM examples to compare baselines without
pretending a synthetic seed set represents production traffic. The final test
must not be tuned against.

## Options considered

- One randomly split file with no lock
- 500 train / 100 validation / 200 test plus 100 adversarial
- A much larger generated corpus
- Real production tickets at v1

## Decision

Create a deterministic synthetic seed set with 500 train, 100 validation, 200
hash-locked test, and 100 separately reported adversarial examples. Store
provenance, annotation status, review status, distributions, and hashes.

## Reason

The requested size is large enough to exercise training and per-class metrics
but small enough to inspect. Separate validation supports prompt, model, and
routing calibration while the test stays untouched. Synthetic data avoids
committing employee PII.

## Evidence

The supplied references show why a visible, mutable, or undersized eval set is
misleading: one advertised 600-case set contains only six rows, and another
uses eight visible cases repeatedly.

## Tradeoffs

Template-generated language is easier and less diverse than real tickets. Its
absolute metrics will overestimate field performance and cannot establish
production readiness.

## What would change the decision

Access to privacy-reviewed, de-identified, human-annotated tickets with a lawful
retention and evaluation process. Those would form a new dataset version, not
silently replace v1.
