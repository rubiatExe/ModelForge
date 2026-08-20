# ModelForge

**Can a LoRA-tuned open-weight model handle most enterprise IAM ticket triage — escalating uncertain cases to a frontier model — at materially lower cost?**

ModelForge is built to answer that question with *defensible evidence*, not impressive-looking invented numbers. Every claim in this repo is backed by a committed, machine-readable artifact.

---

## What's in this repo

| Layer | What it does |
|---|---|
| **Data** | 900-example synthetic IAM dataset, split-locked, hash-verified, provenance-tagged |
| **Models** | Classical TF-IDF baseline · HuggingFace small model · LoRA adapter · Frontier (OpenAI-compatible) |
| **Evaluation** | Deterministic metrics, calibration, failure taxonomy, human/judge review, regression gates |
| **Routing** | Composite uncertainty signal with fail-closed escalation and threshold sweep |
| **API** | FastAPI service with typed errors, privacy-safe telemetry, and model registry |
| **Docs** | Architecture, labeling guide, model card, failure analysis, 7 ADRs |

---

## Evidence status

Only experiments that produced committed artifacts count as evidence.

| Experiment | Status | What it proves |
|---|---|---|
| TF-IDF + logistic regression | ✅ **Measured** | Issue-type classification baseline on the synthetic benchmark |
| Label-quality ablation (0%, 10%, 30% noise) | ✅ **Measured** | Sensitivity of the classical model to training-label corruption |
| `heuristic-demo-v1` API fixture | ⚙️ **Harness only** | Software plumbing, not model quality |
| Qwen2.5-0.5B base model | 🔲 **Not run** | Requires model download + GPU |
| LoRA/PEFT fine-tune | 🔲 **Not run** | Requires completed training run |
| Frontier adapter | 🔲 **Not run** | Requires paid API credentials |
| Routing threshold calibration | 🔲 **Not run** | Policy is labeled `uncalibrated`; not a production recommendation |

The authoritative measured artifacts are:
- [`experiments/results/classical_tfidf_logreg_v1.json`](experiments/results/classical_tfidf_logreg_v1.json)
- [`experiments/results/classical_label_quality_ablation_v1.json`](experiments/results/classical_label_quality_ablation_v1.json)

---

## Classical baseline results

> ⚠️ A perfect score here is a **warning about benchmark simplicity**, not a production signal. The template generator creates strong lexical cues. No LLM has been evaluated yet.

**TF-IDF + Logistic Regression — `issue_type` only**

| Split | Macro F1 | Avg latency | p95 latency | Cost/request |
|---|---:|---:|---:|---:|
| Validation | 1.0000 | — | — | $0.00 |
| Locked test | 1.0000 | 0.19 ms | 0.20 ms | $0.00 |

**Label-quality ablation (30% noise drops test F1 from 1.000 → 0.960)**

| Injected noise | Corrupted train labels | Validation F1 | Test F1 |
|---:|---:|---:|---:|
| 0% | 0 | 1.000 | 1.000 |
| 10% | 50 | 1.000 | 1.000 |
| 30% | 150 | 0.970 | 0.960 |

![Per-class F1 on locked test](experiments/figures/classical-test-per-class-f1.svg)
![Latency](experiments/figures/classical-latency.svg)
![Label quality ablation](experiments/figures/classical-label-quality-ablation.svg)

---

## Architecture

```
Versioned JSONL dataset
        │
        ├── Classical baseline (TF-IDF + LogReg)
        ├── Base small model (Qwen2.5-0.5B)
        ├── LoRA adapter (PEFT, response-only loss)
        └── Frontier adapter (OpenAI-compatible)
                │
        Deterministic evaluation
        (metrics · calibration · failures · regression gate)
                │
        Threshold sweep on validation
        (cached paired outputs at 0.60 → 0.95)
                │
        Versioned routing policy
                │
        FastAPI  ──▶  small model first
                  └─▶  fail-closed frontier escalation
                │
        Privacy-safe operational telemetry (JSONL, no ticket content)
```

Every model adapter crosses the same strict `TriageResult` Pydantic boundary. Evidence must be a literal substring of the ticket text — non-literal evidence is rejected at the model boundary, not silently passed through.

See [`docs/architecture.md`](docs/architecture.md) and the [decision log](docs/decisions/README.md).

---

## Dataset

`iam_ticket_triage@1.0.0` — deterministically generated from templates, no production data.

| Split | Examples | Use |
|---|---:|---|
| `train` | 500 | Fit models and adapters |
| `validation` | 100 | Tune prompts, thresholds, routing policy |
| `test` | 200 | **Final evaluation only** — hash-locked |
| `adversarial` | 100 | Robustness testing — 6 declared categories |

The loader enforces cross-split duplicate detection, conflicting-label detection, manifest SHA-256 verification, and a hard block on loading test/adversarial data during training.

---

## Quick start

**Requires Python 3.11–3.13.**

```bash
# 1. Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dev dependencies
pip install -e '.[dev]'

# 3. Run linting + full test suite
make lint
make test

# 4. Verify dataset integrity (idempotent — safe to re-run)
python scripts/generate_dataset.py

# 5. Re-run the classical experiment (writes to /tmp to avoid overwriting committed evidence)
python -m modelforge.experiments.run_classical \
  --output /tmp/modelforge-classical-result.json \
  --model-output /tmp/modelforge-classical-model.joblib

# 6. Re-run the label-quality ablation
python -m modelforge.experiments.run_data_ablation \
  --noise-fractions 0 0.1 0.3 \
  --seed 20260820 \
  --output /tmp/modelforge-label-quality-ablation.json

# 7. Regenerate the SVG figures from artifacts
python -m modelforge.experiments.render_measured_figures --overwrite
```

---

## Run the API

The keyless default registers `heuristic-demo-v1` — a deterministic harness fixture that makes no model calls.

```bash
modelforge-api
```

```bash
# Health check
curl http://127.0.0.1:8000/health

# Triage a ticket
curl -X POST http://127.0.0.1:8000/v1/models/heuristic-demo-v1/triage \
  -H 'content-type: application/json' \
  -d '{
    "ticket_id": "TKT-DEMO-1",
    "subject": "SSO redirect loop after Okta",
    "body": "Okta shows success but an SSO redirect loop affects multiple users.",
    "employee_department": "Sales",
    "submitted_at": "2026-08-20T12:00:00Z"
  }'
```

**Available endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service status |
| `GET` | `/v1/models` | Registered model list |
| `POST` | `/v1/models/{model_id}/triage` | Triage via specific model |
| `POST` | `/v1/triage` | Triage via routing policy |
| `POST` | `/v1/evaluations/run` | Trigger an evaluation run |
| `GET` | `/v1/evaluations/{run_id}` | Fetch evaluation result |
| `GET` | `/v1/telemetry` | Operational metrics |

> `/health` reports `degraded` while the routing policy is uncalibrated. `POST /v1/triage` returns `503 FRONTIER_UNAVAILABLE` when escalation is needed but no frontier is configured — this is intentional fail-closed behavior.

---

## Optional: LoRA fine-tuning

Requires a model download, optional ML dependencies, and suitable hardware (GPU/MPS).

```bash
pip install -e '.[train,dev]'

modelforge-train \
  --config experiments/configs/qwen_lora_r16_v1.yaml \
  --project-root .
```

The config targets `Qwen/Qwen2.5-0.5B-Instruct` at LoRA rank 16. The run manifest records model revision, dataset hashes, config hash, code SHA, and packages — runs are immutable. **No committed run exists yet in this repo.**

To serve a trained local model, copy `.env.example` to `.env` and fill in the HuggingFace and/or frontier provider variables.

---

## Key design decisions

| # | Decision | Short rationale |
|---|---|---|
| [ADR-0001](docs/decisions/0001-initial-small-model.md) | Start with Qwen2.5-0.5B | Smallest CPU/MPS-accessible instruct model; easy to swap |
| [ADR-0002](docs/decisions/0002-lora-over-full-finetuning.md) | LoRA over full fine-tune | Lower cost, reversible, no catastrophic forgetting risk |
| [ADR-0003](docs/decisions/0003-dataset-size-and-isolation.md) | 500/100/200/100 split | Sufficient signal without overinvesting before quality is known |
| [ADR-0004](docs/decisions/0004-deterministic-first-evaluation.md) | Deterministic evaluators first | LLM judges are optional and must be validated before gating |
| [ADR-0005](docs/decisions/0005-confidence-routing.md) | Composite uncertainty signal | Single logprob is unreliable; inspectable components are safer |
| [ADR-0006](docs/decisions/0006-defer-quantization.md) | Defer quantization | Premature optimization before quality is measured |
| [ADR-0007](docs/decisions/0007-routing-threshold-remains-uncalibrated.md) | Threshold uncalibrated | Requires a completed paired-output sweep on real model results |

---

## Repository map

```
src/modelforge/
├── schemas/        # Pydantic contracts and IAM enums
├── datasets/       # Integrity, provenance, and split-lock checks
├── models/         # Classical, HuggingFace, frontier, and demo adapters
├── training/       # LoRA training and immutable run manifests
├── evaluation/     # Metrics, calibration, judge/human review, regression gates
├── routing/        # Confidence scoring and fail-closed router
├── api/            # FastAPI service, model registry, evaluation endpoints
├── economics/      # Cost tracking
└── telemetry/      # Privacy-safe operational telemetry

data/iam_triage_v1/ # Dataset splits + manifest + lock
experiments/
├── configs/        # Experiment configs (YAML)
├── results/        # Committed machine-readable results (JSON)
└── figures/        # Generated SVG figures
docs/
├── decisions/      # Architecture decision records (ADRs)
├── architecture.md
├── evaluation.md
├── experiments.md
├── failure-analysis.md
├── model-card.md
├── LABELING_GUIDE.md
└── security.md
tests/
├── unit/           # 20 unit test files
├── integration/    # API integration tests
└── evals/          # Golden evaluation fixtures
```

---

## Security and limitations

Ticket text is **untrusted input**. Both generative adapters enforce literal-substring evidence grounding at the model boundary. Request sizes, field lengths, and provider deadlines are bounded. Telemetry stores only operational fields — never ticket IDs, content, evidence, or provider response bodies.

**This repo does not implement:** authentication, authorization, distributed rate limiting, encryption-at-rest, or regional data controls. Use an authenticated gateway and a governed durable store before any production deployment.

**Current limitations:**
- All examples are template-generated and pending human review
- Measured results cover only the `issue_type` field (one of five scored fields)
- No base/LoRA/frontier quality, memory, cost, or load result exists yet
- The routing score is not a calibrated probability
- The development routing threshold is not approved for production

See [`docs/security.md`](docs/security.md) and [`SOURCE_INVENTORY.md`](SOURCE_INVENTORY.md) for the full picture.

---

## License

MIT — see [`LICENSE`](LICENSE).
