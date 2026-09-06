# Qwen LoRA Evidence Runbook

This is the operational procedure for converting the configured Qwen LoRA
pipeline into reproducible evidence. It deliberately contains no result claims:
the model, adapter, metrics, and routing policy remain unmeasured until this
procedure completes.

## Pinned candidate

The training config pins both the base model and tokenizer to the immutable
`Qwen/Qwen2.5-0.5B-Instruct` commit
`7ae557604adf67be50417f59c2c2f167def9a775`. Do not replace this with `main`,
a tag, or another mutable reference. If a later model revision is intentionally
evaluated, give it a new experiment name and a new configuration artifact.

## Hosted Google Colab workflow

Use [`notebooks/modelforge_qwen_lora_colab.ipynb`](../notebooks/modelforge_qwen_lora_colab.ipynb)
when the base model must stay off the developer laptop. The notebook clones an
exact reviewed Git commit into a hosted Colab VM; Qwen is downloaded directly
from Hugging Face into that VM's temporary cache. It never mounts Google Drive
or uploads files from the developer machine.

The notebook is deliberately fail-closed. It requires a GPU, verifies the Git
origin, exact commit, and clean checkout, installs the hash-pinned Linux/CUDA
environment in `requirements/colab-linux-py312-cu128.lock`, checks the complete
test suite, and validates the training manifest before uploading anything. It
then copies only approved evidence files into a dedicated bundle, scans the
bundle for Hugging Face token-like values and secret-like filenames, and writes
a SHA-256 evidence index.
It also runs the real pinned Qwen tokenizer across every training row, executes
one adapter-backed request through FastAPI's in-process `TestClient`, and
exercises the router's high-confidence local and escalation-required
fail-closed branches with deterministic fixtures. Those three JSON artifacts
contain hashes and control facts, not ticket or model-response bodies.

Before opening the notebook:

1. Commit and push the reviewed ModelForge source. Put that exact 40-character
   SHA in `SOURCE_COMMIT`; never use a branch name such as `main`.
2. Create a **private Hugging Face dataset repository** for evidence. Create a
   short-lived fine-grained token with write access only to that repository and
   save it as the Colab secret `HF_TOKEN`. Do not paste it into notebook code.
3. In Colab, select the `2026.07` past runtime and a hosted NVIDIA GPU. That
   runtime pins Python 3.12 and PyTorch 2.11, matching the compiled CUDA 12.8
   lock. If that runtime is no longer offered, regenerate and review the lock
   for the replacement runtime before training.
4. Check the Colab compute-unit balance before connecting. For strict zero-unit
   operation it must be `0` or exhausted; Colab documents no free-tier opt-out
   while a positive balance remains. Then verify that Colab assigned a
   free-tier, standard-memory T4 and that you did not select Premium GPU, High
   RAM, a GCP runtime, or an Enterprise runtime. Notebook code cannot inspect
   billing. Leave `ZERO_COST_CONFIRMED = False` and stop if either condition is
   unclear; change it to `True` only after this manual check.
5. Choose a fresh `RUN_ID`. The remote path includes both the source SHA and run
   ID, and the notebook refuses to overwrite an existing prefix.

The first hosted run stops after complete base and adapter validation. It does
not expose the locked test split or claim calibrated routing. The persisted
routing smoke is explicitly fixture-based control-flow evidence. Once the
final private upload and checksum verification succeed, disconnect and delete
the Colab runtime, then revoke the short-lived token.

## Before downloading or training

1. Confirm the worktree is clean and choose an immutable external archive
   destination for the eventual `experiments/runs/<experiment_name>` directory.
   Run directories and model weights are intentionally ignored by Git.
2. Confirm the optional training stack and accelerator state. Do not start an
   unplanned CPU run of a 0.5B model.

   ```bash
   ./.venv/bin/python -c "import torch; print({'cuda': torch.cuda.is_available(), 'mps': torch.backends.mps.is_available()})"
   ```

3. Verify the manifest-checked synthetic dataset and software checks:

   ```bash
   ./.venv/bin/python scripts/generate_dataset.py
   ./.venv/bin/python -m ruff check .
   ./.venv/bin/python -m pytest -q
   ```

4. Confirm that the model download, local disk use, and any cloud or frontier
   costs are authorized. The training command will download the model only when
   it is not already in the Hugging Face cache.

## Train one immutable LoRA run

Use a unique `experiment_name` for every attempt. A failed or completed run
directory is never reused.

```bash
./.venv/bin/modelforge-train \
  --config experiments/configs/qwen_lora_r16_v1.yaml \
  --project-root .
```

The successful run writes `config.json`, `adapter/`, checkpoints, and
`manifest.json` beneath `experiments/runs/qwen_lora_r16_v1/`. The manifest
records the resolved model/tokenizer commits, dataset/prompt/config hashes,
hardware, packages, losses, and hashes of every adapter artifact.

Immediately archive the whole run directory to immutable storage and retain its
archive URI and SHA-256 alongside the manifest. Do not commit model weights or
ticket-bearing evaluation output to Git without a privacy review.

## Evaluate before touching the locked test split

Evaluate the pinned base model and the exact adapter on the complete validation
split. The adapter command requires its training manifest and rehashes the
adapter files before it will write an evaluation artifact.

```bash
REV=7ae557604adf67be50417f59c2c2f167def9a775
RUN=experiments/runs/qwen_lora_r16_v1

./.venv/bin/modelforge-eval \
  --backend hf-base \
  --experiment-id qwen-base-validation-v1 \
  --split validation \
  --output experiments/results/qwen-base-validation-v1.json \
  --model-name-or-path Qwen/Qwen2.5-0.5B-Instruct \
  --revision "$REV" \
  --serving-id small-iam-triage-v1

./.venv/bin/modelforge-eval \
  --backend hf-adapter \
  --experiment-id qwen-lora-validation-v1 \
  --split validation \
  --output experiments/results/qwen-lora-validation-v1.json \
  --model-name-or-path Qwen/Qwen2.5-0.5B-Instruct \
  --revision "$REV" \
  --adapter-name-or-path "$RUN/adapter" \
  --training-manifest "$RUN/manifest.json" \
  --serving-id small-iam-triage-v1
```

Review the full-record metrics, failure taxonomy, evidence grounding, latency,
cost accounting, and invalid-output/error counts. The confidence diagnostic
measures model self-report; it does not make the router's composite score a
calibrated probability.

Before any locked-test evaluation, write and commit a regression policy with
predeclared overall and slice floors. Do not choose thresholds or metric floors
after inspecting test results.

## Router and service evidence

The Colab run produces two deliberately narrow service artifacts. The
adapter-backed FastAPI smoke proves that one manifest-linked adapter request
returned HTTP 200, validated as `TriageResult`, and carried the expected
`X-ModelForge-Model` header. The uncalibrated routing smoke proves that the real
router keeps a high-scoring deterministic fixture local and raises a typed
`FrontierUnavailableError` for low-confidence and local-model-error fixtures
when no frontier is configured. It is labeled `result_status: fixture`, records
`confidence_is_probability: false`, and makes no calibration claim.

Confidence routing is a separate experiment. It needs paired, complete
validation artifacts from a small model and a configured frontier model with
dated pricing provenance. Run the routing sweep only on validation data, lock a
policy from that result, and then evaluate the fixed candidates on the locked
test split. Do not claim calibrated or production routing before this happens.

To serve the trained adapter locally after it is verified, set these values in
an uncommitted `.env` file:

```dotenv
MODELFORGE_RUNTIME_BACKEND=huggingface
MODELFORGE_SMALL_MODEL=Qwen/Qwen2.5-0.5B-Instruct
MODELFORGE_SMALL_MODEL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
MODELFORGE_ADAPTER_PATH=experiments/runs/qwen_lora_r16_v1/adapter
MODELFORGE_LOCAL_FILES_ONLY=true
```

Use `small-iam-triage-v1` as the model identity in evaluation and any eventual
routing policy so the policy cannot be attached to a different serving model.

## What can be claimed after each stage

| Completed evidence | Defensible statement |
| --- | --- |
| Full pinned-tokenizer mask audit + linked training manifest | Applied response-only supervision to every manifest-verified training row without truncation in that run. |
| Training manifest + archived adapter | Fine-tuned a Qwen2.5-0.5B adapter with LoRA. |
| Base and adapter validation artifacts | Evaluated base and LoRA candidates on the versioned synthetic benchmark. |
| Adapter FastAPI smoke artifact | Served one schema-valid, manifest-linked adapter response through the in-process FastAPI contract. |
| Uncalibrated routing smoke artifact | Exercised local and fail-closed router control-flow branches with deterministic fixtures; this does not establish calibration. |
| Predeclared policy + fixed locked-test reports | Reported the corresponding measured metrics on the locked synthetic test set. |
| Paired frontier sweep + locked routing policy | Implemented and validation-selected a selective frontier-fallback policy. |

Every statement remains limited to the synthetic IAM benchmark until a separate
privacy-reviewed, human-annotated dataset provides real-ticket evidence.
