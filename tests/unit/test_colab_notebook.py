from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/modelforge_qwen_lora_colab.ipynb"
CONFIG_PATH = PROJECT_ROOT / "experiments/configs/qwen_lora_r16_colab_v1.yaml"
LOCK_PATH = PROJECT_ROOT / "requirements/colab-linux-py312-cu128.lock"
QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def _notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code() -> str:
    cells = _notebook()["cells"]
    assert isinstance(cells, list)
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    )


def _markdown() -> str:
    cells = _notebook()["cells"]
    assert isinstance(cells, list)
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "markdown"
    )


def test_colab_notebook_is_unexecuted_valid_python() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    assert isinstance(cells, list)
    assert cells
    for cell in cells:
        assert isinstance(cell, dict)
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        ast.parse("".join(cell.get("source", [])))


def test_colab_notebook_has_fail_closed_privacy_and_provenance_guards() -> None:
    code = _code()
    markdown = _markdown()
    lowered = code.lower()
    for forbidden in (
        "drive.mount",
        "files.upload",
        "login(",
        "--confirm-locked-test",
        '"--split", "test"',
        "trust_remote_code=true",
    ):
        assert forbidden not in lowered
    assert "subprocess.run" in code
    assert "check=True" in code
    assert "HF_HUB_DISABLE_IMPLICIT_TOKEN" in code
    assert "verify_git_checkout" in code
    assert "verify_training_manifest" in code
    assert "verify_response_only_mask_audit" in code
    assert "prepare_evidence_bundle" in code
    assert "verify_downloaded_bundle" in code
    assert "ZERO_COST_CONFIRMED = False" in code
    assert "if not ZERO_COST_CONFIRMED" in code
    assert "Code cannot inspect billing" in code
    assert "compute-unit balance is 0 or exhausted" in code
    assert "before you press Connect" in markdown
    assert "attestation, not a technical billing control" in markdown
    assert "confidence-routing-smoke" in code
    assert "response-mask-audit" in code
    assert "adapter-api-smoke" in code
    assert '"--local-files-only"' in code
    assert code.index("response-mask-audit") < code.index("modelforge.training.train_lora")
    assert code.index("adapter-validation") < code.index("adapter-api-smoke")
    assert code.index("adapter-api-smoke") < code.index("final-validation")
    for artifact_name in (
        "response-only-mask-audit.json",
        "confidence-routing-smoke.json",
        "adapter-api-smoke.json",
    ):
        assert artifact_name in code
    assert "private" in lowered
    assert "REMOTE_RUN_PATH" in code
    assert QWEN_REVISION in code
    assert not any(
        token.startswith("hf_") and len(token) > 10
        for token in code.replace('"', " ").replace("'", " ").split()
    )


def test_colab_config_and_lock_are_pinned() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["model_name_or_path"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config["model_revision"] == QWEN_REVISION
    assert config["tokenizer_revision"] == QWEN_REVISION
    assert config["trust_remote_code"] is False
    assert config["local_files_only"] is False
    assert config["require_resolved_revision"] is True
    assert config["device"] == "cuda"
    assert config["output_root"].startswith("/content/")

    lock = LOCK_PATH.read_text(encoding="utf-8")
    assert "torch==2.11.0+cu128" in lock
    assert "transformers==4.57.6" in lock
    assert "peft==0.20.0" in lock
    assert "--hash=sha256:" in lock
    assert "git+" not in lock
    assert "-e " not in lock
