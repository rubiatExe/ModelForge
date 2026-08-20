from pathlib import Path

import pytest

from modelforge.experiments.run_classical import _percentile, run_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "iam_triage_v1"


def test_nearest_rank_percentile() -> None:
    assert _percentile([40, 10, 30, 20], 0.5) == 20
    assert _percentile([40, 10, 30, 20], 0.95) == 40
    assert _percentile([], 0.95) == 0


def test_classical_experiment_refuses_existing_model_before_training(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "existing.joblib"
    model_path.write_bytes(b"unit fixture")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_experiment(
            dataset_root=DATASET_ROOT,
            output_path=tmp_path / "new-result.json",
            model_path=model_path,
        )

    assert model_path.read_bytes() == b"unit fixture"
    assert not (tmp_path / "new-result.json").exists()
