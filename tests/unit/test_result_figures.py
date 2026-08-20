from pathlib import Path

import pytest

from modelforge.experiments.render_measured_figures import render_figures, svg_text

ROOT = Path(__file__).resolve().parents[2]


def test_svg_text_escapes_xml_control_characters() -> None:
    assert svg_text('<script data-x="1">&') == "&lt;script data-x=&quot;1&quot;&gt;&amp;"


def test_renderer_uses_measured_artifacts_and_refuses_overwrite(tmp_path: Path) -> None:
    arguments = {
        "classical_path": ROOT / "experiments/results/classical_tfidf_logreg_v1.json",
        "ablation_path": ROOT
        / "experiments/results/classical_label_quality_ablation_v1.json",
        "output_directory": tmp_path,
    }
    outputs = render_figures(**arguments)
    assert len(outputs) == 3
    for output in outputs:
        payload = output.read_text(encoding="utf-8")
        assert "Synthetic IAM benchmark" in payload
        assert "issue_type" in payload
        assert "<svg" in payload and payload.endswith("</svg>\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render_figures(**arguments)


def test_renderer_can_explicitly_replace_complete_figure_set(tmp_path: Path) -> None:
    arguments = {
        "classical_path": ROOT / "experiments/results/classical_tfidf_logreg_v1.json",
        "ablation_path": ROOT
        / "experiments/results/classical_label_quality_ablation_v1.json",
        "output_directory": tmp_path,
    }
    first = render_figures(**arguments)
    second = render_figures(**arguments, overwrite=True)
    assert first == second
