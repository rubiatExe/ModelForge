"""Render dependency-free SVG charts from committed measured artifacts."""

from __future__ import annotations

import argparse
import html
import math
from collections.abc import Iterable
from pathlib import Path

from modelforge.datasets import atomic_write_bytes
from modelforge.experiments.run_classical import ClassicalExperimentArtifact
from modelforge.experiments.run_data_ablation import DataQualityAblationArtifact

_WIDTH = 960
_HEIGHT = 520
_LEFT = 90
_RIGHT = 30
_TOP = 78
_BOTTOM = 112
_PLOT_WIDTH = _WIDTH - _LEFT - _RIGHT
_PLOT_HEIGHT = _HEIGHT - _TOP - _BOTTOM


def svg_text(value: object) -> str:
    """Escape untrusted artifact text before placing it in XML."""

    return html.escape(str(value), quote=True)


def _header(title: str, description: str) -> list[str]:
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-labelledby="title desc" viewBox="0 0 {_WIDTH} {_HEIGHT}">',
        f"<title id=\"title\">{svg_text(title)}</title>",
        f"<desc id=\"desc\">{svg_text(description)}</desc>",
        "<style>",
        "text{font-family:Inter,system-ui,-apple-system,sans-serif;fill:#172033}",
        ".title{font-size:22px;font-weight:700}.subtitle{font-size:13px;fill:#526078}",
        ".axis{font-size:12px;fill:#46546b}.value{font-size:12px;font-weight:650}",
        ".grid{stroke:#d7dde8;stroke-width:1}.frame{fill:#fff;stroke:#9aa7ba}",
        ".series-a{fill:#1769aa;stroke:#1769aa}.series-b{fill:#d1495b;stroke:#d1495b}",
        "</style>",
        '<rect width="960" height="520" fill="#ffffff"/>',
        f'<text class="title" x="{_LEFT}" y="34">{svg_text(title)}</text>',
        f'<text class="subtitle" x="{_LEFT}" y="57">{svg_text(description)}</text>',
        f'<rect class="frame" x="{_LEFT}" y="{_TOP}" width="{_PLOT_WIDTH}" height="{_PLOT_HEIGHT}"/>',
    ]


def _footer(lines: list[str], note: str) -> str:
    lines.extend(
        [
            f'<text class="subtitle" x="{_LEFT}" y="{_HEIGHT - 18}">{svg_text(note)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _linear_ticks(low: float, high: float, count: int = 5) -> tuple[float, ...]:
    if high <= low:
        raise ValueError("chart domain must be increasing")
    return tuple(low + (high - low) * index / count for index in range(count + 1))


def _y(value: float, low: float, high: float) -> float:
    return _TOP + _PLOT_HEIGHT * (1.0 - (value - low) / (high - low))


def _grid(lines: list[str], low: float, high: float, *, decimals: int) -> None:
    for tick in _linear_ticks(low, high):
        y = _y(tick, low, high)
        lines.append(
            f'<line class="grid" x1="{_LEFT}" y1="{y:.2f}" x2="{_LEFT + _PLOT_WIDTH}" y2="{y:.2f}"/>'
        )
        lines.append(
            f'<text class="axis" x="{_LEFT - 10}" y="{y + 4:.2f}" text-anchor="end">{tick:.{decimals}f}</text>'
        )


def render_per_class_f1(artifact: ClassicalExperimentArtifact) -> str:
    labels = tuple(artifact.test.metrics.labels)
    values = tuple(float(artifact.test.metrics.per_class[label].f1) for label in labels)
    display = {
        "SSO_AUTHENTICATION_FAILURE": "SSO auth",
        "MFA_FAILURE": "MFA",
        "ACCOUNT_LOCKED_OR_DISABLED": "Account lock",
        "ACCESS_DENIED_AFTER_LOGIN": "Access denied",
        "PROVISIONING_OR_SYNC_FAILURE": "Provisioning/sync",
        "OTHER_IAM": "Other IAM",
    }
    lines = _header(
        "Locked-test per-class F1",
        "Synthetic IAM benchmark · classical issue_type classifier · 200 locked cases",
    )
    _grid(lines, 0.0, 1.0, decimals=1)
    step = _PLOT_WIDTH / len(values)
    bar_width = step * 0.58
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = _LEFT + index * step + (step - bar_width) / 2
        y = _y(value, 0.0, 1.0)
        height = _TOP + _PLOT_HEIGHT - y
        lines.append(
            f'<rect class="series-a" x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{height:.2f}"/>'
        )
        lines.append(
            f'<text class="value" x="{x + bar_width / 2:.2f}" y="{max(_TOP + 15, y - 8):.2f}" text-anchor="middle">{value:.3f}</text>'
        )
        lines.append(
            f'<text class="axis" x="{x + bar_width / 2:.2f}" y="{_TOP + _PLOT_HEIGHT + 24}" text-anchor="middle">{svg_text(display.get(label, label))}</text>'
        )
    lines.append(
        f'<text class="axis" transform="translate(24 {_TOP + _PLOT_HEIGHT / 2}) rotate(-90)" text-anchor="middle">F1 (0–1)</text>'
    )
    return _footer(
        lines,
        "All classes scored 1.000; this saturation is evidence of template simplicity, not production readiness.",
    )


def render_latency(artifact: ClassicalExperimentArtifact) -> str:
    groups = ("Validation", "Locked test")
    metrics = ("Average", "p50", "p95")
    values = (
        (
            artifact.validation.average_latency_ms,
            artifact.validation.p50_latency_ms,
            artifact.validation.p95_latency_ms,
        ),
        (
            artifact.test.average_latency_ms,
            artifact.test.p50_latency_ms,
            artifact.test.p95_latency_ms,
        ),
    )
    high = max(value for group in values for value in group) * 1.18
    lines = _header(
        "Classical inference latency",
        "Synthetic IAM benchmark · sequential single-request issue_type inference",
    )
    _grid(lines, 0.0, high, decimals=3)
    group_step = _PLOT_WIDTH / len(groups)
    bar_width = group_step / 7
    colors = ("#1769aa", "#2a9d8f", "#d1495b")
    for group_index, group_name in enumerate(groups):
        center = _LEFT + group_step * (group_index + 0.5)
        for metric_index, value in enumerate(values[group_index]):
            x = center + (metric_index - 1) * bar_width * 1.35 - bar_width / 2
            y = _y(value, 0.0, high)
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{_TOP + _PLOT_HEIGHT - y:.2f}" fill="{colors[metric_index]}"/>'
            )
            lines.append(
                f'<text class="value" x="{x + bar_width / 2:.2f}" y="{y - 7:.2f}" text-anchor="middle">{value:.3f}</text>'
            )
        lines.append(
            f'<text class="axis" x="{center:.2f}" y="{_TOP + _PLOT_HEIGHT + 26}" text-anchor="middle">{group_name}</text>'
        )
    legend_x = _LEFT
    for index, name in enumerate(metrics):
        x = legend_x + index * 135
        lines.append(
            f'<rect x="{x}" y="{_HEIGHT - 72}" width="13" height="13" fill="{colors[index]}"/>'
        )
        lines.append(f'<text class="axis" x="{x + 20}" y="{_HEIGHT - 61}">{name}</text>')
    lines.append(
        f'<text class="axis" transform="translate(24 {_TOP + _PLOT_HEIGHT / 2}) rotate(-90)" text-anchor="middle">Latency (ms)</text>'
    )
    return _footer(
        lines,
        "Measured on one macOS process; excludes deserialization, concurrency, and production load.",
    )


def render_label_quality(artifact: DataQualityAblationArtifact) -> str:
    points = tuple(sorted(artifact.points, key=lambda item: item.label_noise_fraction))
    fractions = tuple(point.label_noise_fraction for point in points)
    validation = tuple(float(point.validation.metrics.macro_f1) for point in points)
    test = tuple(float(point.test.metrics.macro_f1) for point in points)
    all_values = validation + test
    low = max(0.0, math.floor((min(all_values) - 0.015) * 100) / 100)
    high = min(1.0, math.ceil((max(all_values) + 0.005) * 100) / 100)
    if high <= low:
        low = max(0.0, high - 0.1)
    lines = _header(
        "Training-label quality ablation",
        "Synthetic IAM benchmark · classical issue_type macro F1 · gold validation/test unchanged",
    )
    _grid(lines, low, high, decimals=2)
    max_fraction = max(fractions) or 1.0

    def x_position(value: float) -> float:
        return _LEFT + (value / max_fraction) * _PLOT_WIDTH

    for tick in fractions:
        x = x_position(tick)
        lines.append(
            f'<text class="axis" x="{x:.2f}" y="{_TOP + _PLOT_HEIGHT + 24}" text-anchor="middle">{tick:.0%}</text>'
        )
    for series, css_class, name, offset in (
        (validation, "series-a", "Validation", -10),
        (test, "series-b", "Locked test", 18),
    ):
        coordinates = [
            (x_position(fraction), _y(value, low, high))
            for fraction, value in zip(fractions, series, strict=True)
        ]
        path = " ".join(
            ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
            for index, (x, y) in enumerate(coordinates)
        )
        lines.append(f'<path d="{path}" class="{css_class}" fill="none" stroke-width="3"/>')
        for (x, y), value in zip(coordinates, series, strict=True):
            lines.append(f'<circle class="{css_class}" cx="{x:.2f}" cy="{y:.2f}" r="5"/>')
            lines.append(
                f'<text class="value" x="{x:.2f}" y="{y + offset:.2f}" text-anchor="middle">{value:.3f}</text>'
            )
        legend_y = _HEIGHT - 61
        legend_x = _LEFT + (0 if name == "Validation" else 145)
        lines.append(
            f'<line class="{css_class}" x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 24}" y2="{legend_y - 4}" stroke-width="3"/>'
        )
        lines.append(f'<text class="axis" x="{legend_x + 32}" y="{legend_y}">{name}</text>')
    lines.append(
        f'<text class="axis" x="{_LEFT + _PLOT_WIDTH / 2}" y="{_TOP + _PLOT_HEIGHT + 51}" text-anchor="middle">Injected training-label noise</text>'
    )
    lines.append(
        f'<text class="axis" transform="translate(24 {_TOP + _PLOT_HEIGHT / 2}) rotate(-90)" text-anchor="middle">Macro F1</text>'
    )
    return _footer(
        lines,
        "30% cyclic corruption lowered locked-test macro F1 to 0.960; this is a controlled stressor, not human error.",
    )


def render_figures(
    *,
    classical_path: Path,
    ablation_path: Path,
    output_directory: Path,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    classical = ClassicalExperimentArtifact.model_validate_json(
        classical_path.read_text(encoding="utf-8")
    )
    ablation = DataQualityAblationArtifact.model_validate_json(
        ablation_path.read_text(encoding="utf-8")
    )
    if classical.result_status != "measured" or ablation.result_status != "measured":
        raise ValueError("figures require measured artifacts")
    outputs = (
        output_directory / "classical-test-per-class-f1.svg",
        output_directory / "classical-latency.svg",
        output_directory / "classical-label-quality-ablation.svg",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite measured figures: {existing}")
    payloads: Iterable[str] = (
        render_per_class_f1(classical),
        render_latency(classical),
        render_label_quality(ablation),
    )
    for path, payload in zip(outputs, payloads, strict=True):
        atomic_write_bytes(path, payload.encode("utf-8"))
    return outputs


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> int:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classical",
        type=Path,
        default=root / "experiments" / "results" / "classical_tfidf_logreg_v1.json",
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        default=root / "experiments" / "results" / "classical_label_quality_ablation_v1.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=root / "experiments" / "figures",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for output in render_figures(
        classical_path=args.classical,
        ablation_path=args.ablation,
        output_directory=args.output_directory,
        overwrite=args.overwrite,
    ):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
