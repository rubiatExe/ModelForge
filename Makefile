.PHONY: install test lint data baseline ablation figures api

install:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .

data:
	python scripts/generate_dataset.py

baseline:
	python -m modelforge.experiments.run_classical

ablation:
	python -m modelforge.experiments.run_data_ablation

figures:
	python -m modelforge.experiments.render_measured_figures --overwrite

api:
	uvicorn modelforge.api.app:app --reload
