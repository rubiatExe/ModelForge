FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY prompts ./prompts
COPY experiments/results ./experiments/results
COPY data/iam_triage_v1 ./data/iam_triage_v1
RUN python -m pip install --upgrade pip && python -m pip install .

RUN useradd --create-home --uid 10001 modelforge
USER modelforge

EXPOSE 8000
CMD ["uvicorn", "modelforge.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
