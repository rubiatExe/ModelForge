"""TF-IDF + logistic-regression issue-type baseline.

Scikit-learn and joblib are imported only when the baseline is fitted, loaded,
or saved. Loading a joblib artifact can execute Python, so ``load`` must only be
used with a trusted ModelForge-produced path.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from modelforge.models.base import (
    ModelConfigurationError,
    ModelDependencyError,
    ModelPrediction,
    ModelRole,
)
from modelforge.models.heuristic import (
    HeuristicTriageModel,
    action_for_issue,
    routing_for_issue,
)
from modelforge.schemas import IssueType, TicketInput
from modelforge.schemas._base import StrictBaseModel

_ARTIFACT_VERSION = "1"


class IssueTypePrediction(StrictBaseModel):
    issue_type: IssueType
    confidence: float = Field(ge=0.0, le=1.0)


def _require_sklearn() -> tuple[type[Any], type[Any]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ModelDependencyError(
            "the classical baseline requires scikit-learn; install the project dependencies",
            cause=exc,
        ) from exc
    return TfidfVectorizer, LogisticRegression


def _require_joblib() -> Any:
    try:
        import joblib
    except ImportError as exc:
        raise ModelDependencyError(
            "saving/loading the classical baseline requires joblib",
            cause=exc,
        ) from exc
    return joblib


def ticket_text(ticket: TicketInput) -> str:
    return f"{ticket.subject}\n{ticket.body}"


class TfidfIssueTypeModel:
    """A reproducible issue-type baseline with a full triage adapter wrapper."""

    role: ModelRole = "classical"

    def __init__(
        self,
        *,
        model_id: str = "tfidf-logreg-issue-type-v1",
        random_state: int = 17,
        max_features: int = 20_000,
        max_iter: int = 1_000,
    ) -> None:
        if not model_id.strip():
            raise ModelConfigurationError("classical model_id cannot be empty")
        self.model_id = model_id
        self.random_state = random_state
        self.max_features = max_features
        self.max_iter = max_iter
        self._vectorizer: Any | None = None
        self._classifier: Any | None = None
        self._model_version = "untrained"
        self._fallback = HeuristicTriageModel()

    @property
    def is_fitted(self) -> bool:
        return self._vectorizer is not None and self._classifier is not None

    @property
    def model_version(self) -> str:
        return self._model_version

    def fit(
        self,
        tickets: Sequence[TicketInput | str],
        labels: Sequence[IssueType | str],
    ) -> TfidfIssueTypeModel:
        if not tickets or len(tickets) != len(labels):
            raise ModelConfigurationError("tickets and labels must have the same non-zero length")
        try:
            normalized_labels = [IssueType(label).value for label in labels]
        except ValueError as exc:
            raise ModelConfigurationError("classical labels must be valid IssueType values", cause=exc) from exc
        if len(set(normalized_labels)) < 2:
            raise ModelConfigurationError("logistic regression requires at least two issue classes")
        texts = [ticket_text(item) if isinstance(item, TicketInput) else str(item) for item in tickets]
        Vectorizer, Classifier = _require_sklearn()
        vectorizer = Vectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=self.max_features,
        )
        features = vectorizer.fit_transform(texts)
        classifier = Classifier(
            class_weight="balanced",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        classifier.fit(features, normalized_labels)
        fingerprint_rows = [
            {"text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "label": label}
            for text, label in zip(texts, normalized_labels, strict=True)
        ]
        identity = {
            "artifact_version": _ARTIFACT_VERSION,
            "random_state": self.random_state,
            "max_features": self.max_features,
            "max_iter": self.max_iter,
            "rows": fingerprint_rows,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._vectorizer = vectorizer
        self._classifier = classifier
        self._model_version = digest[:16]
        return self

    def predict_issue_type(self, ticket: TicketInput | str) -> IssueTypePrediction:
        if not self.is_fitted:
            raise ModelConfigurationError("classical baseline has not been fitted or loaded")
        text = ticket_text(ticket) if isinstance(ticket, TicketInput) else str(ticket)
        features = self._vectorizer.transform([text])
        probabilities = self._classifier.predict_proba(features)[0]
        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return IssueTypePrediction(
            issue_type=IssueType(self._classifier.classes_[best_index]),
            confidence=float(probabilities[best_index]),
        )

    async def triage(
        self,
        ticket: TicketInput,
        *,
        timeout_s: float | None = None,
    ) -> ModelPrediction:
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        started = time.perf_counter()
        issue = self.predict_issue_type(ticket)
        fallback = await self._fallback.triage(ticket, timeout_s=timeout_s)
        action = fallback.result.recommended_action
        if action.value != "ESCALATE_SECURITY_REVIEW":
            action = action_for_issue(issue.issue_type)
        result = fallback.result.model_copy(
            update={
                "issue_type": issue.issue_type,
                "routing_team": routing_for_issue(issue.issue_type),
                "recommended_action": action,
                "confidence": issue.confidence,
            }
        )
        return ModelPrediction(
            result=result,
            model_id=self.model_id,
            model_version=self.model_version,
            role=self.role,
            latency_ms=(time.perf_counter() - started) * 1_000,
            estimated_cost_usd=0.0,
            metadata={
                "task": "issue_type",
                "confidence_kind": "logistic_regression_class_probability",
                "non_issue_fields": "heuristic_demo",
            },
        )

    async def predict(
        self,
        ticket: TicketInput,
        *,
        timeout_s: float | None = None,
    ) -> ModelPrediction:
        return await self.triage(ticket, timeout_s=timeout_s)

    def save(self, path: Path) -> Path:
        """Atomically save an already-fitted trusted local artifact."""

        if not self.is_fitted:
            raise ModelConfigurationError("cannot save an unfitted classical baseline")
        joblib = _require_joblib()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "artifact_version": _ARTIFACT_VERSION,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "random_state": self.random_state,
            "max_features": self.max_features,
            "max_iter": self.max_iter,
            "vectorizer": self._vectorizer,
            "classifier": self._classifier,
        }
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
                temporary_name = handle.name
            try:
                joblib.dump(bundle, temporary_name)
            except Exception as exc:
                raise ModelConfigurationError("failed to save classical baseline artifact", cause=exc) from exc
            os.replace(temporary_name, path)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return path

    @classmethod
    def load(cls, path: Path) -> TfidfIssueTypeModel:
        """Load a trusted artifact. Joblib files are not safe for untrusted input."""

        joblib = _require_joblib()
        try:
            bundle = joblib.load(Path(path))
        except Exception as exc:
            raise ModelConfigurationError(
                "classical baseline artifact is unavailable or invalid",
                cause=exc,
            ) from exc
        required = {
            "artifact_version",
            "model_id",
            "model_version",
            "random_state",
            "max_features",
            "max_iter",
            "vectorizer",
            "classifier",
        }
        if not isinstance(bundle, dict) or set(bundle) != required:
            raise ModelConfigurationError("classical baseline artifact has an invalid structure")
        if bundle["artifact_version"] != _ARTIFACT_VERSION:
            raise ModelConfigurationError("unsupported classical baseline artifact version")
        instance = cls(
            model_id=bundle["model_id"],
            random_state=bundle["random_state"],
            max_features=bundle["max_features"],
            max_iter=bundle["max_iter"],
        )
        instance._vectorizer = bundle["vectorizer"]
        instance._classifier = bundle["classifier"]
        instance._model_version = bundle["model_version"]
        if not hasattr(instance._vectorizer, "transform") or not hasattr(
            instance._classifier, "predict_proba"
        ):
            raise ModelConfigurationError("classical baseline artifact contains invalid estimators")
        return instance
