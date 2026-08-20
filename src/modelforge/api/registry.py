"""Small allowlisted runtime registry with injected model instances."""

from __future__ import annotations

import threading

from modelforge.api.contracts import ModelDescriptor
from modelforge.models.base import TriageModel


class ModelRegistry:
    def __init__(self, models: tuple[TriageModel, ...] = ()) -> None:
        self._models: dict[str, TriageModel] = {}
        self._lock = threading.Lock()
        for model in models:
            self.register(model)

    def register(self, model: TriageModel) -> None:
        if not isinstance(model, TriageModel):
            raise TypeError("model does not satisfy the TriageModel protocol")
        with self._lock:
            if model.model_id in self._models:
                raise ValueError(f"model already registered: {model.model_id}")
            self._models[model.model_id] = model

    def get(self, model_id: str) -> TriageModel | None:
        with self._lock:
            return self._models.get(model_id)

    def require(self, model_id: str) -> TriageModel:
        model = self.get(model_id)
        if model is None:
            raise KeyError(model_id)
        return model

    def descriptors(self) -> list[ModelDescriptor]:
        with self._lock:
            models = tuple(self._models.values())
        return [
            ModelDescriptor(
                model_id=model.model_id,
                role=model.role,
                ready=True,
                note=(
                    "Local deterministic harness fixture; not empirical model evidence."
                    if model.role == "demo"
                    else ""
                ),
            )
            for model in sorted(models, key=lambda item: item.model_id)
        ]
