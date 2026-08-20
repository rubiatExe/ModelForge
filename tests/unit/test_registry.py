from modelforge.api.registry import ModelRegistry
from modelforge.models.heuristic import HeuristicTriageModel


def test_registry_is_an_allowlist() -> None:
    model = HeuristicTriageModel()
    registry = ModelRegistry((model,))
    assert registry.require(model.model_id) is model
    assert registry.get("unknown") is None
    assert registry.descriptors()[0].role == "demo"
