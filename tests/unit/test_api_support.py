from modelforge.api.contracts import EvaluationRunRequest
from modelforge.api.evaluation_store import EvaluationStore


def test_evaluation_store_state_transitions() -> None:
    store = EvaluationStore(max_runs=2)
    run = store.create(EvaluationRunRequest(model_id="small-v1", split="validation"))
    assert store.mark_running(run.run_id).status == "running"
    completed = store.complete(run.run_id, {"macro_f1": 0.9})
    assert completed.status == "complete"
    assert completed.report == {"macro_f1": 0.9}


def test_evaluation_store_is_bounded() -> None:
    store = EvaluationStore(max_runs=1)
    first = store.create(EvaluationRunRequest(model_id="small-v1"))
    second = store.create(EvaluationRunRequest(model_id="small-v1"))
    assert store.get(first.run_id) is None
    assert store.get(second.run_id) is not None
