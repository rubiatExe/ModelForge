from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import modelforge.models.prompting as prompting_module
from modelforge.models import (
    FrontierCompletion,
    FrontierTriageModel,
    HeuristicTriageModel,
    HuggingFaceGeneration,
    HuggingFaceModelConfig,
    HuggingFaceTriageModel,
    ModelConfigurationError,
    ModelOutputError,
    ModelTimeoutError,
    OpenAICompatibleConfig,
    OpenAICompatibleFrontierProvider,
    TfidfIssueTypeModel,
    TransformersBackend,
    TriageModel,
    load_iam_prompt,
    parse_triage_json,
    render_iam_messages,
)
from modelforge.models._prompt_resource import IAM_TRIAGE_PROMPT_TEXT
from modelforge.schemas import IssueType, TicketInput


def ticket() -> TicketInput:
    return TicketInput(
        ticket_id="TKT-1001",
        subject="MFA push fails for our team",
        body="Three teammates cannot approve the MFA push after signing in.",
        employee_department="Sales",
        submitted_at=datetime(2026, 8, 20, tzinfo=UTC),
    )


def result_payload() -> dict[str, Any]:
    return {
        "issue_type": "MFA_FAILURE",
        "severity": "P2",
        "routing_team": "IDENTITY_PLATFORM",
        "affected_scope": "MULTIPLE_USERS",
        "recommended_action": "INVESTIGATE_MFA_CONFIGURATION",
        "evidence": ["MFA push", "Three teammates"],
        "confidence": 0.84,
    }


def result_json() -> str:
    return json.dumps(result_payload(), separators=(",", ":"))


def test_prompt_artifact_is_versioned_packaged_and_marks_ticket_untrusted() -> None:
    prompt = load_iam_prompt()
    assert prompt.version == "iam_triage_v1"
    assert len(prompt.sha256) == 64
    assert Path("prompts/iam_triage_v1.txt").read_text(encoding="utf-8") == IAM_TRIAGE_PROMPT_TEXT
    messages = render_iam_messages(ticket(), prompt=prompt)
    assert messages[0]["role"] == "system"
    assert "untrusted data" in messages[0]["content"]
    assert '"ticket_id":"TKT-1001"' in messages[1]["content"]
    assert "{{ticket_json}}" not in messages[1]["content"]


def test_prompt_loader_uses_packaged_fallback_when_repo_artifact_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompting_module,
        "default_prompt_path",
        lambda: tmp_path / "missing-root-prompt.txt",
    )
    prompt = prompting_module.load_iam_prompt()
    assert prompt.version == "iam_triage_v1"
    assert prompt.path is None


def test_strict_parser_accepts_valid_enum_json() -> None:
    parsed = parse_triage_json(result_json())
    assert parsed.issue_type is IssueType.MFA_FAILURE
    assert parsed.confidence == 0.84


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n{}\n```",
        "before " + result_json(),
        json.dumps({**result_payload(), "extra": True}),
        result_json().replace('"confidence":0.84', '"confidence":0.84,"confidence":0.7'),
        result_json().replace("0.84", "NaN"),
        "[]",
    ],
)
def test_strict_parser_rejects_wrappers_duplicates_nonfinite_and_shape(raw: str) -> None:
    with pytest.raises(ModelOutputError) as captured:
        parse_triage_json(raw)
    assert captured.value.raw_output == raw
    assert raw not in str(captured.value)


def test_heuristic_is_a_privacy_safe_typed_demo_model() -> None:
    async def scenario() -> None:
        model = HeuristicTriageModel()
        assert isinstance(model, TriageModel)
        prediction = await model.triage(ticket())
        assert prediction.result.issue_type is IssueType.MFA_FAILURE
        assert prediction.metadata["local_only"] is True
        serialized = prediction.model_dump_json()
        assert ticket().ticket_id not in serialized
        assert ticket().body not in serialized
        assert prediction.raw_output is None

    asyncio.run(scenario())


class _FakeFrontierProvider:
    provider_id = "fake-frontier"

    async def complete(self, messages: Any, *, timeout_s: float) -> FrontierCompletion:
        assert timeout_s == 2.0
        assert messages[0]["role"] == "system"
        return FrontierCompletion(
            content=result_json(),
            model_version="frontier-fixture-2026-08-20",
            input_tokens=100,
            output_tokens=40,
            estimated_cost_usd=0.004,
            provider_request_id="provider-request-1",
        )


def test_provider_neutral_frontier_model_captures_usage_and_validates() -> None:
    async def scenario() -> None:
        model = FrontierTriageModel(_FakeFrontierProvider(), default_timeout_s=2.0)
        prediction = await model.triage(ticket())
        assert prediction.result.issue_type is IssueType.MFA_FAILURE
        assert prediction.input_tokens == 100
        assert prediction.output_tokens == 40
        assert prediction.estimated_cost_usd == 0.004
        assert prediction.provider_request_id == "provider-request-1"
        assert "raw_output" not in prediction.model_dump()

    asyncio.run(scenario())


def test_frontier_rejects_schema_valid_but_unsupported_evidence() -> None:
    payload = result_payload()
    payload["evidence"] = ["invented administrator diagnosis"]

    class UnsupportedEvidenceProvider:
        provider_id = "unsupported-evidence-fixture"

        async def complete(
            self,
            messages: Any,
            *,
            timeout_s: float,
        ) -> FrontierCompletion:
            return FrontierCompletion(
                content=json.dumps(payload),
                model_version="frontier-fixture",
                input_tokens=10,
                output_tokens=8,
                estimated_cost_usd=0.001,
            )

    async def scenario() -> None:
        model = FrontierTriageModel(UnsupportedEvidenceProvider())
        with pytest.raises(ModelOutputError) as captured:
            await model.triage(ticket())
        assert "not present" in str(captured.value)
        assert captured.value.input_tokens == 10
        assert captured.value.output_tokens == 8
        assert captured.value.estimated_cost_usd == 0.001
        assert "invented administrator diagnosis" not in str(captured.value)

    asyncio.run(scenario())


def test_frontier_invalid_completion_retains_raw_output_without_echoing_it() -> None:
    raw = '{"ticket_body":"sensitive malformed completion"}'

    class MalformedProvider:
        provider_id = "malformed-frontier"

        async def complete(
            self,
        messages: Any,
        *,
        timeout_s: float,
    ) -> FrontierCompletion:
            return FrontierCompletion(content=raw, model_version="malformed-fixture")

    async def scenario() -> None:
        model = FrontierTriageModel(MalformedProvider())
        with pytest.raises(ModelOutputError) as captured:
            await model.triage(ticket())
        assert captured.value.raw_output == raw
        assert raw not in str(captured.value)
        assert raw not in repr(captured.value)

    asyncio.run(scenario())


class _FakeResponse:
    status_code = 200
    headers = {"x-request-id": "header-request"}

    def json(self) -> dict[str, Any]:
        return {
            "id": "body-request",
            "model": "frontier-revision",
            "choices": [{"message": {"content": result_json()}}],
            "usage": {"prompt_tokens": 1_000, "completion_tokens": 100},
        }


class _FakeHttpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((url, kwargs))
        return _FakeResponse()

    async def aclose(self) -> None:
        self.closed = True


def test_openai_compatible_adapter_uses_injected_client_and_prices_tokens() -> None:
    async def scenario() -> None:
        client = _FakeHttpClient()
        config = OpenAICompatibleConfig(
            base_url="https://provider.invalid/v1",
            api_key="top-secret",
            model="frontier-alias",
            input_usd_per_million=2.0,
            output_usd_per_million=8.0,
            pricing_source="fixture price card",
            pricing_as_of="2026-08-20",
        )
        provider = OpenAICompatibleFrontierProvider(config, client=client)
        completion = await provider.complete(
            [{"role": "user", "content": "fixture"}],
            timeout_s=1.0,
        )
        assert completion.estimated_cost_usd == pytest.approx(0.0028)
        assert completion.provider_request_id == "body-request"
        assert client.calls[0][0] == "https://provider.invalid/v1/chat/completions"
        assert client.calls[0][1]["json"]["response_format"] == {"type": "json_object"}
        assert "top-secret" not in repr(config)

    asyncio.run(scenario())


class _SlowHttpClient(_FakeHttpClient):
    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        await asyncio.sleep(0.1)
        return _FakeResponse()


def test_openai_compatible_adapter_enforces_deadline() -> None:
    async def scenario() -> None:
        provider = OpenAICompatibleFrontierProvider(
            OpenAICompatibleConfig(
                base_url="https://provider.invalid/v1",
                api_key="secret",
                model="frontier",
            ),
            client=_SlowHttpClient(),
        )
        with pytest.raises(ModelTimeoutError):
            await provider.complete([{"role": "user", "content": "fixture"}], timeout_s=0.01)

    asyncio.run(scenario())


def test_openai_environment_configuration_errors_are_typed() -> None:
    with pytest.raises(ModelConfigurationError):
        OpenAICompatibleConfig.from_env({})
    with pytest.raises(ModelConfigurationError):
        OpenAICompatibleConfig.from_env(
            {
                "MODELFORGE_FRONTIER_BASE_URL": "https://provider.invalid/v1",
                "MODELFORGE_FRONTIER_API_KEY": "secret",
                "MODELFORGE_FRONTIER_MODEL": "frontier",
                "MODELFORGE_FRONTIER_TIMEOUT_SECONDS": "not-a-number",
            }
        )


def test_nonzero_frontier_prices_require_source_and_date() -> None:
    with pytest.raises(ValueError, match="pricing_source"):
        OpenAICompatibleConfig(
            base_url="https://provider.invalid/v1",
            api_key="secret",
            model="frontier",
            input_usd_per_million=1.0,
        )


class _FakeHuggingFaceBackend:
    async def generate(self, messages: Any, *, timeout_s: float) -> HuggingFaceGeneration:
        assert messages[-1]["role"] == "user"
        return HuggingFaceGeneration(
            content=result_json(),
            model_version="deadbeef+adapter:fixture",
            input_tokens=123,
            output_tokens=45,
        )


def test_huggingface_adapter_is_testable_without_ml_dependencies() -> None:
    async def scenario() -> None:
        config = HuggingFaceModelConfig(
            model_name_or_path="fixture/base",
            revision="deadbeef",
            adapter_name_or_path="fixture/adapter",
        )
        model = HuggingFaceTriageModel(config, backend=_FakeHuggingFaceBackend())
        prediction = await model.triage(ticket())
        assert prediction.model_version == "deadbeef+adapter:fixture"
        assert prediction.result.issue_type is IssueType.MFA_FAILURE
        assert prediction.metadata["adapter_loaded"] is True

    asyncio.run(scenario())


def test_huggingface_rejects_schema_valid_but_unsupported_evidence() -> None:
    payload = result_payload()
    payload["evidence"] = ["a fabricated root cause"]

    class UnsupportedEvidenceBackend:
        async def generate(
            self,
            messages: Any,
            *,
            timeout_s: float,
        ) -> HuggingFaceGeneration:
            return HuggingFaceGeneration(
                content=json.dumps(payload),
                model_version="hf-fixture",
                input_tokens=12,
                output_tokens=9,
            )

    async def scenario() -> None:
        config = HuggingFaceModelConfig(
            model_name_or_path="fixture/base",
            revision="deadbeef",
        )
        model = HuggingFaceTriageModel(config, backend=UnsupportedEvidenceBackend())
        with pytest.raises(ModelOutputError) as captured:
            await model.triage(ticket())
        assert "not present" in str(captured.value)
        assert captured.value.model_version == "hf-fixture"
        assert "a fabricated root cause" not in str(captured.value)

    asyncio.run(scenario())


def test_huggingface_invalid_generation_retains_raw_output_without_echoing_it() -> None:
    raw = "malformed completion containing sensitive ticket text"

    class MalformedBackend:
        async def generate(
            self,
        messages: Any,
        *,
        timeout_s: float,
    ) -> HuggingFaceGeneration:
            return HuggingFaceGeneration(
                content=raw,
                model_version="malformed-fixture",
                input_tokens=10,
                output_tokens=4,
            )

    async def scenario() -> None:
        config = HuggingFaceModelConfig(
            model_name_or_path="fixture/base",
            revision="deadbeef",
        )
        model = HuggingFaceTriageModel(config, backend=MalformedBackend())
        with pytest.raises(ModelOutputError) as captured:
            await model.triage(ticket())
        assert captured.value.raw_output == raw
        assert raw not in str(captured.value)
        assert raw not in repr(captured.value)

    asyncio.run(scenario())


def test_huggingface_native_timeout_never_allows_concurrent_model_calls() -> None:
    async def scenario() -> None:
        backend = TransformersBackend(
            HuggingFaceModelConfig(model_name_or_path="fixture/base", revision="deadbeef")
        )
        backend._model = object()  # avoid lazy dependency loading in this concurrency unit test
        started = threading.Event()
        release = threading.Event()
        state = {"active": 0, "max_active": 0, "calls": 0}
        state_lock = threading.Lock()

        def blocking_generate(messages: Any) -> HuggingFaceGeneration:
            with state_lock:
                state["active"] += 1
                state["calls"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            started.set()
            release.wait(timeout=1.0)
            with state_lock:
                state["active"] -= 1
            return HuggingFaceGeneration(
                content=result_json(),
                model_version="fixture",
                input_tokens=1,
                output_tokens=1,
            )

        backend._generate_sync = blocking_generate  # type: ignore[method-assign]
        try:
            with pytest.raises(ModelTimeoutError):
                await backend.generate([], timeout_s=0.05)
            assert started.is_set()
            second = asyncio.create_task(backend.generate([], timeout_s=0.5))
            await asyncio.sleep(0.02)
            assert state["calls"] == 1  # second call is queued behind timed-out native work
            release.set()
            await second
            assert state["calls"] == 2
            assert state["max_active"] == 1
        finally:
            release.set()
            backend.close()

    asyncio.run(scenario())


def test_classical_baseline_fit_save_load_when_optional_dependency_is_available(
    tmp_path: Path,
) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    model = TfidfIssueTypeModel(random_state=7).fit(
        [
            "MFA authenticator push failed",
            "verification code MFA rejected",
            "account locked after attempts",
            "disabled user account",
        ],
        [
            IssueType.MFA_FAILURE,
            IssueType.MFA_FAILURE,
            IssueType.ACCOUNT_LOCKED_OR_DISABLED,
            IssueType.ACCOUNT_LOCKED_OR_DISABLED,
        ],
    )
    before = model.predict_issue_type("MFA push cannot approve")
    artifact = model.save(tmp_path / "baseline.joblib")
    loaded = TfidfIssueTypeModel.load(artifact)
    after = loaded.predict_issue_type("MFA push cannot approve")
    assert before.issue_type is IssueType.MFA_FAILURE
    assert before == after
    assert loaded.model_version == model.model_version
