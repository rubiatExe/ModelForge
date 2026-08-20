"""Model adapters and their shared asynchronous contract."""

from modelforge.models.base import (
    ModelAdapterError,
    ModelConfigurationError,
    ModelDependencyError,
    ModelInferenceError,
    ModelLoadError,
    ModelOutputError,
    ModelPrediction,
    ModelProviderError,
    ModelRole,
    ModelTimeoutError,
    TriageModel,
)
from modelforge.models.classical import IssueTypePrediction, TfidfIssueTypeModel
from modelforge.models.frontier import (
    FrontierCompletion,
    FrontierProvider,
    FrontierTriageModel,
    OpenAICompatibleConfig,
    OpenAICompatibleFrontierProvider,
)
from modelforge.models.heuristic import HeuristicTriageModel
from modelforge.models.huggingface import (
    HuggingFaceGeneration,
    HuggingFaceModelConfig,
    HuggingFaceTriageModel,
    TransformersBackend,
)
from modelforge.models.prompting import (
    IAM_PROMPT_VERSION,
    PromptDefinition,
    load_iam_prompt,
    parse_triage_json,
    render_iam_messages,
    require_grounded_evidence,
)

__all__ = [
    "HeuristicTriageModel",
    "IAM_PROMPT_VERSION",
    "IssueTypePrediction",
    "ModelAdapterError",
    "ModelConfigurationError",
    "ModelDependencyError",
    "ModelInferenceError",
    "ModelLoadError",
    "ModelOutputError",
    "ModelPrediction",
    "ModelProviderError",
    "ModelRole",
    "ModelTimeoutError",
    "PromptDefinition",
    "FrontierCompletion",
    "FrontierProvider",
    "FrontierTriageModel",
    "HuggingFaceGeneration",
    "HuggingFaceModelConfig",
    "HuggingFaceTriageModel",
    "OpenAICompatibleConfig",
    "OpenAICompatibleFrontierProvider",
    "TfidfIssueTypeModel",
    "TransformersBackend",
    "TriageModel",
    "load_iam_prompt",
    "parse_triage_json",
    "render_iam_messages",
    "require_grounded_evidence",
]
