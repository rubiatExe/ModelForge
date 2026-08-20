"""Single-sequence chat tokenization with response-only supervision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from modelforge.models.base import ModelDependencyError

IGNORE_INDEX = -100


class TrainingTokenizationError(ValueError):
    """An example cannot safely produce response-only training loss."""


@dataclass(frozen=True, slots=True)
class EncodedChatSequence:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    supervised_tokens: int
    was_truncated: bool

    def as_features(self) -> dict[str, list[int]]:
        return {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "labels": list(self.labels),
        }


def _flat_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise TrainingTokenizationError(f"tokenizer {field} must be a list")
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise TrainingTokenizationError("batched tokenization is not supported here")
        return list(value[0])
    return list(value)


def _flat_offsets(value: Any) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise TrainingTokenizationError("tokenizer offset_mapping must be a list")
    # A normal single example is ``[(start, end), ...]``; a batched result is
    # ``[[(start, end), ...]]``. Offset tuples must not be mistaken for a batch.
    if (
        len(value) == 1
        and isinstance(value[0], list)
        and (not value[0] or isinstance(value[0][0], (list, tuple)))
    ):
        return list(value[0])
    return list(value)


def tokenize_response_only(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    max_length: int,
    minimum_response_tokens: int = 1,
) -> EncodedChatSequence:
    """Tokenize the complete conversation once and mask all pre-response loss.

    Character offsets from a fast tokenizer locate the assistant span in that
    one sequence. We never assume that separately tokenized prompt IDs are a
    prefix of prompt-plus-response IDs. Truncation that would remove the entire
    response is a hard data error rather than an all-``-100`` training example.
    """

    if max_length <= 0 or minimum_response_tokens <= 0:
        raise ValueError("max_length and minimum_response_tokens must be positive")
    if not messages or messages[-1].get("role") != "assistant":
        raise TrainingTokenizationError("the final chat message must be the assistant response")
    response = messages[-1].get("content")
    if not isinstance(response, str) or not response:
        raise TrainingTokenizationError("assistant response cannot be empty")
    try:
        rendered = tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as exc:
        raise TrainingTokenizationError("tokenizer could not render the chat template") from exc
    if not isinstance(rendered, str):
        raise TrainingTokenizationError("chat template did not render text")
    response_start_character = rendered.rfind(response)
    if response_start_character < 0:
        raise TrainingTokenizationError("assistant response was transformed by the chat template")

    try:
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        input_ids = _flat_list(encoded["input_ids"], "input_ids")
        offsets = _flat_offsets(encoded["offset_mapping"])
    except TrainingTokenizationError:
        raise
    except Exception as exc:
        raise TrainingTokenizationError(
            "response masking requires a fast tokenizer with offset_mapping"
        ) from exc
    if len(input_ids) != len(offsets) or not input_ids:
        raise TrainingTokenizationError("tokenizer returned inconsistent IDs and offsets")

    first_response_token: int | None = None
    for index, offset in enumerate(offsets):
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise TrainingTokenizationError("tokenizer returned an invalid offset")
        start, end = int(offset[0]), int(offset[1])
        if end > response_start_character and end > start:
            first_response_token = index
            break
    if first_response_token is None:
        raise TrainingTokenizationError("assistant response produced no tokens")

    labels = [IGNORE_INDEX] * first_response_token + [int(value) for value in input_ids[first_response_token:]]
    was_truncated = len(input_ids) > max_length
    input_ids = [int(value) for value in input_ids[:max_length]]
    labels = labels[:max_length]
    supervised = sum(value != IGNORE_INDEX for value in labels)
    if supervised < minimum_response_tokens:
        raise TrainingTokenizationError(
            "truncation left too few supervised response tokens; reduce the prompt or increase max_length"
        )
    return EncodedChatSequence(
        input_ids=tuple(input_ids),
        attention_mask=tuple(1 for _ in input_ids),
        labels=tuple(labels),
        supervised_tokens=supervised,
        was_truncated=was_truncated,
    )


def pad_response_only_batch(
    batch: Sequence[Mapping[str, Sequence[int]]],
    *,
    pad_token_id: int,
    pad_to_multiple_of: int | None = 8,
) -> dict[str, list[list[int]]]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    lengths = [len(item["input_ids"]) for item in batch]
    if any(length == 0 for length in lengths):
        raise TrainingTokenizationError("cannot collate an empty sequence")
    max_length = max(lengths)
    if pad_to_multiple_of:
        max_length = ((max_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    output = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        input_ids = list(item["input_ids"])
        attention_mask = list(item["attention_mask"])
        labels = list(item["labels"])
        if not (len(input_ids) == len(attention_mask) == len(labels)):
            raise TrainingTokenizationError("feature lengths do not match")
        if all(label == IGNORE_INDEX for label in labels):
            raise TrainingTokenizationError("batch contains an example with no supervised tokens")
        pad_count = max_length - len(input_ids)
        output["input_ids"].append(input_ids + [pad_token_id] * pad_count)
        output["attention_mask"].append(attention_mask + [0] * pad_count)
        output["labels"].append(labels + [IGNORE_INDEX] * pad_count)
    return output


class ResponseOnlyDataCollator:
    """Right-pad response-only features and lazily construct Torch tensors."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        pad_to_multiple_of: int | None = 8,
        torch_module: Any | None = None,
    ) -> None:
        if tokenizer.pad_token_id is None:
            raise TrainingTokenizationError("tokenizer must define pad_token_id")
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of
        self._torch = torch_module

    def __call__(self, batch: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, Any]:
        padded = pad_response_only_batch(
            batch,
            pad_token_id=self.pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        torch = self._torch
        if torch is None:
            try:
                import torch
            except ImportError as exc:
                raise ModelDependencyError(
                    "training collation requires the 'train' extra",
                    cause=exc,
                ) from exc
        return {name: torch.tensor(values, dtype=torch.long) for name, values in padded.items()}
