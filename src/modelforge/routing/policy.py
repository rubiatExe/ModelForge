"""Versioned routing policy loaded from an experiment artifact."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RoutingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,63}$")
    version: str = Field(min_length=1, max_length=40)
    status: Literal["uncalibrated", "calibrated"]
    threshold: float = Field(ge=0.0, le=1.0)
    small_model_id: str = Field(min_length=1, max_length=200)
    frontier_model_id: str = Field(min_length=1, max_length=200)
    dataset_version: str = Field(min_length=1, max_length=80)
    validation_run_id: str | None = Field(default=None, max_length=100)
    created_at: datetime
    notes: str = Field(default="", max_length=1_000)

    @model_validator(mode="after")
    def calibrated_policy_has_evidence(self) -> RoutingPolicy:
        if self.status == "calibrated" and not self.validation_run_id:
            raise ValueError("a calibrated policy requires validation_run_id")
        return self

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def load_routing_policy(path: Path, *, require_calibrated: bool = False) -> RoutingPolicy:
    try:
        policy = RoutingPolicy.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"routing policy not found: {path}") from exc
    except Exception as exc:
        raise ValueError(f"invalid routing policy: {path}") from exc
    if require_calibrated and policy.status != "calibrated":
        raise ValueError("production requires a calibrated routing policy")
    return policy
