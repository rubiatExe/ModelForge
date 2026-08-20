"""Shared Pydantic configuration for ModelForge domain artifacts."""

from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    """Reject unknown fields and keep validated artifacts immutable."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )
