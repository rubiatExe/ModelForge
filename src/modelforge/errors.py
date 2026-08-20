"""Shared exceptions with safe, user-facing messages."""

from __future__ import annotations


class ModelForgeError(Exception):
    """Base class for expected ModelForge failures."""


class ConfigurationError(ModelForgeError):
    """Runtime configuration is absent or invalid."""


class ModelInferenceError(ModelForgeError):
    """A model could not produce a valid prediction."""


class FrontierUnavailableError(ModelForgeError):
    """A required escalation could not reach a configured frontier model."""


class DatasetIntegrityError(ModelForgeError):
    """A dataset failed schema, leakage, provenance, or hash checks."""


class EvaluationError(ModelForgeError):
    """An evaluation could not produce comparable evidence."""
