"""Typed exception taxonomy for v2 runtime control flow."""

from __future__ import annotations


class V2RuntimeError(RuntimeError):
    """Base class for v2 runtime failures."""


class PolicyViolationError(V2RuntimeError):
    """Raised when an input violates policy or governance constraints."""


class PlanningError(V2RuntimeError):
    """Raised when no valid execution plan can be produced."""


class QueryCompilationError(V2RuntimeError):
    """Raised when SQL compilation fails."""


class QueryExecutionError(V2RuntimeError):
    """Raised when SQL execution fails."""


class ProviderDegradedError(V2RuntimeError):
    """Raised when requested provider cannot serve a required LLM operation."""


class ContradictionDetectedError(V2RuntimeError):
    """Raised when intent contains hard contradictions requiring clarification."""

