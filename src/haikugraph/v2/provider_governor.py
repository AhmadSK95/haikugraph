"""Provider governor for explicit degradation accounting and retry policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from haikugraph.v2.exceptions import ProviderDegradedError


@dataclass(frozen=True)
class RetryBudget:
    max_retries: int = 1
    backoff_ms: int = 250


@dataclass(frozen=True)
class ProviderOutcome:
    provider_effective: str
    fallback_used: dict[str, Any]


def ensure_provider_integrity(
    *,
    requested_mode: str,
    use_llm: bool,
    requested_provider: str | None,
    runtime_payload: dict[str, Any] | None,
    strict: bool = True,
) -> ProviderOutcome:
    """Normalize provider-effective metadata and detect hidden fallback."""
    payload = dict(runtime_payload or {})
    provider_effective = str(payload.get("provider") or requested_provider or "deterministic")
    degraded = bool(payload.get("llm_degraded"))
    degraded_reason = str(payload.get("llm_degraded_reason") or "")
    fallback = {
        "used": degraded,
        "reason": degraded_reason,
        "requested_mode": requested_mode,
        "requested_provider": str(requested_provider or ""),
    }

    if use_llm and strict and requested_mode in {"local", "openai", "anthropic"}:
        llm_effective = bool(payload.get("llm_effective"))
        if not llm_effective:
            raise ProviderDegradedError(
                f"Explicit provider '{requested_mode}' unavailable: {degraded_reason or 'no LLM step executed'}"
            )
    return ProviderOutcome(provider_effective=provider_effective, fallback_used=fallback)

