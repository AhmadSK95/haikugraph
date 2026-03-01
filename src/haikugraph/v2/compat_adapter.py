"""Compatibility adapter that projects v2 diagnostics onto existing response model."""

from __future__ import annotations

import hashlib
import json

from haikugraph.agents.contracts import AssistantQueryResponse
from haikugraph.v2.types import AssistantResponseV2


def _slice_signature(response: AssistantQueryResponse) -> str:
    payload = {
        "sql": str(response.sql or ""),
        "contract": dict(response.contract_spec or {}),
        "runtime_mode": str((response.runtime or {}).get("mode") or ""),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def apply_v2_compat_fields(
    response: AssistantQueryResponse,
    v2: AssistantResponseV2,
    *,
    analysis_version: str = "v2",
) -> AssistantQueryResponse:
    quality = v2.quality
    insight = v2.insight
    runtime_payload = dict(response.runtime or {})
    response.analysis_version = analysis_version
    response.slice_signature = v2.slice_signature or _slice_signature(response)
    response.stage_timings_ms = dict(v2.stage_timings_ms or {})
    response.quality_flags = list((quality.quality_flags if quality else []) or [])
    response.truth_score = float(quality.truth_score) if quality else None
    response.provider_effective = str((quality.provider_effective if quality else "") or "")
    response.fallback_used = dict((quality.fallback_used if quality else {}) or {})
    response.assumptions = list((insight.assumptions if insight else []) or [])
    runtime_payload["stage_events"] = [e.model_dump() for e in (v2.stage_events or [])]
    runtime_payload["stage_state"] = "completed" if v2.stage_events else "unknown"
    response.runtime = runtime_payload
    return response
