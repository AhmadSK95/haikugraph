"""Execution result helpers for v2."""

from __future__ import annotations

from haikugraph.agents.contracts import AssistantQueryResponse
from haikugraph.v2.types import ExecutionResultV2


def summarize_execution(response: AssistantQueryResponse) -> ExecutionResultV2:
    return ExecutionResultV2(
        success=bool(response.success),
        row_count=int(response.row_count or 0),
        latency_ms=float(response.execution_time_ms or 0.0),
    )

