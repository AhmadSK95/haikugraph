"""Trust, SLO, and incident service wrapper."""

from __future__ import annotations

from typing import Any


class TrustService:
    def __init__(self, runtime_store: Any):
        self._runtime_store = runtime_store

    def trust_dashboard(self, *, tenant_id: str | None, hours: int = 168) -> dict[str, Any]:
        return dict(self._runtime_store.trust_dashboard(tenant_id=tenant_id, hours=hours) or {})

    def evaluate_slo(
        self,
        *,
        tenant_id: str | None,
        hours: int,
        min_runs: int,
        success_rate_target: float,
        p95_execution_ms_target: float,
        warning_rate_target: float,
    ) -> dict[str, Any]:
        return dict(
            self._runtime_store.evaluate_slo(
                tenant_id=tenant_id,
                hours=hours,
                min_runs=min_runs,
                success_rate_target=success_rate_target,
                p95_execution_ms_target=p95_execution_ms_target,
                warning_rate_target=warning_rate_target,
            )
            or {}
        )

    def stage_slo_snapshot(self, *, tenant_id: str | None, hours: int) -> dict[str, Any]:
        return dict(self._runtime_store.stage_slo_snapshot(tenant_id=tenant_id, hours=hours) or {})

    def list_incidents(self, *, tenant_id: str | None, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._runtime_store.list_incidents(tenant_id=tenant_id, limit=limit) or [])

    def update_incident_status(
        self,
        *,
        incident_id: str,
        status: str,
        note: str = "",
        acknowledged_by: str = "",
    ) -> bool:
        return bool(
            self._runtime_store.update_incident_status(
                incident_id=incident_id,
                status=status,
                note=note,
                acknowledged_by=acknowledged_by,
            )
        )
