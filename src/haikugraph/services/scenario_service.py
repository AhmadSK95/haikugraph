"""Scenario set API service wrapper."""

from __future__ import annotations

from typing import Any


class ScenarioService:
    def __init__(self, runtime_store: Any):
        self._runtime_store = runtime_store

    def list_scenarios(self, *, tenant_id: str, connection_id: str, status: str = "", limit: int = 100) -> list[dict[str, Any]]:
        return list(
            self._runtime_store.list_scenario_sets(
                tenant_id=tenant_id,
                connection_id=connection_id,
                status=status,
                limit=limit,
            )
            or []
        )

    def upsert_scenario(self, payload: dict[str, Any], *, tenant_id: str, connection_id: str) -> dict[str, Any]:
        return dict(
            self._runtime_store.upsert_scenario_set(
                tenant_id=tenant_id,
                connection_id=connection_id,
                scenario_set_id=payload.get("scenario_set_id"),
                name=payload.get("name"),
                assumptions=list(payload.get("assumptions") or []),
                status=payload.get("status", "draft"),
            )
            or {}
        )

    def get_scenario(self, *, scenario_set_id: str, tenant_id: str) -> dict[str, Any] | None:
        row = self._runtime_store.get_scenario_set(
            scenario_set_id=scenario_set_id,
            tenant_id=tenant_id,
        )
        return dict(row) if row else None
