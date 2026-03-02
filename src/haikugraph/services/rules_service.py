"""Business-rules API service wrapper."""

from __future__ import annotations

from typing import Any


class RulesService:
    def __init__(self, backend: Any):
        self._backend = backend

    def list_rules(self, *, tenant_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return list(self._backend.list_business_rules(tenant_id=tenant_id, limit=limit) or [])

    def create_rule(self, payload: dict[str, Any], *, tenant_id: str, created_by: str = "") -> str:
        return str(self._backend.create_business_rule(**payload, tenant_id=tenant_id, created_by=created_by) or "")

    def set_rule_status(self, *, rule_id: str, status: str, note: str, tenant_id: str, approved_by: str = "") -> dict[str, Any]:
        return dict(
            self._backend.set_business_rule_status(
                rule_id=rule_id,
                status=status,
                note=note,
                tenant_id=tenant_id,
                actor=approved_by,
            )
            or {}
        )

    def update_rule(self, payload: dict[str, Any], *, tenant_id: str, updated_by: str = "") -> dict[str, Any]:
        return dict(self._backend.update_business_rule(**payload, tenant_id=tenant_id, actor=updated_by) or {})

    def rollback_rule(self, rule_id: str, *, tenant_id: str, rolled_back_by: str = "") -> dict[str, Any]:
        return dict(
            self._backend.rollback_business_rule(
                rule_id,
                tenant_id=tenant_id,
                actor=rolled_back_by,
            )
            or {}
        )
