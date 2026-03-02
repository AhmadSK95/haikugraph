"""Corrections API service wrapper."""

from __future__ import annotations

from typing import Any


class CorrectionsService:
    """Adapter over existing correction APIs exposed by runtime backend."""

    def __init__(self, backend: Any):
        self._backend = backend

    def record_feedback(self, payload: dict[str, Any], *, tenant_id: str) -> dict[str, Any]:
        return self._backend.record_feedback(**payload, tenant_id=tenant_id)

    def list_corrections(self, *, tenant_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return list(self._backend.list_corrections(tenant_id=tenant_id, limit=limit) or [])

    def set_correction_enabled(self, *, correction_id: str, enabled: bool, tenant_id: str) -> bool:
        return bool(
            self._backend.set_correction_enabled(
                correction_id=correction_id,
                enabled=enabled,
                tenant_id=tenant_id,
            )
        )

    def rollback_correction(self, correction_id: str, *, tenant_id: str) -> dict[str, Any]:
        return dict(self._backend.rollback_correction(correction_id, tenant_id=tenant_id) or {})

