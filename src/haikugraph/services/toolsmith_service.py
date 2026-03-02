"""Toolsmith, glossary, and teaching service wrapper."""

from __future__ import annotations

from typing import Any


class ToolsmithService:
    def __init__(self, backend: Any):
        self._backend = backend

    def record_fix(self, payload: dict[str, Any], *, tenant_id: str) -> dict[str, Any]:
        return dict(self._backend.record_fix(**payload, tenant_id=tenant_id) or {})

    def list_tool_candidates(self, *, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._backend.list_tool_candidates(tenant_id=tenant_id, limit=limit) or [])

    def stage_tool_candidate(self, tool_id: str, *, tenant_id: str) -> dict[str, Any]:
        return dict(self._backend.stage_tool_candidate(tool_id, tenant_id=tenant_id) or {})

    def promote_tool_candidate(self, tool_id: str, *, tenant_id: str) -> dict[str, Any]:
        return dict(self._backend.promote_tool_candidate(tool_id, tenant_id=tenant_id) or {})

    def rollback_tool_candidate(self, tool_id: str, *, tenant_id: str) -> dict[str, Any]:
        return dict(self._backend.rollback_tool_candidate(tool_id, tenant_id=tenant_id) or {})

    def list_glossary(self, *, tenant_id: str) -> list[dict[str, Any]]:
        return list(self._backend.list_glossary(tenant_id=tenant_id) or [])

    def upsert_glossary_term(self, payload: dict[str, Any], *, tenant_id: str) -> dict[str, Any]:
        return dict(self._backend.upsert_glossary_term(**payload, tenant_id=tenant_id) or {})

    def list_teachings(self, *, tenant_id: str) -> list[dict[str, Any]]:
        return list(self._backend.list_teachings(tenant_id=tenant_id) or [])

    def add_teaching(self, payload: dict[str, Any], *, tenant_id: str) -> dict[str, Any]:
        return dict(self._backend.add_teaching(**payload, tenant_id=tenant_id) or {})

