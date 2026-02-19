"""Compatibility shim for legacy orchestrator imports.

This module intentionally delegates all execution to the unified
`AgenticAnalyticsTeam` runtime so the project has a single architecture path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from haikugraph.agents.contracts import AssistantQueryResponse
from haikugraph.poc.agentic_team import (
    AgenticAnalyticsTeam,
    AutonomyConfig,
    RuntimeSelection,
)


@dataclass
class OrchestratorConfig:
    """Legacy config surface mapped to unified agentic runtime."""

    use_llm: bool = True
    llm_provider: str | None = None
    llm_mode: str = "auto"
    storyteller_mode: bool = False
    tenant_id: str = "public"
    auto_correction: bool = True
    strict_truth: bool = True
    max_refinement_loops: int = 2
    max_candidate_plans: int = 5


class AnalystOrchestrator:
    """Legacy orchestrator API backed by AgenticAnalyticsTeam.

    This keeps backward import compatibility while enforcing one architecture.
    """

    def __init__(
        self,
        db_path: Path | str,
        config: OrchestratorConfig | None = None,
    ):
        self.db_path = Path(db_path)
        self.config = config or OrchestratorConfig()
        self.team = AgenticAnalyticsTeam(self.db_path)

    def _runtime_selection(self) -> RuntimeSelection:
        mode = (self.config.llm_mode or "auto").strip().lower()
        provider = (self.config.llm_provider or "").strip().lower() or None

        if not self.config.use_llm or mode == "deterministic":
            return RuntimeSelection(
                requested_mode=mode or "deterministic",
                mode="deterministic",
                use_llm=False,
                provider=None,
                reason="legacy orchestrator requested deterministic mode",
                intent_model=None,
                narrator_model=None,
            )

        if mode in {"openai", "local"}:
            if mode == "openai":
                provider = "openai"
            elif mode == "local":
                provider = "ollama"
            return RuntimeSelection(
                requested_mode=mode,
                mode=mode,
                use_llm=True,
                provider=provider,
                reason="legacy orchestrator delegated runtime mode",
                intent_model=None,
                narrator_model=None,
            )

        # Default legacy behavior: auto mode, provider left to runtime resolution.
        return RuntimeSelection(
            requested_mode="auto",
            mode="auto",
            use_llm=True,
            provider=provider,
            reason="legacy orchestrator delegated auto mode",
            intent_model=None,
            narrator_model=None,
        )

    def run(
        self,
        goal: str,
        constraints: dict[str, Any] | None = None,
    ) -> AssistantQueryResponse:
        constraints = constraints or {}
        runtime = self._runtime_selection()
        autonomy = AutonomyConfig(
            mode="bounded",
            auto_correction=bool(self.config.auto_correction),
            strict_truth=bool(self.config.strict_truth),
            max_refinement_rounds=max(0, int(self.config.max_refinement_loops)),
            max_candidate_plans=max(1, int(self.config.max_candidate_plans)),
        )
        context = constraints.get("conversation_context")
        conversation_context = context if isinstance(context, list) else []
        return self.team.run(
            goal,
            runtime,
            tenant_id=self.config.tenant_id,
            conversation_context=conversation_context,
            storyteller_mode=bool(self.config.storyteller_mode),
            autonomy=autonomy,
        )

    def close(self) -> None:
        self.team.close()

    def __enter__(self) -> "AnalystOrchestrator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
