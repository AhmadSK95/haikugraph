"""Structured stage-event bus for deterministic v2 orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Literal

from haikugraph.v2.types import StageEventV2


STAGE_ORDER: tuple[str, ...] = (
    "semantic_profiler",
    "intent_engine",
    "planner",
    "query_compiler",
    "executor",
    "evaluator",
    "insight_engine",
)


class StageTransitionError(RuntimeError):
    """Raised when stage transitions violate the deterministic state machine."""


@dataclass
class StageEventBusV2:
    """Collects stage lifecycle events with strict stage ordering."""

    stage_order: tuple[str, ...] = STAGE_ORDER
    _events: list[StageEventV2] = field(default_factory=list, init=False)
    _active_stage: str | None = field(default=None, init=False)
    _completed_count: int = field(default=0, init=False)
    _started_at: float = field(default_factory=time.perf_counter, init=False)
    _sequence: int = field(default=0, init=False)

    @property
    def state(self) -> str:
        if self._active_stage:
            return f"{self._active_stage}:running"
        if self._completed_count >= len(self.stage_order):
            return "completed"
        if self._completed_count == 0:
            return "initialized"
        return f"ready:{self.stage_order[self._completed_count]}"

    def _elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._started_at) * 1000, 2)

    def _expected_next_stage(self) -> str | None:
        if self._completed_count >= len(self.stage_order):
            return None
        return self.stage_order[self._completed_count]

    def _emit(
        self,
        *,
        stage: str,
        status: Literal["started", "completed", "failed", "skipped"],
        detail: dict[str, Any] | None = None,
    ) -> None:
        self._sequence += 1
        self._events.append(
            StageEventV2(
                sequence=self._sequence,
                stage=stage,
                status=status,
                elapsed_ms=self._elapsed_ms(),
                detail=dict(detail or {}),
            )
        )

    def start_stage(self, stage: str) -> None:
        expected = self._expected_next_stage()
        if expected is None:
            raise StageTransitionError("All stages are already completed.")
        if stage != expected:
            raise StageTransitionError(f"Invalid stage start order: expected '{expected}', got '{stage}'.")
        if self._active_stage is not None:
            raise StageTransitionError(
                f"Cannot start stage '{stage}' while '{self._active_stage}' is still active."
            )
        self._active_stage = stage
        self._emit(stage=stage, status="started")

    def complete_stage(
        self,
        stage: str,
        *,
        status: Literal["completed", "failed", "skipped"] = "completed",
        detail: dict[str, Any] | None = None,
    ) -> None:
        if self._active_stage != stage:
            raise StageTransitionError(
                f"Cannot complete stage '{stage}' while active stage is '{self._active_stage}'."
            )
        self._emit(stage=stage, status=status, detail=detail)
        self._active_stage = None
        if status == "completed":
            self._completed_count += 1

    def fail_stage(self, stage: str, *, error: str, detail: dict[str, Any] | None = None) -> None:
        payload = dict(detail or {})
        payload["error"] = error
        self.complete_stage(stage, status="failed", detail=payload)

    def events(self) -> list[StageEventV2]:
        return list(self._events)
