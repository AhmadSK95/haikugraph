"""Shared runtime selection types for API and v2 orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class RuntimeSelection:
    requested_mode: str
    mode: str
    use_llm: bool
    provider: str | None
    reason: str
    intent_model: str | None = None
    narrator_model: str | None = None


@dataclass(frozen=True)
class AutonomyConfig:
    mode: str = "bounded"
    auto_correction: bool = True
    strict_truth: bool = True
    max_refinement_rounds: int = 2
    max_candidate_plans: int = 5


def load_dotenv_file(path: str = ".env") -> None:
    """Minimal dotenv loader to keep server boot independent from legacy runtime."""
    target = Path(path)
    if not target.exists():
        return
    try:
        for raw in target.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            os.environ.setdefault(key, value.strip())
    except Exception:
        return

