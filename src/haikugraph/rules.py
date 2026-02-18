"""Lightweight rules compatibility layer.

Some execution paths import `haikugraph.rules` for optional filter rules.
This module keeps those imports stable even when no external rules file exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_rules(path: str | Path | None = None) -> dict[str, Any]:
    """Load optional rules from YAML if present; otherwise return empty config."""
    target = Path(path) if path else Path("rules.yaml")
    if not target.exists():
        return {}

    try:
        import yaml  # type: ignore
    except Exception:
        return {}

    try:
        parsed = yaml.safe_load(target.read_text()) or {}
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def apply_entity_rules(
    table_name: str,
    existing_where_clauses: list[str],
    question: str = "",
    include_defaults: bool = True,
    include_global: bool = True,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return clauses unchanged when no compatible rules are configured.

    The execution engine expects this function to be non-fatal and additive.
    """
    _ = (table_name, question, include_defaults, include_global)
    return list(existing_where_clauses), []

