"""Fast per-round test target selection."""

from __future__ import annotations

from typing import Iterable


DEFAULT_FAST_SUITES: list[str] = [
    "tests/test_api_runtime_features.py",
    "tests/test_brd_explainability.py",
    "tests/test_api_autonomy_features.py",
    "tests/test_slice_completion.py",
    "tests/test_advanced_analytics_packs.py",
]

_PATH_RULES: list[tuple[str, list[str]]] = [
    (
        "src/haikugraph/poc/",
        [
            "tests/test_api_runtime_features.py",
            "tests/test_brd_explainability.py",
            "tests/test_api_autonomy_features.py",
            "tests/test_slice_completion.py",
        ],
    ),
    (
        "src/haikugraph/analytics/",
        [
            "tests/test_advanced_analytics_packs.py",
            "tests/test_api_runtime_features.py",
        ],
    ),
    (
        "src/haikugraph/api/server.py",
        [
            "tests/test_api_runtime_features.py",
        ],
    ),
    (
        "src/haikugraph/llm/",
        [
            "tests/test_llm_router_latency.py",
            "tests/test_api_runtime_features.py",
        ],
    ),
    (
        "scripts/",
        [
            "tests/test_round11_fresh_fast.py",
            "tests/test_round_capability_prompt_generator.py",
        ],
    ),
]


def select_fast_suites(changed_paths: Iterable[str] | None = None) -> list[str]:
    """Select fast suites for a round based on changed file paths."""
    if not changed_paths:
        return list(DEFAULT_FAST_SUITES)
    selected: list[str] = []
    seen = set()
    for path in changed_paths:
        clean = str(path or "").strip()
        if not clean:
            continue
        for prefix, suites in _PATH_RULES:
            if clean.startswith(prefix):
                for suite in suites:
                    if suite in seen:
                        continue
                    seen.add(suite)
                    selected.append(suite)
    if not selected:
        return list(DEFAULT_FAST_SUITES)
    return selected

