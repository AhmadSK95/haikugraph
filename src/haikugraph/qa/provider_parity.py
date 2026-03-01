"""Provider/mode parity report helpers with drift alerting."""

from __future__ import annotations

import json
from typing import Any


def _contract_signature(contract: dict[str, Any]) -> str:
    if not isinstance(contract, dict):
        return ""
    payload = {
        "metric": str(contract.get("metric") or ""),
        "table": str(contract.get("table") or ""),
        "dimensions": list(contract.get("dimensions") or []),
        "time_scope": str(contract.get("time_scope") or ""),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def build_provider_parity_report(
    rows: list[dict[str, Any]],
    *,
    success_delta_threshold: float = 0.05,
    contract_drift_threshold: float = 0.10,
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(
            {
                "goal": str(row.get("goal") or "").strip(),
                "mode": str(row.get("mode") or "").strip() or "unknown",
                "success": bool(row.get("success")),
                "contract_signature": _contract_signature(dict(row.get("contract_spec") or {})),
            }
        )

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in normalized:
        by_mode.setdefault(row["mode"], []).append(row)

    mode_stats: dict[str, dict[str, Any]] = {}
    for mode, mode_rows in by_mode.items():
        runs = len(mode_rows)
        success_runs = sum(1 for r in mode_rows if r["success"])
        mode_stats[mode] = {
            "runs": runs,
            "success_rate": round(success_runs / max(1, runs), 4),
        }

    goal_mode_contracts: dict[str, dict[str, set[str]]] = {}
    for row in normalized:
        goal = row["goal"]
        mode = row["mode"]
        sig = row["contract_signature"]
        if not goal or not sig:
            continue
        goal_mode_contracts.setdefault(goal, {}).setdefault(mode, set()).add(sig)

    drift_cases = 0
    compared_cases = 0
    drift_examples: list[dict[str, Any]] = []
    for goal, mode_map in goal_mode_contracts.items():
        if len(mode_map) < 2:
            continue
        compared_cases += 1
        union_count = len(set().union(*mode_map.values()))
        drift = union_count > 1
        if drift:
            drift_cases += 1
            if len(drift_examples) < 5:
                drift_examples.append(
                    {
                        "goal": goal,
                        "modes": {mode: sorted(list(sigs))[:2] for mode, sigs in mode_map.items()},
                    }
                )

    drift_rate = (drift_cases / max(1, compared_cases)) if compared_cases else 0.0
    baseline = mode_stats.get("deterministic", {"success_rate": 1.0})
    baseline_success = float(baseline.get("success_rate") or 1.0)

    alerts: list[dict[str, Any]] = []
    for mode, stats in mode_stats.items():
        if mode == "deterministic":
            continue
        success_rate = float(stats.get("success_rate") or 0.0)
        delta = baseline_success - success_rate
        if delta > success_delta_threshold:
            alerts.append(
                {
                    "type": "success_drift",
                    "mode": mode,
                    "baseline": round(baseline_success, 4),
                    "actual": round(success_rate, 4),
                    "delta": round(delta, 4),
                }
            )

    if drift_rate > contract_drift_threshold:
        alerts.append(
            {
                "type": "contract_drift",
                "drift_rate": round(drift_rate, 4),
                "threshold": round(float(contract_drift_threshold), 4),
                "compared_cases": compared_cases,
            }
        )

    status = "ok" if not alerts else "alert"
    return {
        "status": status,
        "summary": {
            "rows": len(normalized),
            "modes": sorted(mode_stats.keys()),
            "mode_stats": mode_stats,
            "compared_cases": compared_cases,
            "contract_drift_cases": drift_cases,
            "contract_drift_rate": round(drift_rate, 4),
        },
        "alerts": alerts,
        "drift_examples": drift_examples,
    }
