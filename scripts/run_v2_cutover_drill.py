#!/usr/bin/env python3
"""Run local canary + shadow + rollback drill for runtime cutover evidence.

Uses in-process FastAPI TestClient to avoid socket/port constraints while still
executing full API behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from haikugraph.api.server import create_app


def _run_mode_probe(client: TestClient, runtime_mode: str) -> dict[str, Any]:
    prompts = [
        "How many transactions are there?",
        "Show MT103 transaction count by state for November 2025.",
    ]
    session_id = f"cutover-{runtime_mode}"
    headers = {"x-datada-role": "admin", "x-datada-tenant-id": "public"}
    rows: list[dict[str, Any]] = []

    for idx, goal in enumerate(prompts):
        payload = {
            "goal": goal,
            "llm_mode": "deterministic",
            "session_id": session_id,
            "tenant_id": "public",
            "role": "admin",
        }
        started = time.perf_counter()
        resp = client.post("/api/assistant/query", json=payload, headers=headers)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        body: dict[str, Any] = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        rows.append(
            {
                "idx": idx,
                "goal": goal,
                "status_code": int(resp.status_code),
                "success": bool(body.get("success")),
                "execution_time_ms": float(body.get("execution_time_ms") or elapsed_ms),
                "analysis_version": str(body.get("analysis_version") or ""),
                "provider_effective": str(body.get("provider_effective") or ""),
                "shadow_diff_present": bool(((body.get("runtime") or {}).get("shadow_diff"))),
                "slice_signature_present": bool(str(body.get("slice_signature") or "").strip()),
            }
        )

    followup_payload = {
        "goal": "Now keep same slice and add total amount too.",
        "llm_mode": "deterministic",
        "session_id": session_id,
        "tenant_id": "public",
        "role": "admin",
    }
    started = time.perf_counter()
    resp = client.post("/api/assistant/query", json=followup_payload, headers=headers)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    rows.append(
        {
            "idx": len(rows),
            "goal": followup_payload["goal"],
            "status_code": int(resp.status_code),
            "success": bool(body.get("success")),
            "execution_time_ms": float(body.get("execution_time_ms") or elapsed_ms),
            "analysis_version": str(body.get("analysis_version") or ""),
            "provider_effective": str(body.get("provider_effective") or ""),
            "shadow_diff_present": bool(((body.get("runtime") or {}).get("shadow_diff"))),
            "slice_signature_present": bool(str(body.get("slice_signature") or "").strip()),
        }
    )

    ok_rows = [r for r in rows if int(r["status_code"]) == 200 and bool(r["success"])]
    pass_rate = round((100.0 * len(ok_rows) / max(1, len(rows))), 2)
    return {
        "runtime_mode": runtime_mode,
        "rows": rows,
        "pass_rate_pct": pass_rate,
    }


def _run_stage(stage_name: str, runtime_mode: str, db_path: Path) -> dict[str, Any]:
    os.environ["HG_RUNTIME_VERSION"] = runtime_mode
    os.environ["HG_DB_PATH"] = str(db_path.resolve())
    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        probe = _run_mode_probe(client, runtime_mode)
    probe["stage"] = stage_name
    return probe


def main() -> int:
    parser = argparse.ArgumentParser(description="Run runtime cutover drill")
    parser.add_argument("--db-path", default="data/haikugraph.db")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    db_path = Path(args.db_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use a private snapshot to avoid DuckDB file lock conflicts with live servers.
    snapshot_dir = Path(tempfile.mkdtemp(prefix="v2-cutover-drill-"))
    snapshot_db = snapshot_dir / "cutover_snapshot.duckdb"
    shutil.copy2(db_path, snapshot_db)

    stages = [
        ("canary_v2", "v2"),
        ("shadow_validation", "shadow"),
        ("rollback_v1", "v1"),
    ]
    stage_rows = [_run_stage(stage_name, runtime_mode, snapshot_db) for stage_name, runtime_mode in stages]

    stage_map = {row["stage"]: row for row in stage_rows}
    v2_ok = float(stage_map["canary_v2"]["pass_rate_pct"]) >= 100.0
    shadow_ok = (
        float(stage_map["shadow_validation"]["pass_rate_pct"]) >= 100.0
        and any(bool(r.get("shadow_diff_present")) for r in list(stage_map["shadow_validation"]["rows"] or []))
    )
    rollback_ok = float(stage_map["rollback_v1"]["pass_rate_pct"]) >= 100.0
    drill_passed = bool(v2_ok and shadow_ok and rollback_ok)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "snapshot_db_path": str(snapshot_db),
        "criteria": {
            "v2_canary_pass_rate_pct": 100.0,
            "shadow_requires_diff_signal": True,
            "rollback_v1_pass_rate_pct": 100.0,
        },
        "checks": {
            "v2_canary_ok": v2_ok,
            "shadow_validation_ok": shadow_ok,
            "rollback_v1_ok": rollback_ok,
        },
        "drill_passed": drill_passed,
        "stages": stage_rows,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"v2_cutover_drill_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        snapshot_db.unlink(missing_ok=True)
        snapshot_dir.rmdir()
    except Exception:
        pass
    print(json.dumps({"cutover_drill_report": str(out_path), "drill_passed": drill_passed}, indent=2))
    return 0 if drill_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
