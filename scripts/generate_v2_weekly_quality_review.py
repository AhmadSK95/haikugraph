#!/usr/bin/env python3
"""Generate weekly quality review artifact from latest evidence files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _latest(reports_dir: Path, pattern: str) -> Path | None:
    matches = list(reports_dir.glob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _load(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# v2 Weekly Quality Review",
        "",
        f"- Generated at: `{report.get('generated_at')}`",
        "",
        "## Gate Evidence Snapshot",
    ]
    for row in report.get("gate_snapshot", []):
        lines.append(f"- {row['gate']}: `{row['status']}` ({row['evidence']})")
    lines.extend(
        [
            "",
            "## Blockers",
        ]
    )
    blockers = report.get("blockers") or []
    if blockers:
        for b in blockers:
            lines.append(f"- {b}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Action Summary",
            f"- Truth score: `{report.get('truth_score')}`",
            f"- Release gate passed: `{report.get('release_gate_passed')}`",
            f"- Cutover drill passed: `{report.get('cutover_drill_passed')}`",
            f"- Drift status: `{report.get('drift_status')}`",
            f"- Load/stress pass: `{report.get('load_stress_passed')}`",
            f"- UI quality gate pass: `{report.get('ui_quality_passed')}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate weekly v2 quality review report")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    truth_path = _latest(reports_dir, "v2_qa_truth_report_*.json")
    drill_path = _latest(reports_dir, "v2_cutover_drill_*.json")
    drift_path = _latest(reports_dir, "v2_quality_drift_alarm_*.json")
    load_path = _latest(reports_dir, "v2_load_stress_trend_*.json")
    ui_path = _latest(reports_dir, "v2_ui_quality_gate_*.json")

    truth = _load(truth_path)
    drill = _load(drill_path)
    drift = _load(drift_path)
    load = _load(load_path)
    ui = _load(ui_path)

    summary = dict(truth.get("summary") or {})
    release_gate_passed = bool(summary.get("release_gate_passed"))
    truth_score = float(summary.get("composite_truth_score") or 0.0)
    floor_violations = list(summary.get("floor_violations") or [])
    cutover_passed = bool(drill.get("drill_passed"))
    drift_status = str(drift.get("status") or "unknown")
    load_passed = bool(load.get("overall_pass"))
    ui_passed = bool(ui.get("overall_pass"))

    blockers: list[str] = []
    if not release_gate_passed:
        blockers.append("Release gate is not passed.")
    if floor_violations:
        blockers.append(f"Floor violations detected: {', '.join(map(str, floor_violations))}")
    if not cutover_passed:
        blockers.append("Cutover drill is not passed.")
    if drift_status != "pass":
        blockers.append("Quality drift alarm is failing.")
    if not load_passed:
        blockers.append("Load/stress trend report indicates SLO misses.")
    if not ui_passed:
        blockers.append("UI quality gate has unresolved failures.")

    gate_snapshot = [
        {"gate": "G6", "status": "PASS" if release_gate_passed and not floor_violations else "FAIL", "evidence": str(truth_path or "")},
        {"gate": "G7", "status": "PASS" if cutover_passed else "FAIL", "evidence": str(drill_path or "")},
        {"gate": "G8", "status": "PASS" if drift_status == "pass" else "FAIL", "evidence": str(drift_path or "")},
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "truth_report": str(truth_path or ""),
        "cutover_drill_report": str(drill_path or ""),
        "drift_report": str(drift_path or ""),
        "load_stress_report": str(load_path or ""),
        "ui_quality_report": str(ui_path or ""),
        "truth_score": round(truth_score, 2),
        "release_gate_passed": release_gate_passed,
        "cutover_drill_passed": cutover_passed,
        "drift_status": drift_status,
        "load_stress_passed": load_passed,
        "ui_quality_passed": ui_passed,
        "floor_violations": floor_violations,
        "blockers": blockers,
        "gate_snapshot": gate_snapshot,
        "overall_ready": not blockers,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"v2_weekly_quality_review_{stamp}.json"
    md_path = out_dir / f"v2_weekly_quality_review_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path), "overall_ready": report["overall_ready"]}, indent=2))
    return 0 if report["overall_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
