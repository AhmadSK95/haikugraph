#!/usr/bin/env python3
"""Update v2 trackers strictly from generated artifacts."""

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


def _set_table_status(text: str, row_id: str, status: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"| {row_id} |"):
            parts = line.split("|")
            if len(parts) >= 5:
                # For gate table status is 4th visible column, for workstream table status is 4th.
                # Indices include leading/trailing empty segments.
                status_idx = 3
                parts[status_idx] = f" {status} "
                line = "|".join(parts)
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _set_gate_evidence(text: str, gate_id: str, evidence: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"| {gate_id} |"):
            parts = line.split("|")
            if len(parts) >= 6:
                parts[5] = f" `{evidence}` "
                line = "|".join(parts)
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync v2 trackers from artifact evidence")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--tracker", default="trackers/V2_MASTER_EXECUTION_TRACKER.md")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir).resolve()
    tracker_path = Path(args.tracker).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    truth_path = _latest(reports_dir, "v2_qa_truth_report_*.json")
    drill_path = _latest(reports_dir, "v2_cutover_drill_*.json")
    drift_path = _latest(reports_dir, "v2_quality_drift_alarm_*.json")
    load_path = _latest(reports_dir, "v2_load_stress_trend_*.json")
    ui_path = _latest(reports_dir, "v2_ui_quality_gate_*.json")
    weekly_path = _latest(reports_dir, "v2_weekly_quality_review_*.json")

    truth = _load(truth_path)
    drill = _load(drill_path)
    drift = _load(drift_path)
    load = _load(load_path)
    ui = _load(ui_path)
    weekly = _load(weekly_path)

    truth_ok = bool((truth.get("summary") or {}).get("release_gate_passed"))
    drill_ok = bool(drill.get("drill_passed"))
    drift_ok = str(drift.get("status") or "") == "pass"
    load_ok = bool(load.get("overall_pass"))
    ui_ok = bool(ui.get("overall_pass"))
    weekly_ok = bool(weekly.get("overall_ready"))

    if not tracker_path.exists():
        raise SystemExit(f"Tracker not found: {tracker_path}")
    text = tracker_path.read_text(encoding="utf-8")

    # Gate table.
    if truth_ok and truth_path is not None:
        text = _set_table_status(text, "G6", "DONE")
        text = _set_gate_evidence(text, "G6", str(truth_path.relative_to(tracker_path.parents[1])))
    if drill_ok and drill_path is not None:
        text = _set_table_status(text, "G7", "DONE")
        text = _set_gate_evidence(text, "G7", str(drill_path.relative_to(tracker_path.parents[1])))
    if drift_ok and truth_ok and drill_ok:
        text = _set_table_status(text, "G8", "DONE")
        evidence = str((weekly_path or drift_path).relative_to(tracker_path.parents[1])) if (weekly_path or drift_path) else ""
        if evidence:
            text = _set_gate_evidence(text, "G8", evidence)

    # In-progress task rows.
    if load_ok:
        text = _set_table_status(text, "D09", "DONE")
    if ui_ok:
        text = _set_table_status(text, "E06", "DONE")
        text = _set_table_status(text, "E07", "DONE")
        text = _set_table_status(text, "E08", "DONE")
    if truth_ok:
        text = _set_table_status(text, "G08", "DONE")
    if weekly_ok:
        text = _set_table_status(text, "H04", "DONE")
    if weekly_ok and truth_ok and drill_ok and drift_ok:
        text = _set_table_status(text, "H06", "DONE")
        text = _set_table_status(text, "H08", "DONE")
        text = _set_table_status(text, "F07", "DONE")
        text = _set_table_status(text, "F08", "DONE")

    tracker_path.write_text(text, encoding="utf-8")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tracker": str(tracker_path),
        "inputs": {
            "truth_report": str(truth_path or ""),
            "cutover_drill": str(drill_path or ""),
            "drift_report": str(drift_path or ""),
            "load_stress_report": str(load_path or ""),
            "ui_quality_report": str(ui_path or ""),
            "weekly_review_report": str(weekly_path or ""),
        },
        "checks": {
            "truth_ok": truth_ok,
            "drill_ok": drill_ok,
            "drift_ok": drift_ok,
            "load_ok": load_ok,
            "ui_ok": ui_ok,
            "weekly_ok": weekly_ok,
        },
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"v2_tracker_sync_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"tracker": str(tracker_path), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
