#!/usr/bin/env python3
"""Quality drift alarm for v2 truth reports."""

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


def _latest_truth_reports(reports_dir: Path) -> list[Path]:
    matches = list(reports_dir.glob("v2_qa_truth_report_*.json"))
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object payload in {path}")
    return payload


def _suite_scores(payload: dict[str, Any]) -> dict[str, float]:
    suites = dict(payload.get("suites") or {})
    out: dict[str, float] = {}
    for suite_id, row in suites.items():
        if not isinstance(row, dict):
            continue
        try:
            out[str(suite_id)] = float(row.get("score_pct") or 0.0)
        except Exception:
            out[str(suite_id)] = 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v2 quality drift alarm")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--baseline-report", default="")
    parser.add_argument("--candidate-report", default="")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--max-composite-drop", type=float, default=2.0)
    parser.add_argument("--max-suite-drop", type=float, default=3.0)
    parser.add_argument("--strict-floor", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_arg = str(args.baseline_report).strip()
    candidate_arg = str(args.candidate_report).strip()
    if baseline_arg and candidate_arg:
        baseline = Path(baseline_arg).resolve()
        candidate = Path(candidate_arg).resolve()
    else:
        truth_reports = _latest_truth_reports(reports_dir)
        if not truth_reports:
            raise SystemExit("Missing truth reports in reports directory.")
        if candidate_arg:
            candidate = Path(candidate_arg).resolve()
        else:
            candidate = truth_reports[0]
        if baseline_arg:
            baseline = Path(baseline_arg).resolve()
        else:
            baseline = next((p for p in truth_reports if p.resolve() != candidate.resolve()), None)
            if baseline is None:
                raise SystemExit("Need at least two distinct truth reports (or pass --baseline-report).")

    if not baseline.exists():
        raise SystemExit(f"Missing baseline report: {baseline}")
    if not candidate.exists():
        raise SystemExit(f"Missing candidate report: {candidate}")

    baseline_payload = _load_json(baseline)
    candidate_payload = _load_json(candidate)
    baseline_summary = dict(baseline_payload.get("summary") or {})
    candidate_summary = dict(candidate_payload.get("summary") or {})
    baseline_suites = _suite_scores(baseline_payload)
    candidate_suites = _suite_scores(candidate_payload)

    baseline_composite = float(baseline_summary.get("composite_truth_score") or 0.0)
    candidate_composite = float(candidate_summary.get("composite_truth_score") or 0.0)
    composite_drop = round(baseline_composite - candidate_composite, 4)

    alerts: list[dict[str, Any]] = []
    if composite_drop > float(args.max_composite_drop):
        alerts.append(
            {
                "kind": "composite_drop",
                "baseline": baseline_composite,
                "candidate": candidate_composite,
                "drop": composite_drop,
                "threshold": float(args.max_composite_drop),
            }
        )

    for suite_id, base_score in sorted(baseline_suites.items()):
        cand_score = float(candidate_suites.get(suite_id, 0.0))
        drop = round(base_score - cand_score, 4)
        if drop > float(args.max_suite_drop):
            alerts.append(
                {
                    "kind": "suite_drop",
                    "suite_id": suite_id,
                    "baseline": base_score,
                    "candidate": cand_score,
                    "drop": drop,
                    "threshold": float(args.max_suite_drop),
                }
            )

    floor_violations = list(candidate_summary.get("floor_violations") or [])
    if bool(args.strict_floor) and floor_violations:
        alerts.append(
            {
                "kind": "floor_violation",
                "detail": floor_violations,
            }
        )

    status = "pass" if not alerts else "fail"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "baseline_report": str(baseline),
        "candidate_report": str(candidate),
        "baseline_composite": baseline_composite,
        "candidate_composite": candidate_composite,
        "composite_drop": composite_drop,
        "thresholds": {
            "max_composite_drop": float(args.max_composite_drop),
            "max_suite_drop": float(args.max_suite_drop),
            "strict_floor": bool(args.strict_floor),
        },
        "alerts": alerts,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"v2_quality_drift_alarm_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"drift_report": str(out_path), "status": status, "alert_count": len(alerts)}, indent=2))
    return 0 if not alerts else 2


if __name__ == "__main__":
    raise SystemExit(main())
