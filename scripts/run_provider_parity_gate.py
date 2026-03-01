#!/usr/bin/env python3
"""Evaluate provider parity drift from benchmark/source-truth style artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from haikugraph.qa.provider_parity import build_provider_parity_report


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    if isinstance(payload.get("rows"), list):
        return [item for item in payload.get("rows") if isinstance(item, dict)]

    runs = payload.get("runs")
    if isinstance(runs, list):
        out: list[dict[str, Any]] = []
        for item in runs:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "goal": str(item.get("question") or item.get("goal") or ""),
                    "mode": str(item.get("mode_actual") or item.get("mode_requested") or item.get("llm_mode") or ""),
                    "success": bool(item.get("success")),
                    "contract_spec": dict(item.get("contract_spec") or {}),
                }
            )
        return out

    return []


def _to_markdown(report: dict[str, Any], source: Path) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Provider Parity Gate Report",
        "",
        f"- Generated at: `{datetime.utcnow().isoformat(timespec='seconds')}Z`",
        f"- Source artifact: `{source}`",
        f"- Status: `{report.get('status')}`",
        f"- Rows analyzed: `{summary.get('rows', 0)}`",
        f"- Compared cases: `{summary.get('compared_cases', 0)}`",
        f"- Contract drift rate: `{summary.get('contract_drift_rate', 0.0)}`",
        "",
        "## Mode Stats",
    ]
    mode_stats = summary.get("mode_stats") or {}
    for mode in sorted(mode_stats.keys()):
        item = mode_stats.get(mode) or {}
        lines.append(
            f"- `{mode}`: runs={item.get('runs', 0)}, success_rate={item.get('success_rate', 0.0)}"
        )

    alerts = report.get("alerts") or []
    lines.append("")
    lines.append("## Alerts")
    if alerts:
        for alert in alerts:
            lines.append(f"- `{alert.get('type')}`: {json.dumps(alert, sort_keys=True)}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run provider parity gate on benchmark JSON")
    parser.add_argument("--input-json", required=True, help="Path to benchmark/source-truth JSON artifact")
    parser.add_argument("--out-json", default="", help="Output JSON report path")
    parser.add_argument("--out-md", default="", help="Output markdown report path")
    parser.add_argument("--success-delta-threshold", type=float, default=0.05)
    parser.add_argument("--contract-drift-threshold", type=float, default=0.10)
    args = parser.parse_args()

    src = Path(args.input_json).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Input artifact not found: {src}")

    rows = _load_rows(src)
    report = build_provider_parity_report(
        rows,
        success_delta_threshold=float(args.success_delta_threshold),
        contract_drift_threshold=float(args.contract_drift_threshold),
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.out_json).expanduser() if args.out_json else (Path("reports") / f"provider_parity_gate_{timestamp}.json")
    out_md = Path(args.out_md).expanduser() if args.out_md else out_json.with_suffix(".md")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(report, src), encoding="utf-8")

    print(f"saved_json={out_json}")
    print(f"saved_md={out_md}")
    print(f"status={report.get('status')}")
    print(f"alerts={len(report.get('alerts') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
