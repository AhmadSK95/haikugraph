#!/usr/bin/env python3
"""Static + API-backed UI reliability gate for the current shell."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from haikugraph.api.server import create_app


def _contains_all(text: str, required: list[str]) -> list[str]:
    missing: list[str] = []
    for token in required:
        if token not in text:
            missing.append(token)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Run UI quality gate checks")
    parser.add_argument("--db-path", default="data/haikugraph.db")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    js_path = repo / "src/haikugraph/api/ui.js"
    css_path = repo / "src/haikugraph/api/ui.css"
    html_path = repo / "src/haikugraph/api/ui.html"

    js = js_path.read_text(encoding="utf-8")
    css = css_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")

    checks: list[dict[str, Any]] = []

    # E06 explainability modal timeline + contract checks
    e06_missing = _contains_all(
        js,
        [
            "Agent decision timeline",
            "Advanced diagnostics (JSON)",
            "contract_validation",
            "stage_timings_ms",
        ],
    )
    checks.append(
        {
            "id": "E06",
            "name": "Explainability modal timeline + contract checks",
            "passed": not e06_missing,
            "missing": e06_missing,
        }
    )

    # E07 rules UX validation + previews
    e07_missing = _contains_all(
        js,
        [
            "Rule name and at least one trigger are required.",
            "Invalid JSON in action payload.",
            "action payload",
            "startRuleEditByIndex",
        ],
    )
    checks.append(
        {
            "id": "E07",
            "name": "Rule management validation + previews",
            "passed": not e07_missing,
            "missing": e07_missing,
        }
    )

    # E08 accessibility/responsive/perf static checks
    e08_missing = []
    e08_missing.extend(_contains_all(html, ['<meta name="viewport"']))
    e08_missing.extend(_contains_all(css, ["@media(max-width:640px)"]))
    if js_path.stat().st_size > 120_000:
        e08_missing.append("ui.js exceeds 120KB budget")
    if css_path.stat().st_size > 40_000:
        e08_missing.append("ui.css exceeds 40KB budget")
    checks.append(
        {
            "id": "E08",
            "name": "Accessibility/responsive/performance shell checks",
            "passed": not e08_missing,
            "missing": e08_missing,
        }
    )

    # API-backed smoke for served modular assets
    app = create_app(db_path=Path(args.db_path).resolve())
    with TestClient(app) as client:
        page = client.get("/")
        css_resp = client.get("/ui/assets/ui.css")
        js_resp = client.get("/ui/assets/ui.js")
    smoke_ok = (
        page.status_code == 200
        and "/ui/assets/ui.css" in page.text
        and "/ui/assets/ui.js" in page.text
        and css_resp.status_code == 200
        and js_resp.status_code == 200
    )
    checks.append(
        {
            "id": "UI_SMOKE",
            "name": "Modular UI assets served",
            "passed": bool(smoke_ok),
            "missing": [] if smoke_ok else ["UI asset route or shell markup failed"],
        }
    )

    failed = [c for c in checks if not bool(c.get("passed"))]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(Path(args.db_path).resolve()),
        "checks": checks,
        "pass_count": len(checks) - len(failed),
        "total_checks": len(checks),
        "overall_pass": not failed,
    }

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"v2_ui_quality_gate_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out_path), "overall_pass": report["overall_pass"]}, indent=2))
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
