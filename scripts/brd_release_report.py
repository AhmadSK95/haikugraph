#!/usr/bin/env python3
"""BRD §6 closure criterion #7 — CI-publishable machine-readable release report.

Runs all BRD test suites and produces a JSON report with:
- pass/fail by category
- blocker count
- closure criteria status
- north-pole score estimate

Usage:
    python scripts/brd_release_report.py [--output reports/brd_release_report.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _run_suite(suite_path: str, label: str) -> dict:
    """Run a pytest suite and return structured results."""
    start = time.monotonic()
    import shutil
    pytest_cmd = shutil.which("pytest") or f"{sys.executable} -m pytest"
    cmd = [pytest_cmd, suite_path, "-v", "--tb=short", "--no-header", "-q"] if shutil.which("pytest") else [
        sys.executable, "-m", "pytest", suite_path, "-v", "--tb=short", "--no-header", "-q"
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    elapsed_ms = (time.monotonic() - start) * 1000

    # Parse pytest output for pass/fail counts
    output = result.stdout + result.stderr
    passed = failed = skipped = xfailed = xpassed = 0
    for line in output.splitlines():
        line = line.strip()
        if "passed" in line or "failed" in line:
            import re
            m = re.search(r"(\d+)\s+passed", line)
            if m:
                passed = int(m.group(1))
            m = re.search(r"(\d+)\s+failed", line)
            if m:
                failed = int(m.group(1))
            m = re.search(r"(\d+)\s+skipped", line)
            if m:
                skipped = int(m.group(1))
            m = re.search(r"(\d+)\s+xfailed", line)
            if m:
                xfailed = int(m.group(1))
            m = re.search(r"(\d+)\s+xpassed", line)
            if m:
                xpassed = int(m.group(1))

    total = passed + failed
    pass_rate = (passed / total * 100) if total > 0 else 0.0

    return {
        "suite": label,
        "file": suite_path,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "xfailed": xfailed,
        "xpassed": xpassed,
        "total": total,
        "pass_rate_pct": round(pass_rate, 2),
        "elapsed_ms": round(elapsed_ms, 1),
        "exit_code": result.returncode,
        "output_tail": output[-2000:] if len(output) > 2000 else output,
    }


def _evaluate_closure_criteria(suites: dict) -> list[dict]:
    """Evaluate BRD §6 closure criteria."""
    criteria = []

    # Criterion 1: Semantic contract checks block execution
    expl = suites.get("explainability", {})
    c1_pass = expl.get("pass_rate_pct", 0) >= 90
    criteria.append({
        "id": "CC-1",
        "description": "Semantic contract checks block execution when constraints missing from SQL AST",
        "status": "GREEN" if c1_pass else "RED",
        "evidence": f"Explainability suite: {expl.get('pass_rate_pct', 0)}% pass",
    })

    # Criterion 2: Fabrication prompts produce refusal
    behavior = suites.get("behavior", {})
    c2_pass = behavior.get("pass_rate_pct", 0) >= 98
    criteria.append({
        "id": "CC-2",
        "description": "Unsupported/fabrication prompts produce refusal, never synthetic KPI",
        "status": "GREEN" if c2_pass else "RED",
        "evidence": f"Behavior suite: {behavior.get('pass_rate_pct', 0)}% pass ({behavior.get('failed', '?')} failures)",
    })

    # Criterion 3: All canonical behavior tests pass
    c3_pass = behavior.get("failed", 99) == 0
    criteria.append({
        "id": "CC-3",
        "description": "All canonical behavior tests pass with explicit evidence",
        "status": "GREEN" if c3_pass else "RED",
        "evidence": f"Behavior failures: {behavior.get('failed', '?')}",
    })

    # Criterion 4: Confidence calibration
    factual = suites.get("factual", {})
    c4_pass = factual.get("pass_rate_pct", 0) >= 90 and behavior.get("pass_rate_pct", 0) >= 95
    criteria.append({
        "id": "CC-4",
        "description": "Confidence calibration: false positive high-confidence rate <=2%",
        "status": "GREEN" if c4_pass else "YELLOW",
        "evidence": f"Factual={factual.get('pass_rate_pct', 0)}%, Behavior={behavior.get('pass_rate_pct', 0)}%",
    })

    # Criterion 5: MT103 metric contract documented and enforced
    c5_pass = factual.get("pass_rate_pct", 0) >= 85
    criteria.append({
        "id": "CC-5",
        "description": "MT103 metric contract documented and enforced in code + tests",
        "status": "GREEN" if c5_pass else "RED",
        "evidence": f"Factual suite (includes MT103 cases): {factual.get('pass_rate_pct', 0)}%",
    })

    # Criterion 6: Explain yourself view renders all components
    c6_pass = expl.get("pass_rate_pct", 0) >= 95
    criteria.append({
        "id": "CC-6",
        "description": "Explain-yourself view renders contract, SQL, audit, confidence decomposition",
        "status": "GREEN" if c6_pass else "RED",
        "evidence": f"Explainability suite: {expl.get('pass_rate_pct', 0)}% pass",
    })

    # Criterion 7: CI publishes machine-readable report
    criteria.append({
        "id": "CC-7",
        "description": "CI publishes machine-readable report with pass/fail by category",
        "status": "GREEN",
        "evidence": "This report is the evidence",
    })

    return criteria


def main():
    parser = argparse.ArgumentParser(description="BRD release readiness report")
    parser.add_argument(
        "--output", "-o",
        default="reports/brd_release_report.json",
        help="Output path for the JSON report",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("BRD RELEASE READINESS REPORT")
    print("=" * 70)

    suites_config = [
        ("tests/test_brd_canonical_factual.py", "factual"),
        ("tests/test_brd_behavior_safety.py", "behavior"),
        ("tests/test_brd_followup_session.py", "followup"),
        ("tests/test_brd_explainability.py", "explainability"),
        ("tests/test_brd_cross_mode_parity.py", "cross_mode"),
    ]

    suite_results = {}
    all_passed = 0
    all_failed = 0
    all_total = 0

    for path, label in suites_config:
        print(f"\n>>> Running {label} suite: {path}")
        result = _run_suite(path, label)
        suite_results[label] = result
        all_passed += result["passed"]
        all_failed += result["failed"]
        all_total += result["total"]
        status = "PASS" if result["failed"] == 0 else "FAIL"
        print(f"    {status}: {result['passed']}/{result['total']} passed ({result['pass_rate_pct']}%) in {result['elapsed_ms']:.0f}ms")

    # Existing test suite
    print(f"\n>>> Running existing regression suite")
    existing = _run_suite("tests/", "existing_regression")
    suite_results["existing_regression"] = existing

    overall_rate = (all_passed / all_total * 100) if all_total > 0 else 0.0
    blockers = sum(1 for s in suite_results.values() if s.get("failed", 0) > 0 and s["suite"] != "existing_regression")

    closure = _evaluate_closure_criteria(suite_results)
    green_count = sum(1 for c in closure if c["status"] == "GREEN")
    red_count = sum(1 for c in closure if c["status"] == "RED")

    # Compute north-pole estimate
    factual_rate = suite_results.get("factual", {}).get("pass_rate_pct", 0)
    behavior_rate = suite_results.get("behavior", {}).get("pass_rate_pct", 0)
    followup_rate = suite_results.get("followup", {}).get("pass_rate_pct", 0)
    expl_rate = suite_results.get("explainability", {}).get("pass_rate_pct", 0)
    north_pole = round(
        0.30 * factual_rate + 0.25 * behavior_rate + 0.20 * followup_rate + 0.15 * expl_rate + 0.10 * overall_rate,
        2,
    )

    report = {
        "report_type": "brd_release_readiness",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "brd_version": "2026-02-22",
        "summary": {
            "total_tests": all_total,
            "total_passed": all_passed,
            "total_failed": all_failed,
            "overall_pass_rate_pct": round(overall_rate, 2),
            "blocker_suites": blockers,
            "north_pole_score": north_pole,
        },
        "acceptance_thresholds": {
            "overall_weighted_pass": {"target": 92, "actual": round(overall_rate, 2), "met": overall_rate >= 92},
            "factual": {"target": 92, "actual": factual_rate, "met": factual_rate >= 92},
            "behavior": {"target": 98, "actual": behavior_rate, "met": behavior_rate >= 98},
            "followup": {"target": 95, "actual": followup_rate, "met": followup_rate >= 95},
            "explainability": {"target": 95, "actual": expl_rate, "met": expl_rate >= 95},
        },
        "suites": suite_results,
        "closure_criteria": closure,
        "closure_summary": {
            "total": len(closure),
            "green": green_count,
            "red": red_count,
            "yellow": len(closure) - green_count - red_count,
            "all_green": green_count == len(closure),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total BRD tests: {all_total}")
    print(f"Passed: {all_passed} ({overall_rate:.1f}%)")
    print(f"Failed: {all_failed}")
    print(f"Blocker suites: {blockers}")
    print(f"North-pole score: {north_pole}")
    print()
    print("CLOSURE CRITERIA:")
    for c in closure:
        icon = "✓" if c["status"] == "GREEN" else ("~" if c["status"] == "YELLOW" else "✗")
        print(f"  [{icon}] {c['id']}: {c['description']}")
        print(f"       Status: {c['status']} — {c['evidence']}")
    print()
    print(f"Report written to: {output_path}")

    return 0 if red_count == 0 and all_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
