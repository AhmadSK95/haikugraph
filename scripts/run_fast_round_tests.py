#!/usr/bin/env python3
"""Run a fast, impact-based test slice for iterative product rounds."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from haikugraph.qa.fast_round_runner import select_fast_suites


def _changed_paths(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fast round tests.")
    parser.add_argument("--changed-only", action="store_true", help="Select suites from git diff paths.")
    parser.add_argument("--print-only", action="store_true", help="Print selected suites without running pytest.")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    changed = _changed_paths(repo_root) if args.changed_only else []
    suites = select_fast_suites(changed)
    if not suites:
        print("No suites selected.")
        return 0

    print("Selected fast suites:")
    for suite in suites:
        print(f"- {suite}")
    if args.print_only:
        return 0

    cmd = ["pytest", "-q", *suites]
    print("\nExecuting:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
