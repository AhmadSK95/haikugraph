from __future__ import annotations

import subprocess
from pathlib import Path


def test_repo_hygiene_script_passes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "repo_hygiene_check.sh"
    completed = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"script failed with code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert "[hygiene] PASS" in completed.stdout
