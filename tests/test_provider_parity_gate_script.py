from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def test_provider_parity_gate_script_generates_report() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_provider_parity_gate.py"

    sample = {
        "rows": [
            {
                "goal": "How many transactions are there?",
                "mode": "deterministic",
                "success": True,
                "contract_spec": {
                    "metric": "transaction_count",
                    "table": "datada_mart_transactions",
                    "dimensions": [],
                    "time_scope": "all",
                },
            },
            {
                "goal": "How many transactions are there?",
                "mode": "openai",
                "success": True,
                "contract_spec": {
                    "metric": "quote_count",
                    "table": "datada_mart_quotes",
                    "dimensions": [],
                    "time_scope": "all",
                },
            },
        ]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_json = tmp / "input.json"
        out_json = tmp / "parity.json"
        out_md = tmp / "parity.md"
        input_json.write_text(json.dumps(sample), encoding="utf-8")

        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--input-json",
                str(input_json),
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
                "--contract-drift-threshold",
                "0.01",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert out_json.exists()
        assert out_md.exists()

        report = json.loads(out_json.read_text(encoding="utf-8"))
        assert report.get("status") == "alert"
        assert any(a.get("type") == "contract_drift" for a in report.get("alerts", []))
