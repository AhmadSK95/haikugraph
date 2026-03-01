from __future__ import annotations

from haikugraph.qa.provider_parity import build_provider_parity_report


def test_provider_parity_report_alerts_on_contract_drift() -> None:
    rows = [
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

    report = build_provider_parity_report(rows, contract_drift_threshold=0.01)
    assert report["status"] == "alert"
    assert any(a.get("type") == "contract_drift" for a in report.get("alerts", []))


def test_provider_parity_report_ok_when_contracts_align() -> None:
    rows = [
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
                "metric": "transaction_count",
                "table": "datada_mart_transactions",
                "dimensions": [],
                "time_scope": "all",
            },
        },
    ]

    report = build_provider_parity_report(rows, contract_drift_threshold=0.50)
    assert report["status"] == "ok"
    assert report["summary"]["contract_drift_cases"] == 0
