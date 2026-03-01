from __future__ import annotations

from haikugraph.poc.kpi_decomposition import build_kpi_decomposition


def test_kpi_decomposition_uses_catalog_owner_and_target():
    plan = {
        "canonical_metric_id": "transactions.transaction_count",
        "metric_expr": "COUNT(*)",
        "domain": "transactions",
        "dimensions": ["platform_name"],
        "time_filter": {"kind": "month_year", "month": 12, "year": 2025},
    }
    execution = {
        "row_count": 2,
        "sample_rows": [
            {"platform_name": "B2C-APP", "metric_value": 600},
            {"platform_name": "B2C-WEB", "metric_value": 550},
        ],
    }
    payload = build_kpi_decomposition(plan=plan, execution=execution)
    root = payload.get("root") or {}
    assert root.get("owner") == "payments_ops"
    assert root.get("target") == 1000.0
    assert root.get("target_status") == "on_track"
    assert float(root.get("attainment_pct") or 0.0) >= 100.0


def test_kpi_decomposition_falls_back_to_domain_owner_when_metric_unknown():
    plan = {
        "metric": "custom_metric_x",
        "metric_expr": "SUM(metric_value)",
        "table": "datada_mart_quotes",
        "dimensions": [],
    }
    execution = {"row_count": 1, "sample_rows": [{"metric_value": 42}]}
    payload = build_kpi_decomposition(plan=plan, execution=execution)
    root = payload.get("root") or {}
    assert root.get("domain") == "quotes"
    assert root.get("owner") == "fx_analytics"
    assert root.get("target_source") == "domain_default"


def test_kpi_decomposition_contains_dimension_time_and_ownership_children():
    plan = {
        "canonical_metric_id": "quotes.forex_markup_revenue",
        "metric_expr": "SUM(forex_markup_revenue)",
        "domain": "quotes",
        "dimensions": ["source_currency", "destination_currency"],
        "time_filter": {"kind": "year_only", "year": 2025},
        "secondary_metric": "quote_count",
    }
    execution = {"row_count": 3, "sample_rows": [{"metric_value": 1000}, {"metric_value": 900}]}
    payload = build_kpi_decomposition(plan=plan, execution=execution)
    children = payload.get("children") or []
    node_types = {str(node.get("node_type")) for node in children if isinstance(node, dict)}
    assert "dimension_driver" in node_types
    assert "time_scope" in node_types
    assert "comparison_metric" in node_types
    assert "ownership" in node_types

