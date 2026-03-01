"""KPI decomposition helpers with owner/target catalog enrichment."""

from __future__ import annotations

from typing import Any


_DEFAULT_OWNER_BY_DOMAIN: dict[str, str] = {
    "transactions": "payments_analytics",
    "quotes": "fx_analytics",
    "customers": "customer_analytics",
    "bookings": "deals_analytics",
    "documents": "knowledge_ops",
}

_TABLE_TO_DOMAIN: dict[str, str] = {
    "datada_mart_transactions": "transactions",
    "datada_mart_quotes": "quotes",
    "datada_dim_customers": "customers",
    "datada_mart_bookings": "bookings",
    "datada_document_chunks": "documents",
}

_KPI_CATALOG: dict[str, dict[str, Any]] = {
    "transactions.transaction_count": {
        "owner": "payments_ops",
        "owner_role": "Director, Payments Operations",
        "target": 1000.0,
        "target_direction": "up",
        "unit": "count",
        "cadence": "monthly",
    },
    "transactions.total_amount": {
        "owner": "treasury_ops",
        "owner_role": "Head of Treasury",
        "target": 250000.0,
        "target_direction": "up",
        "unit": "currency",
        "cadence": "monthly",
    },
    "quotes.quote_count": {
        "owner": "fx_product",
        "owner_role": "FX Product Manager",
        "target": 1500.0,
        "target_direction": "up",
        "unit": "count",
        "cadence": "monthly",
    },
    "quotes.forex_markup_revenue": {
        "owner": "fx_revenue",
        "owner_role": "VP Revenue",
        "target": 35000.0,
        "target_direction": "up",
        "unit": "currency",
        "cadence": "monthly",
    },
    "customers.customer_count": {
        "owner": "growth_ops",
        "owner_role": "Growth Operations Lead",
        "target": 15000.0,
        "target_direction": "up",
        "unit": "count",
        "cadence": "quarterly",
    },
    "bookings.booking_count": {
        "owner": "deals_ops",
        "owner_role": "Head of Deal Operations",
        "target": 800.0,
        "target_direction": "up",
        "unit": "count",
        "cadence": "monthly",
    },
}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _infer_domain(plan: dict[str, Any]) -> str:
    domain = str(plan.get("domain") or "").strip().lower()
    if domain:
        return domain
    table = str(plan.get("table") or "").strip().lower()
    return _TABLE_TO_DOMAIN.get(table, "general")


def _infer_kpi_id(plan: dict[str, Any]) -> str:
    canonical = str(plan.get("canonical_metric_id") or "").strip()
    if canonical:
        return canonical
    metric = str(plan.get("metric") or "").strip()
    domain = _infer_domain(plan)
    if metric:
        return f"{domain}.{metric}" if "." not in metric else metric
    return f"{domain}.metric"


def _observed_metric_value(execution: dict[str, Any]) -> float | None:
    rows = execution.get("sample_rows") or []
    if not isinstance(rows, list) or not rows:
        return None
    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        val = _to_float(row.get("metric_value"))
        if val is not None:
            values.append(val)
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    # For grouped outputs, sum supports total KPI accountability view.
    return float(sum(values))


def build_kpi_decomposition(
    *,
    plan: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    """Build KPI ownership/target decomposition from plan + execution."""
    kpi_id = _infer_kpi_id(plan)
    metric_expr = str(plan.get("metric_expr") or "")
    domain = _infer_domain(plan)
    catalog_cfg = dict(_KPI_CATALOG.get(kpi_id) or {})

    owner = str(
        catalog_cfg.get("owner")
        or _DEFAULT_OWNER_BY_DOMAIN.get(domain)
        or "analytics"
    )
    owner_role = str(catalog_cfg.get("owner_role") or "Analytics Owner")
    target = _to_float(catalog_cfg.get("target"))
    target_direction = str(catalog_cfg.get("target_direction") or "up")
    unit = str(catalog_cfg.get("unit") or "number")
    cadence = str(catalog_cfg.get("cadence") or "monthly")
    observed_row_count = int(execution.get("row_count") or 0)
    observed_value = _observed_metric_value(execution)

    variance_to_target = None
    attainment_pct = None
    target_status = "not_evaluated"
    if target is not None and observed_value is not None:
        variance_to_target = float(observed_value - target)
        if abs(target) > 1e-9:
            attainment_pct = round((observed_value / target) * 100.0, 2)
        if target_direction == "up":
            target_status = "on_track" if observed_value >= target else "below_target"
        else:
            target_status = "on_track" if observed_value <= target else "above_target"

    dimensions = [str(dim) for dim in (plan.get("dimensions") or []) if str(dim)]
    children: list[dict[str, Any]] = []
    for dim in dimensions[:6]:
        children.append(
            {
                "node_type": "dimension_driver",
                "name": dim,
                "description": f"Slice KPI by {dim}",
            }
        )
    time_filter = plan.get("time_filter")
    if time_filter:
        children.append(
            {
                "node_type": "time_scope",
                "name": str(time_filter),
                "description": "Applied reporting period for KPI evaluation",
            }
        )

    secondary_metric = str(
        plan.get("secondary_canonical_metric_id")
        or plan.get("secondary_metric")
        or ""
    ).strip()
    if secondary_metric:
        children.append(
            {
                "node_type": "comparison_metric",
                "name": secondary_metric,
                "description": "Secondary KPI used for paired interpretation",
            }
        )

    children.append(
        {
            "node_type": "ownership",
            "name": owner,
            "description": f"{owner_role}; cadence={cadence}",
        }
    )

    root = {
        "kpi_id": kpi_id,
        "formula": metric_expr,
        "domain": domain,
        "owner": owner,
        "owner_role": owner_role,
        "target": target,
        "target_direction": target_direction,
        "target_status": target_status,
        "target_source": "kpi_catalog" if catalog_cfg else "domain_default",
        "unit": unit,
        "cadence": cadence,
        "observed_row_count": observed_row_count,
        "observed_value": observed_value,
        "variance_to_target": variance_to_target,
        "attainment_pct": attainment_pct,
    }
    return {"root": root, "children": children}

