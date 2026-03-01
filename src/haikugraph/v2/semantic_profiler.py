"""Dataset-agnostic semantic profiler for v2 runtime."""

from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb

from haikugraph.v2.types import JoinEdgeV2, SemanticCatalogV2, TableProfileV2


def _dataset_signature(db_path: str, table_payload: list[dict]) -> str:
    raw = f"{Path(db_path).resolve()}::{table_payload!r}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _schema_signature(db_path: str, schema_payload: list[dict]) -> str:
    raw = f"{Path(db_path).resolve()}::{schema_payload!r}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _profile_table(conn: duckdb.DuckDBPyConnection, table_name: str) -> TableProfileV2:
    row_count = int(conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0] or 0)
    columns = [
        str(r[0])
        for r in conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='main' AND table_name=?
            ORDER BY ordinal_position
            """,
            [table_name],
        ).fetchall()
    ]
    id_like = [c for c in columns if c.lower().endswith("_id") or c.lower().endswith("_key")]

    top_null: list[dict[str, float]] = []
    for col in columns[:16]:
        if row_count <= 0:
            break
        null_pct = float(
            conn.execute(
                f'SELECT 100.0 * AVG(CASE WHEN "{col}" IS NULL THEN 1.0 ELSE 0.0 END) FROM "{table_name}"'
            ).fetchone()[0]
            or 0.0
        )
        top_null.append({"column": col, "null_pct": round(null_pct, 2)})
    top_null.sort(key=lambda x: x["null_pct"], reverse=True)
    return TableProfileV2(
        table=table_name,
        row_count=row_count,
        columns=columns,
        id_like_columns=id_like,
        top_null_columns=top_null[:8],
    )


def _infer_join_edges(
    conn: duckdb.DuckDBPyConnection,
    tables: list[TableProfileV2],
    *,
    max_pairs: int = 6,
) -> list[JoinEdgeV2]:
    edges: list[JoinEdgeV2] = []
    pairs_checked = 0
    for i, left in enumerate(tables):
        for right in tables[i + 1 :]:
            common_keys = sorted(set(left.id_like_columns) & set(right.id_like_columns))
            if not common_keys:
                continue
            for key in common_keys[:2]:
                if pairs_checked >= max_pairs:
                    return edges
                pairs_checked += 1
                query = f"""
                WITH l AS (
                    SELECT DISTINCT CAST("{key}" AS VARCHAR) AS k
                    FROM "{left.table}"
                    WHERE "{key}" IS NOT NULL
                ),
                r AS (
                    SELECT DISTINCT CAST("{key}" AS VARCHAR) AS k
                    FROM "{right.table}"
                    WHERE "{key}" IS NOT NULL
                )
                SELECT
                    (SELECT COUNT(*) FROM l) AS l_cnt,
                    (SELECT COUNT(*) FROM r) AS r_cnt,
                    (SELECT COUNT(*) FROM l INNER JOIN r USING(k)) AS inter_cnt
                """
                l_cnt, r_cnt, inter_cnt = conn.execute(query).fetchone()
                l_cnt = int(l_cnt or 0)
                r_cnt = int(r_cnt or 0)
                inter_cnt = int(inter_cnt or 0)
                left_cov = round((100.0 * inter_cnt / max(1, l_cnt)), 2)
                right_cov = round((100.0 * inter_cnt / max(1, r_cnt)), 2)
                confidence = round(min(left_cov, right_cov) / 100.0, 4)
                risk = "low" if confidence >= 0.75 else ("medium" if confidence >= 0.4 else "high")
                edges.append(
                    JoinEdgeV2(
                        left_table=left.table,
                        right_table=right.table,
                        key_column=key,
                        left_coverage_pct=left_cov,
                        right_coverage_pct=right_cov,
                        confidence=confidence,
                        risk=risk,
                    )
                )
    return edges


def profile_dataset(db_path: str) -> SemanticCatalogV2:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        table_names = [
            str(r[0])
            for r in conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema='main'
                ORDER BY table_name
                """
            ).fetchall()
        ]
        tables = [_profile_table(conn, t) for t in table_names if t]
        edges = _infer_join_edges(conn, tables)
        all_columns = [c for t in tables for c in t.columns]
        entities = sorted({c for c in all_columns if c.endswith("_id") or c.endswith("_key")})[:40]
        measures = sorted(
            {
                c
                for c in all_columns
                if any(tok in c.lower() for tok in ["amount", "revenue", "count", "rate", "value", "markup"])
            }
        )[:40]
        dimensions = sorted(
            {
                c
                for c in all_columns
                if any(tok in c.lower() for tok in ["country", "state", "status", "type", "platform", "currency"])
            }
        )[:60]
        time_fields = sorted({c for c in all_columns if c.lower().endswith("_ts") or c.lower().endswith("_at")})[:40]

        table_payload = [{"table": t.table, "rows": t.row_count, "cols": len(t.columns)} for t in tables]
        schema_payload = [{"table": t.table, "columns": list(t.columns)} for t in tables]
        signature = _dataset_signature(db_path, table_payload)
        schema_signature = _schema_signature(db_path, schema_payload)
        high_risk_edges = sum(1 for e in edges if e.risk == "high")
        sparse_tables = sum(
            1
            for t in tables
            if any(float(n.get("null_pct", 0.0)) >= 65.0 for n in (t.top_null_columns or []))
        )
        quality_summary = {
            "table_count": float(len(tables)),
            "high_risk_join_edges": float(high_risk_edges),
            "sparse_table_count": float(sparse_tables),
        }
        return SemanticCatalogV2(
            dataset_signature=signature,
            schema_signature=schema_signature,
            tables=tables,
            join_edges=edges,
            entities=entities,
            measures=measures,
            dimensions=dimensions,
            time_fields=time_fields,
            quality_summary=quality_summary,
        )
    finally:
        conn.close()
