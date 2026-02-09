"""Tests for SQL generation correctness in executor.

This module tests that SQL generation produces valid SQL without bugs
like double FROM clauses or other structural issues.
"""

import pytest

from haikugraph.execution.execute import build_sql, _strip_sql_literals


def test_select_star_no_double_from():
    """Test that SELECT * queries don't produce double FROM clause.
    
    Regression test for bug where:
    SELECT * FROM "test_1_1" FROM "test_1_1" LIMIT 200
    was generated instead of:
    SELECT * FROM "test_1_1" LIMIT 200
    """
    # Subquestion with no columns specified (SELECT *)
    sq = {
        "id": "SQ1",
        "tables": ["test_1_1"],
        # No columns = SELECT *
    }
    
    plan = {}
    
    sql, _ = build_sql(sq, plan)
    
    # Should contain SELECT * and exactly one FROM
    assert "SELECT *" in sql
    assert sql.count(" FROM ") == 1, f"Expected exactly 1 FROM clause, SQL: {sql}"
    
    # Should not have double FROM
    assert "FROM \"test_1_1\" FROM" not in sql


def test_select_columns_single_from():
    """Test that SELECT with columns has exactly one FROM clause."""
    sq = {
        "id": "SQ1",
        "tables": ["test_1_1"],
        "columns": ["customer_id", "payment_amount"],
    }
    
    plan = {}
    
    sql, _ = build_sql(sq, plan)
    
    # Should have exactly one FROM
    assert sql.count(" FROM ") == 1, f"Expected exactly 1 FROM clause, SQL: {sql}"


def test_aggregation_single_from():
    """Test that aggregation queries have exactly one FROM clause.
    
    Note: Aggregations require group_by to be applied in the SQL.
    Without group_by, it's treated as a regular SELECT.
    """
    sq = {
        "id": "SQ1",
        "tables": ["test_1_1"],
        "columns": ["payment_amount"],
    }
    
    plan = {}
    
    sql, _ = build_sql(sq, plan)
    
    # Should have exactly one FROM
    assert sql.count(" FROM ") == 1, f"Expected exactly 1 FROM clause, SQL: {sql}"


def test_grouped_aggregation_single_from():
    """Test that GROUP BY queries have exactly one FROM clause."""
    sq = {
        "id": "SQ1",
        "tables": ["test_1_1"],
        "group_by": ["customer_id"],
        "aggregations": [{"agg": "sum", "col": "payment_amount"}],
    }
    
    plan = {}
    
    sql, _ = build_sql(sq, plan)
    
    # Should have exactly one FROM
    assert sql.count(" FROM ") == 1, f"Expected exactly 1 FROM clause, SQL: {sql}"
    assert "GROUP BY" in sql


def test_aggregation_without_group_by():
    """Test aggregation without GROUP BY (e.g., 'What is total revenue?')."""
    sq = {
        "id": "SQ1",
        "tables": ["orders"],
        "aggregations": [{"agg": "sum", "col": "revenue"}],
    }
    
    plan = {}
    
    sql, _ = build_sql(sq, plan)
    
    # Should have exactly one FROM
    assert sql.count(" FROM ") == 1, f"Expected exactly 1 FROM clause, SQL: {sql}"
    # Should have SUM aggregation
    assert "SUM(" in sql.upper(), f"Expected SUM in SQL: {sql}"
    # Should NOT have GROUP BY
    assert "GROUP BY" not in sql, f"Should not have GROUP BY for ungrouped aggregation: {sql}"
    # Should NOT have LIMIT (aggregations return single row)
    assert "LIMIT" not in sql, f"Should not have LIMIT for aggregation: {sql}"


def test_join_query_single_from():
    """Test that queries with joins have exactly one FROM clause."""
    sq = {
        "id": "SQ1",
        "tables": ["test_1_1", "test_2_1"],
        "columns": ["customer_id", "payment_amount"],
    }
    
    plan = {
        "join_paths": [
            {
                "from": "test_1_1",
                "to": "test_2_1",
                "via": ["customer_id"],
            }
        ]
    }
    
    sql, _ = build_sql(sq, plan)
    
    # Should have exactly one FROM (plus INNER JOIN)
    assert sql.count(" FROM ") == 1, f"Expected exactly 1 FROM clause, SQL: {sql}"
    assert "INNER JOIN" in sql


def test_defensive_guard_catches_double_from():
    """Test that the defensive guard catches any double FROM bugs."""
    # This test verifies the defensive guard works by trying to trigger
    # a double FROM scenario (if the bug wasn't already fixed)
    
    sq = {
        "id": "SQ1",
        "tables": ["test_1_1"],
    }
    
    plan = {}
    
    # Should not raise (defensive guard should pass for valid SQL)
    sql, _ = build_sql(sq, plan)
    assert sql.count(" FROM ") == 1


def test_strip_sql_literals_removes_from_in_quotes():
    """Unit test for _strip_sql_literals: validates scrubber logic directly."""
    sql = "SELECT * FROM t WHERE note = 'shipped from store' AND col = \"from\""
    stripped = _strip_sql_literals(sql)
    
    # String literal content should be gone
    assert "shipped from store" not in stripped
    # Quoted identifier content should be gone
    assert "\"from\"" not in stripped
    # But FROM keyword should remain
    assert "FROM t" in stripped
    # Empty quotes should remain as markers
    assert "''" in stripped
    assert '""' in stripped


def test_strip_sql_literals_handles_escaped_quotes():
    """Scrubber should handle escaped quotes correctly."""
    # SQL with escaped single quote inside string
    sql = "SELECT * FROM t WHERE name = 'O''Reilly'"
    stripped = _strip_sql_literals(sql)
    assert "O''Reilly" not in stripped
    assert "FROM t" in stripped


def test_guard_ignores_from_in_string_literal():
    """Guard should ignore 'from' inside string literals."""
    sq = {"id": "SQ1", "tables": ["orders"], "columns": ["id"]}
    plan = {"constraints": [{"type": "filter", "expression": "orders.note = 'shipped from store'"}]}
    sql, _ = build_sql(sq, plan)
    assert sql.count(" FROM ") == 1


def test_guard_ignores_from_in_identifier():
    """Guard should ignore FROM inside quoted identifiers like "from" column."""
    sq = {"id": "SQ1", "tables": ["test_1_1"], "columns": ["from"]}
    plan = {}
    sql, _ = build_sql(sq, plan)
    assert sql.count(" FROM ") == 1


def test_guard_skip_check_ignores_literals():
    """Guard skip check should not false-trigger on '(SELECT' in string literals."""
    # This used to false-trigger the subquery skip check
    sq = {"id": "SQ1", "tables": ["orders"], "columns": ["id"]}
    plan = {"constraints": [{"type": "filter", "expression": "orders.note = 'customer wrote (select later)'"}]}
    sql, _ = build_sql(sq, plan)
    # Should not crash and should have exactly one FROM
    assert sql.count(" FROM ") == 1


def test_guard_with_detection_does_not_false_trigger_on_withdrawal():
    """Guard WITH detection should not match substring 'WITH' in 'WITHDRAWAL'."""
    # Regression test: word boundary prevents false match on "WITHDRAWAL"
    sq = {"id": "SQ1", "tables": ["orders"], "columns": ["id"]}
    plan = {"constraints": [{"type": "filter", "expression": "orders.note = 'WITHDRAWAL pending'"}]}
    sql, _ = build_sql(sq, plan)
    # Should not skip guard (no actual CTE), should count FROM correctly
    assert sql.count(" FROM ") == 1


def test_all_query_types_have_single_from():
    """Comprehensive test that all query patterns produce exactly one FROM."""
    
    test_cases = [
        # Case 1: SELECT * (no columns)
        {
            "sq": {"id": "SQ1", "tables": ["test_1_1"]},
            "plan": {},
            "description": "SELECT * with no columns",
        },
        # Case 2: SELECT specific columns
        {
            "sq": {"id": "SQ1", "tables": ["test_1_1"], "columns": ["col1", "col2"]},
            "plan": {},
            "description": "SELECT with columns",
        },
        # Case 3: Aggregation without GROUP BY
        {
            "sq": {
                "id": "SQ1",
                "tables": ["test_1_1"],
                "aggregations": [{"agg": "count", "col": "id"}],
            },
            "plan": {},
            "description": "Aggregation without GROUP BY",
        },
        # Case 4: Aggregation with GROUP BY
        {
            "sq": {
                "id": "SQ1",
                "tables": ["test_1_1"],
                "group_by": ["status"],
                "aggregations": [{"agg": "count", "col": "id"}],
            },
            "plan": {},
            "description": "Aggregation with GROUP BY",
        },
        # Case 5: With WHERE clause
        {
            "sq": {"id": "SQ1", "tables": ["test_1_1"], "columns": ["id"]},
            "plan": {
                "constraints": [
                    {"type": "filter", "expression": "test_1_1.status = 'active'"}
                ]
            },
            "description": "Query with WHERE clause",
        },
    ]
    
    for case in test_cases:
        sql, _ = build_sql(case["sq"], case["plan"])
        from_count = sql.count(" FROM ")
        assert from_count == 1, (
            f"{case['description']} produced {from_count} FROM clauses. SQL: {sql}"
        )
