"""Tests for VARCHAR timestamp handling and scalar comparison queries.

This module tests:
- VARCHAR timestamp columns with date_trunc
- VARCHAR timestamp columns with time constraints
- Comparison queries producing scalar results (not time-series)
- Comparison queries with explicit time-series requests
"""

import pytest
import duckdb
from pathlib import Path
import tempfile
from haikugraph.execution.execute import (
    build_sql,
    get_timestamp_expression,
    translate_time_constraint,
    execute_plan,
)
from haikugraph.planning.schema import validate_plan_or_raise


class TestVarcharTimestampHandling:
    """Test VARCHAR timestamp column handling."""
    
    def test_get_timestamp_expression_with_varchar(self):
        """Test that VARCHAR timestamp columns are wrapped with TRY_CAST."""
        # Create temp database with VARCHAR timestamp column
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = duckdb.connect(str(db_path))
            
            # Create table with VARCHAR timestamp
            conn.execute("""
                CREATE TABLE test_varchar_ts (
                    id INTEGER,
                    created_at VARCHAR,
                    amount DOUBLE
                )
            """)
            
            # Test get_timestamp_expression
            ts_expr = get_timestamp_expression("test_varchar_ts", "created_at", conn)
            
            # Should wrap with TRY_CAST
            assert "TRY_CAST" in ts_expr
            assert "TIMESTAMP" in ts_expr
            assert "test_varchar_ts" in ts_expr
            assert "created_at" in ts_expr
            
            conn.close()
    
    def test_get_timestamp_expression_with_real_timestamp(self):
        """Test that real TIMESTAMP columns are used as-is."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = duckdb.connect(str(db_path))
            
            # Create table with TIMESTAMP column
            conn.execute("""
                CREATE TABLE test_real_ts (
                    id INTEGER,
                    created_at TIMESTAMP,
                    amount DOUBLE
                )
            """)
            
            # Test get_timestamp_expression
            ts_expr = get_timestamp_expression("test_real_ts", "created_at", conn)
            
            # Should NOT wrap with TRY_CAST
            assert "TRY_CAST" not in ts_expr
            assert ts_expr == '"test_real_ts"."created_at"'
            
            conn.close()
    
    def test_time_bucket_with_varchar_timestamp(self):
        """Test monthly bucketing with VARCHAR timestamp column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = duckdb.connect(str(db_path))
            
            # Create table with VARCHAR timestamps
            conn.execute("""
                CREATE TABLE orders (
                    id INTEGER,
                    created_at VARCHAR,
                    revenue DOUBLE
                )
            """)
            
            # Insert test data with VARCHAR timestamps
            conn.execute("""
                INSERT INTO orders VALUES
                (1, '2024-01-15', 100.0),
                (2, '2024-02-20', 200.0),
                (3, '2024-01-25', 150.0)
            """)
            
            # Test plan with monthly time bucket
            plan = {
                "original_question": "Monthly revenue",
                "subquestions": [
                    {
                        "id": "SQ1",
                        "tables": ["orders"],
                        "group_by": [
                            {"type": "time_bucket", "grain": "month", "col": "created_at"}
                        ],
                        "aggregations": [
                            {"agg": "sum", "col": "revenue"}
                        ]
                    }
                ]
            }
            
            # Build SQL
            sql, metadata = build_sql(plan["subquestions"][0], plan, conn)
            
            # Should include TRY_CAST for VARCHAR column
            assert "TRY_CAST" in sql
            assert "date_trunc" in sql
            assert "month" in sql
            
            # Execute and verify it works
            result = conn.execute(sql).fetchall()
            assert len(result) == 2  # Two months
            
            conn.close()
    
    def test_time_constraint_with_varchar_timestamp(self):
        """Test time constraints with VARCHAR timestamp columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = duckdb.connect(str(db_path))
            
            # Create table with VARCHAR timestamp
            conn.execute("""
                CREATE TABLE events (
                    id INTEGER,
                    event_date VARCHAR,
                    count INTEGER
                )
            """)
            
            # Test translate_time_constraint with VARCHAR
            constraint_expr = "events.event_date in this_month"
            sql_expr = translate_time_constraint(constraint_expr, conn)
            
            # Should include TRY_CAST for VARCHAR column
            assert "TRY_CAST" in sql_expr
            assert "TIMESTAMP" in sql_expr
            assert "date_trunc" in sql_expr
            
            conn.close()


class TestScalarComparisonQueries:
    """Test that comparison queries produce scalar results by default."""
    
    def test_scalar_comparison_plan_validation(self):
        """Test plan validation for scalar comparison (no group_by)."""
        plan = {
            "original_question": "Revenue this month vs last month",
            "subquestions": [
                {
                    "id": "SQ1_current",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "revenue"}]
                },
                {
                    "id": "SQ2_comparison",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "revenue"}]
                }
            ],
            "constraints": [
                {"type": "time", "expression": "orders.created_at in this_month", "applies_to": "SQ1_current"},
                {"type": "time", "expression": "orders.created_at in previous_month", "applies_to": "SQ2_comparison"}
            ]
        }
        
        # Should validate successfully (no group_by is valid)
        validate_plan_or_raise(plan)
        
        # Verify no group_by in subquestions
        for sq in plan["subquestions"]:
            assert "group_by" not in sq or sq.get("group_by") is None
    
    def test_scalar_comparison_sql_generation(self):
        """Test SQL generation for scalar comparison."""
        plan = {
            "original_question": "Revenue this year vs last year",
            "subquestions": [
                {
                    "id": "SQ1_current",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "revenue"}]
                }
            ],
            "constraints": [
                {"type": "time", "expression": "orders.created_at in this_year", "applies_to": "SQ1_current"}
            ]
        }
        
        sql, metadata = build_sql(plan["subquestions"][0], plan)
        
        # Should NOT have GROUP BY
        assert "GROUP BY" not in sql
        # Should have aggregation
        assert "SUM(" in sql
        # Should have time constraint
        assert "WHERE" in sql
    
    def test_timeseries_comparison_with_explicit_groupby(self):
        """Test time-series comparison when explicitly requested."""
        plan = {
            "original_question": "Monthly revenue this year vs last year",
            "subquestions": [
                {
                    "id": "SQ1_current",
                    "tables": ["orders"],
                    "group_by": [
                        {"type": "time_bucket", "grain": "month", "col": "created_at"}
                    ],
                    "aggregations": [{"agg": "sum", "col": "revenue"}]
                }
            ]
        }
        
        sql, metadata = build_sql(plan["subquestions"][0], plan)
        
        # Should HAVE GROUP BY for time-series
        assert "GROUP BY" in sql
        assert "date_trunc" in sql
        assert "month" in sql
    
    def test_unique_customers_monthly_should_have_groupby(self):
        """Test that 'monthly unique customers' produces time-series, not scalar."""
        plan = {
            "original_question": "Monthly unique customers",
            "subquestions": [
                {
                    "id": "SQ1",
                    "tables": ["orders"],
                    "group_by": [
                        {"type": "time_bucket", "grain": "month", "col": "created_at"}
                    ],
                    "aggregations": [
                        {"agg": "count", "col": "customer_id", "distinct": True}
                    ]
                }
            ]
        }
        
        sql, metadata = build_sql(plan["subquestions"][0], plan)
        
        # Should have GROUP BY for monthly
        assert "GROUP BY" in sql
        assert "date_trunc" in sql
        # Should have COUNT(DISTINCT
        assert "COUNT(DISTINCT" in sql
    
    def test_unique_customers_total_should_not_have_groupby(self):
        """Test that 'total unique customers' produces scalar, not time-series."""
        plan = {
            "original_question": "Total unique customers",
            "subquestions": [
                {
                    "id": "SQ1",
                    "tables": ["orders"],
                    "aggregations": [
                        {"agg": "count", "col": "customer_id", "distinct": True}
                    ]
                }
            ]
        }
        
        sql, metadata = build_sql(plan["subquestions"][0], plan)
        
        # Should NOT have GROUP BY for total
        assert "GROUP BY" not in sql
        # Should have COUNT(DISTINCT
        assert "COUNT(DISTINCT" in sql


class TestComparisonTimeScoping:
    """Test symmetric time scoping for comparison queries."""
    
    def test_comparison_with_scoped_constraints(self):
        """Test that comparison queries require scoped time constraints."""
        plan = {
            "original_question": "This month vs last month",
            "subquestions": [
                {
                    "id": "SQ1_current",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "revenue"}]
                },
                {
                    "id": "SQ2_comparison",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "revenue"}]
                }
            ],
            "constraints": [
                {"type": "time", "expression": "orders.created_at in this_month", "applies_to": "SQ1_current"},
                {"type": "time", "expression": "orders.created_at in previous_month", "applies_to": "SQ2_comparison"}
            ]
        }
        
        # Should validate successfully
        validate_plan_or_raise(plan)
    
    def test_comparison_without_scoped_constraints_fails(self):
        """Test that comparison queries without scoped constraints fail validation."""
        plan = {
            "original_question": "This month vs last month",
            "subquestions": [
                {
                    "id": "SQ1_current",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "revenue"}]
                },
                {
                    "id": "SQ2_comparison",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "revenue"}]
                }
            ],
            "constraints": [
                {"type": "time", "expression": "orders.created_at in this_month"}  # Missing applies_to!
            ]
        }
        
        # Should fail validation
        with pytest.raises(ValueError, match="unscoped time constraint"):
            validate_plan_or_raise(plan)
    
    def test_year_comparison_with_scoping(self):
        """Test year-over-year comparison with proper scoping."""
        plan = {
            "original_question": "Revenue this year vs last year",
            "subquestions": [
                {
                    "id": "SQ1_current",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "revenue"}]
                },
                {
                    "id": "SQ2_comparison",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "revenue"}]
                }
            ],
            "constraints": [
                {"type": "time", "expression": "orders.created_at in this_year", "applies_to": "SQ1_current"},
                {"type": "time", "expression": "orders.created_at in previous_year", "applies_to": "SQ2_comparison"}
            ]
        }
        
        # Should validate successfully
        validate_plan_or_raise(plan)
        
        # Both subquestions should have different time constraints
        assert plan["constraints"][0]["applies_to"] == "SQ1_current"
        assert plan["constraints"][1]["applies_to"] == "SQ2_comparison"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
