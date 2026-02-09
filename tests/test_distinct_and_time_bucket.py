"""Tests for DISTINCT and time bucketing fixes.

This module tests the complete flow:
- DISTINCT counts with proper schema validation
- Time bucketing for monthly/yearly aggregations
- SQL generation for COUNT(DISTINCT col)
- Prevention of SQL injection hacks in column names
"""

import pytest
from haikugraph.planning.schema import AggregationSpec, Plan, validate_plan
from haikugraph.execution.execute import build_sql


class TestDistinctAggregation:
    """Test DISTINCT support in aggregations."""
    
    def test_count_distinct_with_distinct_field(self):
        """Test count aggregation with distinct=true."""
        agg = AggregationSpec(agg="count", col="customer_id", distinct=True)
        assert agg.distinct == True
        assert agg.agg == "count"
        assert agg.col == "customer_id"
    
    def test_count_distinct_alias(self):
        """Test count_distinct as aggregation function."""
        agg = AggregationSpec(agg="count_distinct", col="customer_id")
        assert agg.agg == "count_distinct"
        assert agg.col == "customer_id"
    
    def test_reject_distinct_in_column_name(self):
        """Test that DISTINCT in column name is rejected."""
        with pytest.raises(ValueError, match="contains forbidden SQL keyword"):
            AggregationSpec(agg="count", col="DISTINCT customer_id")
    
    def test_reject_spaces_in_column_name(self):
        """Test that spaces in column names are rejected."""
        with pytest.raises(ValueError, match="contains spaces"):
            AggregationSpec(agg="count", col="customer id")
    
    def test_reject_sql_keywords_in_column(self):
        """Test that other SQL keywords are rejected."""
        forbidden = ["select", "from", "where", "join", "union"]
        for keyword in forbidden:
            with pytest.raises(ValueError, match="contains forbidden SQL keyword"):
                AggregationSpec(agg="sum", col=f"amount {keyword}")


class TestSQLGeneration:
    """Test SQL generation with DISTINCT."""
    
    def test_count_distinct_sql(self):
        """Test that COUNT(DISTINCT col) is generated correctly."""
        plan = {
            "original_question": "How many unique customers?",
            "subquestions": [
                {
                    "id": "SQ1",
                    "tables": ["test_1_1"],
                    "aggregations": [
                        {"agg": "count", "col": "customer_id", "distinct": True}
                    ]
                }
            ]
        }
        
        sql, metadata = build_sql(plan["subquestions"][0], plan)
        
        # Check SQL contains COUNT(DISTINCT
        assert "COUNT(DISTINCT" in sql
        # Check SQL does NOT contain quoted DISTINCT
        assert '"DISTINCT' not in sql
        assert 'DISTINCT"' not in sql
        # Check column is properly quoted
        assert '"test_1_1"."customer_id"' in sql
    
    def test_count_distinct_alias_sql(self):
        """Test that count_distinct as agg name works."""
        plan = {
            "original_question": "Unique customers",
            "subquestions": [
                {
                    "id": "SQ1",
                    "tables": ["test_1_1"],
                    "aggregations": [
                        {"agg": "count_distinct", "col": "customer_id"}
                    ]
                }
            ]
        }
        
        sql, metadata = build_sql(plan["subquestions"][0], plan)
        
        # count_distinct should be converted to COUNT with DISTINCT
        assert "COUNT(DISTINCT" in sql
        assert "COUNT_DISTINCT" not in sql
    
    def test_regular_count_without_distinct(self):
        """Test that regular COUNT doesn't add DISTINCT."""
        plan = {
            "original_question": "How many rows?",
            "subquestions": [
                {
                    "id": "SQ1",
                    "tables": ["test_1_1"],
                    "aggregations": [
                        {"agg": "count", "col": "customer_id"}
                    ]
                }
            ]
        }
        
        sql, metadata = build_sql(plan["subquestions"][0], plan)
        
        # Should have COUNT but not DISTINCT
        assert "COUNT(" in sql
        assert "DISTINCT" not in sql


class TestTimeBucketing:
    """Test time bucketing for monthly/yearly aggregations."""
    
    def test_monthly_time_bucket(self):
        """Test monthly time bucket in group_by."""
        plan = {
            "original_question": "Monthly revenue",
            "subquestions": [
                {
                    "id": "SQ1",
                    "tables": ["test_1_1"],
                    "group_by": [
                        {"type": "time_bucket", "grain": "month", "col": "created_at"}
                    ],
                    "aggregations": [
                        {"agg": "sum", "col": "amount"}
                    ]
                }
            ]
        }
        
        sql, metadata = build_sql(plan["subquestions"][0], plan)
        
        # Check for date_trunc
        assert "date_trunc('month'" in sql
        assert '"created_at"' in sql
        # Check GROUP BY
        assert "GROUP BY" in sql
        assert "date_trunc('month'" in sql
        # Check ORDER BY for time series
        assert "ORDER BY" in sql
        assert '"month"' in sql
    
    def test_yearly_time_bucket(self):
        """Test yearly time bucket."""
        plan = {
            "original_question": "Yearly revenue",
            "subquestions": [
                {
                    "id": "SQ1",
                    "tables": ["test_1_1"],
                    "group_by": [
                        {"type": "time_bucket", "grain": "year", "col": "created_at"}
                    ],
                    "aggregations": [
                        {"agg": "sum", "col": "amount"}
                    ]
                }
            ]
        }
        
        sql, metadata = build_sql(plan["subquestions"][0], plan)
        
        assert "date_trunc('year'" in sql
        assert '"year"' in sql or "year" in sql.lower()
    
    def test_mixed_group_by(self):
        """Test combining time bucket with regular column."""
        plan = {
            "original_question": "Monthly revenue by status",
            "subquestions": [
                {
                    "id": "SQ1",
                    "tables": ["test_1_1"],
                    "group_by": [
                        {"type": "time_bucket", "grain": "month", "col": "created_at"},
                        "status"
                    ],
                    "aggregations": [
                        {"agg": "sum", "col": "amount"}
                    ]
                }
            ]
        }
        
        sql, metadata = build_sql(plan["subquestions"][0], plan)
        
        # Check for both group by elements
        assert "date_trunc('month'" in sql
        assert '"status"' in sql
        assert "GROUP BY" in sql


class TestMonthlyUniqueCustomers:
    """Integration test for 'monthly unique customers' query."""
    
    def test_monthly_unique_customers_plan(self):
        """Test that monthly unique customers generates correct plan structure."""
        # This is what the planner should produce
        plan_dict = {
            "original_question": "monthly unique customers",
            "subquestions": [
                {
                    "id": "SQ1",
                    "tables": ["test_1_1"],
                    "group_by": [
                        {"type": "time_bucket", "grain": "month", "col": "created_at"}
                    ],
                    "aggregations": [
                        {"agg": "count", "col": "customer_id", "distinct": True}
                    ]
                }
            ]
        }
        
        # Validate plan
        is_valid, errors = validate_plan(plan_dict)
        assert is_valid, f"Plan validation failed: {errors}"
        
        # Generate SQL
        sql, metadata = build_sql(plan_dict["subquestions"][0], plan_dict)
        
        # Check SQL has all required elements
        assert "date_trunc('month'" in sql
        assert "COUNT(DISTINCT" in sql
        assert '"customer_id"' in sql
        assert "GROUP BY" in sql
        assert "ORDER BY" in sql
        # Ensure no SQL injection
        assert '"DISTINCT' not in sql
        assert 'DISTINCT"' not in sql


class TestComparisonTimeScoping:
    """Test symmetric time constraints for comparisons."""
    
    def test_last_month_vs_this_month(self):
        """Test 'last month vs this month' comparison."""
        plan_dict = {
            "original_question": "revenue last month vs this month",
            "subquestions": [
                {
                    "id": "SQ1_current",
                    "tables": ["test_1_1"],
                    "aggregations": [
                        {"agg": "sum", "col": "amount"}
                    ]
                },
                {
                    "id": "SQ2_comparison",
                    "tables": ["test_1_1"],
                    "aggregations": [
                        {"agg": "sum", "col": "amount"}
                    ]
                }
            ],
            "constraints": [
                {
                    "type": "time",
                    "expression": "test_1_1.created_at in this_month",
                    "applies_to": "SQ1_current"
                },
                {
                    "type": "time",
                    "expression": "test_1_1.created_at in previous_month",
                    "applies_to": "SQ2_comparison"
                }
            ]
        }
        
        # Should validate successfully
        is_valid, errors = validate_plan(plan_dict)
        assert is_valid, f"Comparison plan failed validation: {errors}"
    
    def test_this_year_vs_last_year(self):
        """Test 'this year vs last year' comparison."""
        plan_dict = {
            "original_question": "revenue this year vs last year",
            "subquestions": [
                {
                    "id": "SQ1_current",
                    "tables": ["test_1_1"],
                    "aggregations": [
                        {"agg": "sum", "col": "amount"}
                    ]
                },
                {
                    "id": "SQ2_comparison",
                    "tables": ["test_1_1"],
                    "aggregations": [
                        {"agg": "sum", "col": "amount"}
                    ]
                }
            ],
            "constraints": [
                {
                    "type": "time",
                    "expression": "test_1_1.created_at in this_year",
                    "applies_to": "SQ1_current"
                },
                {
                    "type": "time",
                    "expression": "test_1_1.created_at in previous_year",
                    "applies_to": "SQ2_comparison"
                }
            ]
        }
        
        is_valid, errors = validate_plan(plan_dict)
        assert is_valid, f"Year comparison failed validation: {errors}"
    
    def test_unscoped_time_constraint_fails(self):
        """Test that unscoped time constraints fail for comparisons."""
        plan_dict = {
            "original_question": "revenue this month vs last month",
            "subquestions": [
                {
                    "id": "SQ1_current",
                    "tables": ["test_1_1"],
                    "aggregations": [{"agg": "sum", "col": "amount"}]
                },
                {
                    "id": "SQ2_comparison",
                    "tables": ["test_1_1"],
                    "aggregations": [{"agg": "sum", "col": "amount"}]
                }
            ],
            "constraints": [
                {
                    "type": "time",
                    "expression": "test_1_1.created_at in this_month"
                    # Missing applies_to!
                }
            ]
        }
        
        is_valid, errors = validate_plan(plan_dict)
        assert not is_valid
        assert any("unscoped" in err.lower() for err in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
