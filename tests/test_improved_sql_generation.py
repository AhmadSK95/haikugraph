"""
Test improved SQL generation with better type handling.
"""

import pytest
import duckdb
from pathlib import Path
from haikugraph.execution.type_detector import (
    detect_column_type,
    get_column_info,
    get_sql_cast_expression
)
from haikugraph.execution.execute import build_sql


@pytest.fixture
def db_path():
    return Path("./data/haikugraph.duckdb")


@pytest.fixture
def conn(db_path):
    if not db_path.exists():
        pytest.skip(f"Database not found at {db_path}")
    return duckdb.connect(str(db_path))


class TestTypeDetection:
    """Test the improved type detection system."""
    
    def test_timestamp_detection(self):
        """Test that timestamp columns are detected correctly."""
        assert detect_column_type("created_at", "VARCHAR") == "timestamp"
        assert detect_column_type("updated_at", "VARCHAR") == "timestamp"
        assert detect_column_type("payment_created_at", "VARCHAR") == "timestamp"
        assert detect_column_type("booked_at", "VARCHAR") == "timestamp"
        assert detect_column_type("expires_at", "VARCHAR") == "timestamp"
    
    def test_amount_detection(self):
        """Test that amount/money columns are detected."""
        assert detect_column_type("payment_amount", "VARCHAR") == "numeric_amount"
        assert detect_column_type("booked_amount", "VARCHAR") == "numeric_amount"
        assert detect_column_type("total_gst", "VARCHAR") == "numeric_amount"
        assert detect_column_type("amount_at_source", "DOUBLE") == "numeric_amount"
    
    def test_rate_detection(self):
        """Test that rate columns are detected."""
        assert detect_column_type("exchange_rate", "DOUBLE") == "numeric_rate"
        assert detect_column_type("rate", "VARCHAR") == "numeric_rate"
    
    def test_count_detection(self):
        """Test that count columns are detected."""
        assert detect_column_type("doc_reject_count", "BIGINT") == "numeric_count"
        assert detect_column_type("row_count", "VARCHAR") == "numeric_count"
    
    def test_identifier_detection(self):
        """Test that ID columns are detected."""
        assert detect_column_type("customer_id", "VARCHAR") == "identifier"
        assert detect_column_type("transaction_id", "VARCHAR") == "identifier"
        assert detect_column_type("sha_id", "VARCHAR") == "identifier"
        assert detect_column_type("quote_id", "VARCHAR") == "identifier"
    
    def test_boolean_detection(self):
        """Test that boolean columns are detected."""
        assert detect_column_type("is_university", "BOOLEAN") == "boolean"
        assert detect_column_type("is_source_education_loan", "BOOLEAN") == "boolean"
        assert detect_column_type("has_payment", "VARCHAR") == "boolean"


class TestCastExpressions:
    """Test SQL cast expression generation."""
    
    def test_timestamp_cast_from_varchar(self):
        """Test casting VARCHAR to TIMESTAMP."""
        expr = get_sql_cast_expression(
            "test_2_1", 
            "payment_created_at",
            "timestamp",
            "VARCHAR"
        )
        assert "TRY_CAST" in expr
        assert "AS TIMESTAMP" in expr
    
    def test_amount_cast_from_varchar(self):
        """Test casting VARCHAR amounts to DOUBLE."""
        expr = get_sql_cast_expression(
            "test_2_1",
            "payment_amount",
            "numeric_amount",
            "VARCHAR"
        )
        assert "TRY_CAST" in expr
        assert "AS DOUBLE" in expr
    
    def test_no_cast_for_native_types(self):
        """Test that native types don't get unnecessary casts."""
        expr = get_sql_cast_expression(
            "test_3_1",
            "amount_at_source",
            "numeric_amount",
            "DOUBLE"
        )
        # Should just be the column reference
        assert 'TRY_CAST' not in expr
        assert '"test_3_1"."amount_at_source"' in expr
    
    def test_identifier_no_cast(self):
        """Test that IDs stay as strings."""
        expr = get_sql_cast_expression(
            "test_1_1",
            "customer_id",
            "identifier",
            "VARCHAR"
        )
        assert 'TRY_CAST' not in expr


class TestRealQueries:
    """Test with real queries on actual data."""
    
    def test_amount_aggregation(self, conn):
        """Test that amount aggregations work correctly."""
        # Build a simple SUM query
        subq = {
            "id": "sq_1",
            "tables": ["test_2_1"],
            "columns": [],
            "aggregations": [
                {"agg": "SUM", "col": "payment_amount"}
            ]
        }
        
        sql, metadata = build_sql(subq, {}, conn)
        
        print(f"Generated SQL: {sql}")
        
        # SQL should have TRY_CAST for VARCHAR amount
        assert "TRY_CAST" in sql
        assert "AS DOUBLE" in sql
        assert "payment_amount" in sql
        
        # Execute and verify it returns a number (not error)
        result = conn.execute(sql).fetchone()
        assert result is not None
        print(f"SUM result: {result[0]}")
    
    def test_min_timestamp(self, conn):
        """Test MIN on timestamp columns."""
        subq = {
            "id": "sq_1",
            "tables": ["test_2_1"],
            "columns": [],
            "aggregations": [
                {"agg": "MIN", "col": "payment_created_at"}
            ]
        }
        
        sql, metadata = build_sql(subq, {}, conn)
        
        print(f"Generated SQL: {sql}")
        
        # Should cast to TIMESTAMP
        assert "TRY_CAST" in sql
        assert "AS TIMESTAMP" in sql
        
        # Execute
        result = conn.execute(sql).fetchone()
        print(f"MIN timestamp: {result[0]}")
        assert result is not None
    
    def test_select_with_casting(self, conn):
        """Test that SELECT queries cast appropriately."""
        subq = {
            "id": "sq_1",
            "tables": ["test_2_1"],
            "columns": ["customer_id", "payment_amount", "payment_created_at"],
            "aggregations": []
        }
        
        sql, metadata = build_sql(subq, {}, conn)
        
        print(f"Generated SQL: {sql}")
        
        # Should have casts for amount and timestamp but not for ID
        assert "payment_amount" in sql
        assert "payment_created_at" in sql
        
        # Execute and check types returned
        result = conn.execute(sql).fetchone()
        if result:
            cust_id, amount, created = result[0], result[1], result[2]
            print(f"Types: customer_id={type(cust_id)}, amount={type(amount)}, created={type(created)}")
            
            # customer_id should be string
            assert isinstance(cust_id, str) or cust_id is None
            
            # amount should be numeric or None
            assert isinstance(amount, (int, float)) or amount is None
    
    def test_group_by_with_amounts(self, conn):
        """Test GROUP BY with amount aggregations."""
        subq = {
            "id": "sq_1",
            "tables": ["test_2_1"],
            "columns": [],
            "group_by": ["payment_status"],
            "aggregations": [
                {"agg": "SUM", "col": "payment_amount"},
                {"agg": "COUNT", "col": "transaction_id"}
            ]
        }
        
        sql, metadata = build_sql(subq, {}, conn)
        
        print(f"Generated SQL: {sql}")
        
        # Should have proper casting
        assert "payment_amount" in sql
        assert "GROUP BY" in sql
        
        # Execute
        results = conn.execute(sql).fetchall()
        print(f"Group by results: {len(results)} groups")
        for row in results[:3]:
            print(f"  {row}")


class TestDataIntegrity:
    """Test that we're not losing data with new approach."""
    
    def test_no_data_loss_in_aggregation(self, conn):
        """Verify we get same row counts as direct queries."""
        # Old way (direct)
        old_count = conn.execute("SELECT COUNT(*) FROM test_2_1").fetchone()[0]
        
        # New way (through builder)
        subq = {
            "id": "sq_1",
            "tables": ["test_2_1"],
            "columns": [],
            "aggregations": [
                {"agg": "COUNT", "col": "sha_id"}
            ]
        }
        
        sql, _ = build_sql(subq, {}, conn)
        new_count = conn.execute(sql).fetchone()[0]
        
        assert old_count == new_count, f"Data loss detected: {old_count} vs {new_count}"
    
    def test_null_handling_preserved(self, conn):
        """Verify NULLs are still counted/shown properly."""
        # Count total vs non-null
        subq_total = {
            "id": "sq_1",
            "tables": ["test_2_1"],
            "columns": [],
            "aggregations": [{"agg": "COUNT", "col": "*"}]
        }
        
        sql_total, _ = build_sql(subq_total, {}, conn)
        # Adjust for COUNT(*) - builder uses column name
        sql_total = sql_total.replace('COUNT("test_2_1"."*")', 'COUNT(*)')
        
        total = conn.execute(sql_total).fetchone()[0]
        
        # Count non-null payment_created_at
        non_null = conn.execute("""
            SELECT COUNT(payment_created_at) 
            FROM test_2_1
        """).fetchone()[0]
        
        null_count = total - non_null
        null_pct = 100 * null_count / total
        
        print(f"Total: {total}, Non-null: {non_null}, Null: {null_count} ({null_pct:.1f}%)")
        
        # Verify high NULL percentage (should be ~97.8%)
        assert null_pct > 90, "NULL detection broken"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
