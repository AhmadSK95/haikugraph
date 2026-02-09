"""
Test various question patterns and edge cases.

Tests for:
- Oldest/newest queries with timestamps
- Time grouping with NULL values  
- MIN/MAX aggregations
- Data quality issues
"""

import pytest
import duckdb
from pathlib import Path


@pytest.fixture
def db_path():
    """Get path to test database."""
    return Path("./data/haikugraph.duckdb")


@pytest.fixture
def conn(db_path):
    """Create database connection."""
    if not db_path.exists():
        pytest.skip(f"Database not found at {db_path}")
    return duckdb.connect(str(db_path))


class TestTimestampQueries:
    """Test MIN/MAX and timestamp-related queries."""
    
    def test_min_timestamp_with_nulls(self, conn):
        """Test that MIN works correctly even with NULL timestamps."""
        result = conn.execute("""
            SELECT MIN(TRY_CAST(payment_created_at AS TIMESTAMP)) as min_date
            FROM test_2_1
        """).fetchone()
        
        assert result is not None
        assert result[0] is not None, "MIN should ignore NULLs and return a value"
        print(f"Min timestamp: {result[0]}")
    
    def test_max_timestamp_with_nulls(self, conn):
        """Test that MAX works correctly even with NULL timestamps."""
        result = conn.execute("""
            SELECT MAX(TRY_CAST(payment_created_at AS TIMESTAMP)) as max_date
            FROM test_2_1
        """).fetchone()
        
        assert result is not None
        assert result[0] is not None, "MAX should ignore NULLs and return a value"
        print(f"Max timestamp: {result[0]}")
    
    def test_count_null_timestamps(self, conn):
        """Test counting NULL timestamps."""
        result = conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(payment_created_at) as non_null,
                COUNT(*) - COUNT(payment_created_at) as nulls
            FROM test_2_1
        """).fetchone()
        
        total, non_null, nulls = result
        print(f"Total: {total}, Non-null: {non_null}, Nulls: {nulls}")
        
        assert total > 0
        if nulls > 0:
            print(f"Warning: {nulls} ({100*nulls/total:.1f}%) records have NULL payment_created_at")


class TestTimeGrouping:
    """Test queries that group by time periods."""
    
    def test_month_grouping_with_nulls(self, conn):
        """Test that EXTRACT(MONTH) handles NULLs properly."""
        result = conn.execute("""
            SELECT 
                EXTRACT(MONTH FROM TRY_CAST(payment_created_at AS TIMESTAMP)) as month,
                COUNT(*) as count
            FROM test_2_1
            GROUP BY month
            ORDER BY month NULLS LAST
        """).fetchall()
        
        # Should have results
        assert len(result) > 0
        
        # Check for NULL month
        null_month_count = sum(row[1] for row in result if row[0] is None)
        if null_month_count > 0:
            print(f"Warning: {null_month_count} records grouped to NULL month")
            
        # Show distribution
        for month, count in result[:12]:
            print(f"Month {month}: {count} records")
    
    def test_month_grouping_exclude_nulls(self, conn):
        """Test that we can exclude NULL months."""
        result = conn.execute("""
            SELECT 
                EXTRACT(MONTH FROM TRY_CAST(payment_created_at AS TIMESTAMP)) as month,
                COUNT(*) as count
            FROM test_2_1
            WHERE TRY_CAST(payment_created_at AS TIMESTAMP) IS NOT NULL
            GROUP BY month
            ORDER BY month
        """).fetchall()
        
        # Should have results
        assert len(result) > 0
        
        # No NULL months
        assert all(row[0] is not None for row in result), "Should not have NULL months"
        
        print(f"Found {len(result)} months with valid data")


class TestDataQuality:
    """Test data quality and identify issues."""
    
    def test_timestamp_format_consistency(self, conn):
        """Check if timestamps are consistently formatted."""
        result = conn.execute("""
            SELECT 
                payment_created_at,
                TRY_CAST(payment_created_at AS TIMESTAMP) as parsed,
                CASE 
                    WHEN payment_created_at IS NULL THEN 'null'
                    WHEN TRY_CAST(payment_created_at AS TIMESTAMP) IS NULL THEN 'invalid'
                    ELSE 'valid'
                END as status
            FROM test_2_1
            LIMIT 100
        """).fetchall()
        
        null_count = sum(1 for row in result if row[2] == 'null')
        invalid_count = sum(1 for row in result if row[2] == 'invalid')
        valid_count = sum(1 for row in result if row[2] == 'valid')
        
        print(f"Sample of 100: Valid={valid_count}, Invalid={invalid_count}, Null={null_count}")
        
        if invalid_count > 0:
            # Show some invalid examples
            invalid_examples = [row[0] for row in result if row[2] == 'invalid'][:5]
            print(f"Invalid timestamp examples: {invalid_examples}")
    
    def test_recommend_filtering_nulls(self, conn):
        """Recommend whether to filter NULL timestamps in queries."""
        result = conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN TRY_CAST(payment_created_at AS TIMESTAMP) IS NOT NULL THEN 1 END) as valid,
                COUNT(CASE WHEN TRY_CAST(payment_created_at AS TIMESTAMP) IS NULL THEN 1 END) as invalid
            FROM test_2_1
        """).fetchone()
        
        total, valid, invalid = result
        invalid_pct = 100 * invalid / total if total > 0 else 0
        
        print(f"Data quality: {valid}/{total} ({100*valid/total:.1f}%) valid timestamps")
        print(f"Recommendation: {'FILTER NULL timestamps' if invalid_pct > 10 else 'Keep all data'}")
        
        assert valid > 0, "Should have at least some valid timestamps"


class TestAggregationPatterns:
    """Test various aggregation patterns."""
    
    def test_oldest_transaction(self, conn):
        """Test finding the oldest transaction."""
        # This is what the UI query should do
        result = conn.execute("""
            SELECT MIN(TRY_CAST(payment_created_at AS TIMESTAMP)) as oldest_date
            FROM test_2_1
            WHERE TRY_CAST(payment_created_at AS TIMESTAMP) IS NOT NULL
        """).fetchone()
        
        assert result is not None
        assert result[0] is not None
        print(f"Oldest transaction: {result[0]}")
    
    def test_transaction_count_by_month(self, conn):
        """Test counting transactions by month."""
        result = conn.execute("""
            SELECT 
                EXTRACT(MONTH FROM TRY_CAST(payment_created_at AS TIMESTAMP)) as month,
                EXTRACT(YEAR FROM TRY_CAST(payment_created_at AS TIMESTAMP)) as year,
                COUNT(*) as count
            FROM test_2_1
            WHERE TRY_CAST(payment_created_at AS TIMESTAMP) IS NOT NULL
            GROUP BY year, month
            ORDER BY year, month
        """).fetchall()
        
        assert len(result) > 0
        print(f"Transactions across {len(result)} month/year combinations")
        
        for year, month, count in result[:10]:
            print(f"{year}-{month:02d}: {count} transactions")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
