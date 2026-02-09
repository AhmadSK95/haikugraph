"""
Improved type detection for SQL generation.

Analyzes column names and sample data to determine semantic types
and generate proper SQL casts.
"""

from typing import Optional, Literal
import re


TypeHint = Literal[
    "timestamp",
    "date", 
    "time",
    "numeric_amount",  # money/currency
    "numeric_rate",    # percentages, exchange rates
    "numeric_count",   # integers like counts, IDs
    "numeric_decimal", # general decimal numbers
    "boolean",
    "text",
    "identifier",      # IDs, codes, hashes
    "unknown"
]


def detect_column_type(
    column_name: str,
    declared_type: str,
    sample_values: list = None
) -> TypeHint:
    """
    Detect semantic type of a column based on name and optional sample data.
    
    Args:
        column_name: Name of the column
        declared_type: SQL type from schema (e.g., "VARCHAR", "DOUBLE")
        sample_values: Optional list of sample values for validation
        
    Returns:
        TypeHint indicating the semantic type
    """
    col_lower = column_name.lower()
    type_upper = declared_type.upper()
    
    # Timestamp/Date detection (highest priority for time-based queries)
    timestamp_patterns = [
        r'created_at$', r'updated_at$', r'deleted_at$',
        r'_at$',  # generic *_at suffix
        r'^timestamp', r'_timestamp',
        r'booked_at', r'expires_at', r'initiated_at', r'locked_dt',
        r'date$', r'_date$',
        r'time$', r'_time$',
    ]
    
    for pattern in timestamp_patterns:
        if re.search(pattern, col_lower):
            return "timestamp"
    
    # Boolean detection
    if 'BOOLEAN' in type_upper or 'BOOL' in type_upper:
        return "boolean"
    
    if col_lower.startswith('is_') or col_lower.startswith('has_'):
        return "boolean"
    
    # Numeric amount detection (money/currency)
    amount_patterns = [
        r'amount', r'payment', r'price', r'cost', r'fee',
        r'charge', r'total', r'revenue', r'balance',
        r'tcs$', r'gst$', r'markup'
    ]
    
    for pattern in amount_patterns:
        if pattern in col_lower:
            return "numeric_amount"
    
    # Numeric rate detection (percentages, ratios)
    rate_patterns = [
        r'rate$', r'_rate$', r'^rate$', r'exchange_rate',
        r'percentage', r'ratio'
    ]
    
    for pattern in rate_patterns:
        if re.search(pattern, col_lower):
            return "numeric_rate"
    
    # Numeric count detection (integers)
    count_patterns = [
        r'count$', r'_count$', r'^count$', r'quantity', r'qty',
        r'number_of', r'num_', r'deal_id$'
    ]
    
    for pattern in count_patterns:
        if re.search(pattern, col_lower):
            return "numeric_count"
    
    # Identifier detection (IDs, codes, hashes)
    if any(term in col_lower for term in ['_id', 'id$', 'sha_', 'code$', 'reference']):
        # But not if it's a count-like ID
        if 'count' not in col_lower:
            return "identifier"
    
    # Fallback to declared SQL type
    if any(t in type_upper for t in ['INT', 'BIGINT', 'SMALLINT']):
        return "numeric_count"
    
    if any(t in type_upper for t in ['DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC']):
        return "numeric_decimal"
    
    if any(t in type_upper for t in ['VARCHAR', 'TEXT', 'CHAR']):
        return "text"
    
    return "unknown"


def get_sql_cast_expression(
    table: str,
    column: str,
    column_type: TypeHint,
    declared_type: str
) -> str:
    """
    Generate appropriate SQL cast expression for a column.
    
    Args:
        table: Table name
        column: Column name  
        column_type: Detected semantic type
        declared_type: SQL type from schema
        
    Returns:
        SQL expression with appropriate casting
    """
    col_ref = f'"{table}"."{column}"'
    type_upper = declared_type.upper()
    
    # If already the right type, no cast needed
    if column_type == "timestamp":
        if any(t in type_upper for t in ['TIMESTAMP', 'DATE']):
            return col_ref
        else:
            # VARCHAR → TIMESTAMP (common case)
            return f'TRY_CAST({col_ref} AS TIMESTAMP)'
    
    elif column_type == "boolean":
        if 'BOOLEAN' in type_upper or 'BOOL' in type_upper:
            return col_ref
        else:
            return f'TRY_CAST({col_ref} AS BOOLEAN)'
    
    elif column_type in ("numeric_amount", "numeric_rate", "numeric_decimal"):
        if any(t in type_upper for t in ['DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC']):
            return col_ref
        else:
            # VARCHAR → DOUBLE (common case for amounts stored as strings)
            return f'TRY_CAST({col_ref} AS DOUBLE)'
    
    elif column_type == "numeric_count":
        if any(t in type_upper for t in ['INT', 'BIGINT', 'SMALLINT']):
            return col_ref
        else:
            # Try BIGINT for counts (handles larger numbers)
            return f'TRY_CAST({col_ref} AS BIGINT)'
    
    elif column_type == "identifier":
        # IDs should stay as strings
        return col_ref
    
    elif column_type == "text":
        # Already text
        return col_ref
    
    else:
        # Unknown - leave as-is
        return col_ref


def get_column_info(conn, table: str, column: str) -> tuple[str, TypeHint]:
    """
    Get column type info from database.
    
    Args:
        conn: DuckDB connection
        table: Table name
        column: Column name
        
    Returns:
        Tuple of (declared_sql_type, detected_semantic_type)
    """
    try:
        # Query schema
        type_query = f"""
            SELECT data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table}' AND column_name = '{column}'
        """
        result = conn.execute(type_query).fetchone()
        
        if result:
            declared_type = result[0]
            semantic_type = detect_column_type(column, declared_type)
            return declared_type, semantic_type
        else:
            return "VARCHAR", "unknown"
            
    except Exception:
        return "VARCHAR", "unknown"


def should_filter_nulls(
    column_type: TypeHint,
    aggregation: Optional[str] = None
) -> bool:
    """
    Determine if NULL values should be filtered for a query.
    
    Args:
        column_type: Semantic type of column
        aggregation: Aggregation function (MIN, MAX, etc.)
        
    Returns:
        True if NULLs should be filtered in WHERE clause
    """
    # For time-based aggregations on high-NULL columns, 
    # we DON'T filter - we show the NULL bucket
    # This is per user request: "don't filter, show in bucket"
    
    # However, for MIN/MAX on timestamps with >90% NULLs,
    # we might want to filter to show meaningful results
    # But user wants to see data quality issues
    
    # Decision: NEVER auto-filter NULLs, always show them
    return False


def get_suggested_filter_clause(
    table: str,
    column: str, 
    column_type: TypeHint,
    null_percentage: float
) -> Optional[str]:
    """
    Suggest a WHERE clause to filter NULLs if beneficial.
    
    Args:
        table: Table name
        column: Column name
        column_type: Semantic type
        null_percentage: Percentage of NULL values (0-100)
        
    Returns:
        Optional WHERE clause suggestion (or None)
    """
    # Only suggest filtering if >50% NULLs and it's a timestamp/amount
    if null_percentage > 50 and column_type in ("timestamp", "numeric_amount"):
        cast_expr = get_sql_cast_expression(table, column, column_type, "VARCHAR")
        return f"{cast_expr} IS NOT NULL"
    
    return None
