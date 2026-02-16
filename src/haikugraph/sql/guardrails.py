"""SQL guardrails for safe query execution.

This module provides validation and sanitization for SQL queries
to ensure read-only, safe execution against user databases.

Safety Features:
- SELECT-only validation (block all write operations)
- Dangerous keyword blocking (DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, GRANT, REVOKE)
- LIMIT enforcement (default 1000, configurable max)
- Query timeout preparation
- Prompt injection prevention patterns
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple


class ValidationResult(NamedTuple):
    """Result of SQL validation."""
    
    is_valid: bool
    error: str | None = None
    warnings: list[str] | None = None


class DangerLevel(str, Enum):
    """Classification of SQL danger level."""
    
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass
class GuardrailConfig:
    """Configuration for SQL guardrails."""
    
    # Limits
    default_limit: int = 1000
    max_limit: int = 10000
    max_result_rows: int = 50000
    query_timeout_seconds: int = 30
    
    # Blocked keywords (case-insensitive, word boundaries)
    blocked_keywords: tuple[str, ...] = (
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
        "CREATE",
        "REPLACE",
        "EXECUTE",
        "EXEC",
        "CALL",
        "SET",  # Block SET statements
        "PRAGMA",  # Block PRAGMA for SQLite
        "ATTACH",  # Block ATTACH DATABASE
        "DETACH",
        "COPY",  # Block COPY for Postgres
        "LOAD",  # Block LOAD DATA
    )
    
    # Additional patterns to block (regex)
    blocked_patterns: tuple[str, ...] = (
        r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)",  # Piggyback attacks
        r"--\s*$",  # SQL comment at end (potential injection)
        r"/\*.*\*/",  # Block comments (potential injection hiding)
        r"UNION\s+ALL\s+SELECT",  # UNION injection
        r"INTO\s+OUTFILE",  # File write
        r"INTO\s+DUMPFILE",  # File dump
        r"LOAD_FILE\s*\(",  # File read
        r"xp_\w+",  # SQL Server extended procedures
        r"sp_\w+",  # SQL Server stored procedures
    )
    
    # Allowed statement prefixes (case-insensitive)
    allowed_prefixes: tuple[str, ...] = (
        "SELECT",
        "WITH",  # CTEs are allowed
        "EXPLAIN",  # EXPLAIN is read-only
    )


# Default configuration
DEFAULT_CONFIG = GuardrailConfig()


def validate_sql(sql: str, config: GuardrailConfig | None = None) -> ValidationResult:
    """Validate SQL query against safety rules.
    
    Args:
        sql: SQL query string to validate
        config: Optional guardrail configuration
    
    Returns:
        ValidationResult with is_valid flag and optional error message
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if not sql or not sql.strip():
        return ValidationResult(is_valid=False, error="Empty SQL query")
    
    # Normalize for checking
    sql_clean = sql.strip()
    sql_upper = sql_clean.upper()
    
    # Check for allowed prefixes
    has_valid_prefix = False
    for prefix in config.allowed_prefixes:
        if sql_upper.startswith(prefix):
            has_valid_prefix = True
            break
    
    if not has_valid_prefix:
        return ValidationResult(
            is_valid=False,
            error=f"Query must start with one of: {', '.join(config.allowed_prefixes)}"
        )
    
    # Check for blocked keywords
    blocked = detect_dangerous_keywords(sql, config)
    if blocked:
        return ValidationResult(
            is_valid=False,
            error=f"Blocked keyword(s) detected: {', '.join(blocked)}"
        )
    
    # Check for blocked patterns
    pattern_match = detect_dangerous_patterns(sql, config)
    if pattern_match:
        return ValidationResult(
            is_valid=False,
            error=f"Dangerous pattern detected: {pattern_match}"
        )
    
    # Collect warnings
    warnings = []
    
    # Warn about missing LIMIT
    if not _has_limit_clause(sql):
        warnings.append("Query has no LIMIT clause; one will be enforced")
    
    # Warn about potential Cartesian products
    if _might_have_cartesian_product(sql):
        warnings.append("Query may produce Cartesian product; check JOIN conditions")
    
    return ValidationResult(is_valid=True, warnings=warnings if warnings else None)


def detect_dangerous_keywords(sql: str, config: GuardrailConfig | None = None) -> list[str]:
    """Detect blocked keywords in SQL query.
    
    Args:
        sql: SQL query string
        config: Optional guardrail configuration
    
    Returns:
        List of detected blocked keywords (empty if none)
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    found = []
    sql_upper = sql.upper()
    
    for keyword in config.blocked_keywords:
        # Use word boundary matching to avoid false positives
        # e.g., "UPDATED_AT" should not match "UPDATE"
        pattern = rf"\b{keyword}\b"
        if re.search(pattern, sql_upper):
            found.append(keyword)
    
    return found


def detect_dangerous_patterns(sql: str, config: GuardrailConfig | None = None) -> str | None:
    """Detect dangerous patterns in SQL query.
    
    Args:
        sql: SQL query string
        config: Optional guardrail configuration
    
    Returns:
        Description of matched pattern, or None if safe
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    for pattern in config.blocked_patterns:
        if re.search(pattern, sql, re.IGNORECASE | re.DOTALL):
            return f"Pattern: {pattern}"
    
    # Check for multiple statements (semicolon not at end)
    # Remove string literals first to avoid false positives
    sql_no_strings = _remove_string_literals(sql)
    semicolons = [m.start() for m in re.finditer(r";", sql_no_strings)]
    
    if len(semicolons) > 1:
        return "Multiple statements detected (only single SELECT allowed)"
    
    if semicolons and semicolons[0] < len(sql_no_strings.strip()) - 1:
        # Semicolon not at end
        remaining = sql_no_strings[semicolons[0] + 1:].strip()
        if remaining and not remaining.startswith("--"):
            return "Multiple statements detected (only single SELECT allowed)"
    
    return None


def enforce_limit(sql: str, config: GuardrailConfig | None = None) -> str:
    """Enforce LIMIT clause on SQL query.
    
    If query has no LIMIT, add the default limit.
    If query has LIMIT exceeding max, reduce to max.
    
    Args:
        sql: SQL query string
        config: Optional guardrail configuration
    
    Returns:
        SQL with LIMIT enforced
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    sql_clean = sql.strip().rstrip(";")
    sql_upper = sql_clean.upper()
    
    # Check for existing LIMIT clause
    limit_match = re.search(
        r"\bLIMIT\s+(\d+)\s*(?:OFFSET\s+\d+)?(?:\s*;?\s*)$",
        sql_upper
    )
    
    if limit_match:
        current_limit = int(limit_match.group(1))
        if current_limit > config.max_limit:
            # Replace with max limit
            new_sql = re.sub(
                r"\bLIMIT\s+\d+",
                f"LIMIT {config.max_limit}",
                sql_clean,
                flags=re.IGNORECASE
            )
            return new_sql
        return sql_clean
    
    # No LIMIT found, add default
    # Check if there's ORDER BY to place LIMIT after it
    if re.search(r"\bORDER\s+BY\b", sql_upper):
        return f"{sql_clean} LIMIT {config.default_limit}"
    
    # Check for GROUP BY without ORDER BY
    if re.search(r"\bGROUP\s+BY\b", sql_upper):
        return f"{sql_clean} LIMIT {config.default_limit}"
    
    # Simple query
    return f"{sql_clean} LIMIT {config.default_limit}"


def sanitize_for_prompt_injection(text: str) -> str:
    """Sanitize text from database to prevent prompt injection.
    
    Database content should never be trusted and must be sanitized
    before including in LLM prompts.
    
    Args:
        text: Raw text from database
    
    Returns:
        Sanitized text safe for LLM prompts
    """
    if not text:
        return ""
    
    # Remove common prompt injection patterns
    patterns_to_remove = [
        r"ignore\s+(previous|all|above)\s+instructions?",
        r"disregard\s+(previous|all|above)\s+instructions?",
        r"forget\s+(previous|all|above)\s+instructions?",
        r"new\s+instructions?:",
        r"system\s*:",
        r"assistant\s*:",
        r"user\s*:",
        r"\[INST\]",
        r"\[/INST\]",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"<<SYS>>",
        r"<</SYS>>",
    ]
    
    result = text
    for pattern in patterns_to_remove:
        result = re.sub(pattern, "[FILTERED]", result, flags=re.IGNORECASE)
    
    # Truncate very long text
    max_len = 1000
    if len(result) > max_len:
        result = result[:max_len] + "... [truncated]"
    
    return result


def classify_danger_level(sql: str, config: GuardrailConfig | None = None) -> DangerLevel:
    """Classify the danger level of a SQL query.
    
    Args:
        sql: SQL query string
        config: Optional guardrail configuration
    
    Returns:
        DangerLevel classification
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    validation = validate_sql(sql, config)
    
    if not validation.is_valid:
        return DangerLevel.BLOCKED
    
    if validation.warnings:
        return DangerLevel.WARNING
    
    return DangerLevel.SAFE


def _has_limit_clause(sql: str) -> bool:
    """Check if SQL has a LIMIT clause."""
    return bool(re.search(r"\bLIMIT\s+\d+", sql, re.IGNORECASE))


def _might_have_cartesian_product(sql: str) -> bool:
    """Heuristic check for potential Cartesian product.
    
    Returns True if query has multiple tables in FROM without
    explicit JOIN conditions or WHERE clause with column equality.
    """
    sql_upper = sql.upper()
    
    # Count comma-separated tables in FROM (old-style join)
    from_match = re.search(r"\bFROM\s+([^;]+?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|$)", sql_upper)
    if from_match:
        from_clause = from_match.group(1)
        # Count commas not inside parentheses
        depth = 0
        comma_count = 0
        for char in from_clause:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                comma_count += 1
        
        # Multiple tables without JOIN keyword
        if comma_count > 0 and "JOIN" not in from_clause:
            # Check for WHERE clause with join condition
            where_match = re.search(r"\bWHERE\b.+", sql_upper)
            if not where_match or "=" not in where_match.group():
                return True
    
    return False


def _remove_string_literals(sql: str) -> str:
    """Remove string literals from SQL for safer pattern matching.
    
    Replaces 'string' and "string" with empty strings.
    """
    # Remove single-quoted strings (including escaped quotes)
    sql = re.sub(r"'([^']|'')*'", "''", sql)
    # Remove double-quoted identifiers
    sql = re.sub(r'"([^"]|"")*"', '""', sql)
    return sql


def get_query_stats(sql: str) -> dict:
    """Extract statistics about a SQL query for analysis.
    
    Args:
        sql: SQL query string
    
    Returns:
        Dictionary with query statistics
    """
    sql_upper = sql.upper()
    
    # Count JOINs
    join_count = len(re.findall(r"\bJOIN\b", sql_upper))
    
    # Count tables in FROM
    from_match = re.search(r"\bFROM\s+([^;]+?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|$)", sql_upper)
    table_count = 1
    if from_match:
        from_clause = from_match.group(1)
        table_count = from_clause.count(",") + 1 + join_count
    
    # Check for aggregations
    has_aggregation = bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", sql_upper))
    
    # Check for subqueries
    has_subquery = sql_upper.count("SELECT") > 1
    
    # Check for GROUP BY
    has_group_by = bool(re.search(r"\bGROUP\s+BY\b", sql_upper))
    
    # Extract LIMIT if present
    limit_match = re.search(r"\bLIMIT\s+(\d+)", sql_upper)
    limit_value = int(limit_match.group(1)) if limit_match else None
    
    return {
        "join_count": join_count,
        "table_count": table_count,
        "has_aggregation": has_aggregation,
        "has_subquery": has_subquery,
        "has_group_by": has_group_by,
        "limit_value": limit_value,
        "query_length": len(sql),
    }
