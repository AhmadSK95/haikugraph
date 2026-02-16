"""SchemaAgent for schema introspection and semantic catalog building.

This agent:
1. Uses SemanticProfiler for deep database understanding
2. Introspects database schema using information_schema
3. Profiles columns (null rates, distinct counts, sample values)
4. Identifies relevant tables/columns for the user's goal using SEMANTIC MAPPING
5. Infers join relationships

KEY IMPROVEMENT: Uses entity-to-table mapping to correctly identify
which table to query (e.g., 'transaction' queries → test_2_1)
"""

import re
from pathlib import Path
from typing import Any

import duckdb

from haikugraph.agents.base import BaseAgent, AgentError
from haikugraph.agents.contracts import (
    AgentStatus,
    ColumnProfile,
    IntakeResult,
    JoinEdge,
    SchemaResult,
    TableProfile,
)
from haikugraph.sql.guardrails import sanitize_for_prompt_injection

# Try to import semantic profiler
try:
    from haikugraph.knowledge.semantic_profiler import SemanticProfiler, DatabaseSemantic
    from haikugraph.sources.registry import DataSourceRegistry
    from haikugraph.knowledge.store import KnowledgeStore
    HAS_SEMANTIC_PROFILER = True
except ImportError:
    HAS_SEMANTIC_PROFILER = False
    SemanticProfiler = None
    DatabaseSemantic = None


# Entity keywords for semantic matching
ENTITY_KEYWORDS = {
    'transaction': ['transaction', 'transactions', 'payment', 'payments', 'mt103', 'transfer'],
    'customer': ['customer', 'customers', 'client', 'user', 'beneficiary', 'payee'],
    'order': ['order', 'orders', 'purchase', 'sale'],
    'product': ['product', 'products', 'item', 'sku'],
    'invoice': ['invoice', 'invoices', 'bill', 'receipt'],
}


class SchemaAgent(BaseAgent[SchemaResult]):
    """Agent for schema introspection and semantic catalog building.
    
    This agent analyzes the database schema to identify relevant
    tables and columns for the user's query, and infers join paths.
    
    KEY IMPROVEMENT: Uses SemanticProfiler to understand entity-to-table
    mappings (e.g., 'transaction' → test_2_1) instead of guessing.
    """
    
    name = "schema_agent"
    
    def __init__(
        self,
        db_path: Path | str,
        *,
        max_sample_rows: int = 5,
        max_tables: int = 50,
        data_dir: Path | str | None = None,
    ):
        """Initialize schema agent.
        
        Args:
            db_path: Path to DuckDB database
            max_sample_rows: Max rows to sample per column
            max_tables: Max tables to profile
            data_dir: Data directory for semantic profiling (auto-detected if not provided)
        """
        super().__init__()
        self.db_path = Path(db_path)
        self.max_sample_rows = max_sample_rows
        self.max_tables = max_tables
        
        # Initialize semantic profiler for deep understanding
        self._semantic: DatabaseSemantic | None = None
        self._profiler: SemanticProfiler | None = None
        
        if HAS_SEMANTIC_PROFILER:
            # Auto-detect data directory
            if data_dir:
                self._data_dir = Path(data_dir)
            else:
                # Try common locations
                for candidate in [self.db_path.parent / "data", self.db_path.parent, Path("./data")]:
                    if candidate.exists() and any(candidate.glob("*.xlsx")):
                        self._data_dir = candidate
                        break
                else:
                    self._data_dir = self.db_path.parent
            
            self._init_semantic_profiler()
    
    def _init_semantic_profiler(self) -> None:
        """Initialize the semantic profiler for deep database understanding."""
        if not HAS_SEMANTIC_PROFILER:
            return
        
        try:
            knowledge_dir = self._data_dir / ".knowledge"
            registry = DataSourceRegistry(storage_path=knowledge_dir / "sources.json")
            knowledge = KnowledgeStore(knowledge_dir)
            
            # Auto-discover data sources if needed
            if not registry.list():
                registry.auto_discover(self._data_dir)
            
            if registry.list():
                self._profiler = SemanticProfiler(
                    registry=registry,
                    knowledge=knowledge,
                    cache_dir=knowledge_dir,
                )
                self._semantic = self._profiler.profile()
                print(f"📚 SchemaAgent: Loaded semantic profile with {len(self._semantic.entity_to_table)} entity mappings")
        except Exception as e:
            print(f"⚠️ SchemaAgent: Could not initialize semantic profiler: {e}")
    
    @property
    def output_schema(self) -> type[SchemaResult]:
        return SchemaResult
    
    def run(
        self,
        intake_result: IntakeResult,
        *,
        force_refresh: bool = False,
    ) -> SchemaResult:
        """Introspect schema and build semantic catalog.
        
        Args:
            intake_result: Result from IntakeAgent
            force_refresh: Force refresh even if cached
        
        Returns:
            SchemaResult with table profiles and join graph
        """
        self._start_timer()
        
        try:
            conn = duckdb.connect(str(self.db_path), read_only=True)
            
            # Get all tables
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            tables = [row[0] for row in conn.execute(tables_query).fetchall()]
            tables = tables[:self.max_tables]
            
            # Profile each table
            table_profiles = []
            for table_name in tables:
                profile = self._profile_table(conn, table_name)
                table_profiles.append(profile)
            
            # Infer join relationships
            join_graph = self._infer_joins(table_profiles)
            
            # Identify relevant tables and columns based on intake
            relevant_tables, relevant_columns = self._find_relevant_schema(
                intake_result, table_profiles
            )
            
            # Suggest metrics and dimensions
            suggested_metrics = self._suggest_metrics(table_profiles, relevant_tables)
            suggested_dimensions = self._suggest_dimensions(table_profiles, relevant_tables)
            suggested_time_col = self._suggest_time_column(table_profiles, relevant_tables)
            
            conn.close()
            
            elapsed = self._stop_timer()
            
            return SchemaResult(
                status=AgentStatus.SUCCESS,
                tables=table_profiles,
                join_graph=join_graph,
                relevant_tables=relevant_tables,
                relevant_columns=relevant_columns,
                suggested_metrics=suggested_metrics,
                suggested_dimensions=suggested_dimensions,
                suggested_time_column=suggested_time_col,
                warnings=[],
                confidence=0.8,
                reasoning=f"Profiled {len(table_profiles)} tables, found {len(relevant_tables)} relevant",
                processing_time_ms=elapsed,
            )
        
        except Exception as e:
            elapsed = self._stop_timer()
            return SchemaResult(
                status=AgentStatus.FAILED,
                tables=[],
                join_graph=[],
                relevant_tables=[],
                relevant_columns=[],
                suggested_metrics=[],
                suggested_dimensions=[],
                warnings=[str(e)],
                confidence=0.0,
                reasoning=f"Schema introspection failed: {str(e)}",
                processing_time_ms=elapsed,
            )
    
    def _profile_table(self, conn: duckdb.DuckDBPyConnection, table_name: str) -> TableProfile:
        """Profile a single table."""
        # Get row count
        row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        
        # Get columns
        columns_query = f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """
        columns_info = conn.execute(columns_query).fetchall()
        
        column_profiles = []
        primary_keys = []
        timestamp_cols = []
        metric_cols = []
        
        for col_name, col_type in columns_info:
            profile = self._profile_column(conn, table_name, col_name, col_type, row_count)
            column_profiles.append(profile)
            
            if profile.is_likely_key:
                primary_keys.append(col_name)
            if profile.semantic_type == "timestamp":
                timestamp_cols.append(col_name)
            if profile.is_likely_metric:
                metric_cols.append(col_name)
        
        return TableProfile(
            name=table_name,
            row_count=row_count,
            columns=column_profiles,
            primary_key_columns=primary_keys,
            timestamp_columns=timestamp_cols,
            metric_columns=metric_cols,
        )
    
    def _profile_column(
        self,
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        col_name: str,
        col_type: str,
        row_count: int,
    ) -> ColumnProfile:
        """Profile a single column."""
        # Get null count and distinct count
        stats_query = f'''
            SELECT 
                COUNT(*) - COUNT("{col_name}") as null_count,
                COUNT(DISTINCT "{col_name}") as distinct_count
            FROM "{table_name}"
        '''
        stats = conn.execute(stats_query).fetchone()
        null_count = stats[0]
        distinct_count = stats[1]
        null_rate = null_count / row_count if row_count > 0 else 0.0
        
        # Get sample values
        sample_query = f'''
            SELECT DISTINCT "{col_name}"
            FROM "{table_name}"
            WHERE "{col_name}" IS NOT NULL
            LIMIT {self.max_sample_rows}
        '''
        samples = conn.execute(sample_query).fetchall()
        sample_values = [sanitize_for_prompt_injection(str(s[0]))[:100] for s in samples]
        
        # Get min/max for numeric and date types
        min_val = None
        max_val = None
        col_type_lower = col_type.lower()
        
        if any(t in col_type_lower for t in ["int", "float", "double", "decimal", "numeric"]):
            minmax_query = f'''
                SELECT MIN("{col_name}"), MAX("{col_name}")
                FROM "{table_name}"
                WHERE "{col_name}" IS NOT NULL
            '''
            minmax = conn.execute(minmax_query).fetchone()
            if minmax[0] is not None:
                min_val = str(minmax[0])
                max_val = str(minmax[1])
        
        elif any(t in col_type_lower for t in ["date", "timestamp"]):
            minmax_query = f'''
                SELECT MIN("{col_name}"), MAX("{col_name}")
                FROM "{table_name}"
                WHERE "{col_name}" IS NOT NULL
            '''
            minmax = conn.execute(minmax_query).fetchone()
            if minmax[0] is not None:
                min_val = str(minmax[0])
                max_val = str(minmax[1])
        
        # Determine semantic type
        semantic_type = self._infer_semantic_type(col_name, col_type, sample_values)
        
        # Determine if likely key or metric
        is_key = self._is_likely_key(col_name, distinct_count, row_count)
        is_metric = self._is_likely_metric(col_name, col_type, semantic_type)
        
        return ColumnProfile(
            name=col_name,
            data_type=col_type,
            semantic_type=semantic_type,
            null_rate=round(null_rate, 4),
            distinct_count=distinct_count,
            sample_values=sample_values,
            min_value=min_val,
            max_value=max_val,
            is_likely_key=is_key,
            is_likely_metric=is_metric,
        )
    
    def _infer_semantic_type(
        self,
        col_name: str,
        col_type: str,
        sample_values: list[str],
    ) -> str | None:
        """Infer semantic type from column name and type."""
        name_lower = col_name.lower()
        type_lower = col_type.lower()
        
        # Timestamp detection
        if any(t in type_lower for t in ["date", "timestamp", "time"]):
            return "timestamp"
        if any(kw in name_lower for kw in ["date", "time", "created", "updated", "at", "_dt"]):
            return "timestamp"
        
        # Money detection
        if any(kw in name_lower for kw in ["amount", "price", "cost", "revenue", "payment", "fee", "total"]):
            return "money"
        
        # Rate detection
        if any(kw in name_lower for kw in ["rate", "percent", "ratio", "pct"]):
            return "rate"
        
        # Identifier detection
        if name_lower.endswith("_id") or name_lower == "id":
            return "identifier"
        
        # Count detection
        if any(kw in name_lower for kw in ["count", "qty", "quantity", "num"]):
            return "count"
        
        # Status detection
        if any(kw in name_lower for kw in ["status", "state", "type", "category"]):
            return "category"
        
        # Text detection
        if any(t in type_lower for t in ["varchar", "text", "char"]):
            return "text"
        
        # Numeric detection
        if any(t in type_lower for t in ["int", "float", "double", "decimal", "numeric"]):
            return "numeric"
        
        return None
    
    def _is_likely_key(self, col_name: str, distinct_count: int, row_count: int) -> bool:
        """Determine if column is likely a key/ID."""
        name_lower = col_name.lower()
        
        # Name-based detection
        if name_lower.endswith("_id") or name_lower == "id":
            return True
        
        # High cardinality check (unique or nearly unique)
        if row_count > 0 and distinct_count / row_count > 0.9:
            return True
        
        return False
    
    def _is_likely_metric(self, col_name: str, col_type: str, semantic_type: str | None) -> bool:
        """Determine if column is likely a metric (numeric measure)."""
        if semantic_type in ["money", "rate", "count", "numeric"]:
            name_lower = col_name.lower()
            # Exclude identifiers
            if name_lower.endswith("_id") or name_lower == "id":
                return False
            return True
        return False
    
    def _infer_joins(self, table_profiles: list[TableProfile]) -> list[JoinEdge]:
        """Infer join relationships between tables."""
        joins = []
        
        # Build a map of column names to tables
        column_map: dict[str, list[tuple[str, ColumnProfile]]] = {}
        for table in table_profiles:
            for col in table.columns:
                if col.is_likely_key:
                    key = col.name.lower()
                    if key not in column_map:
                        column_map[key] = []
                    column_map[key].append((table.name, col))
        
        # Find matching columns across tables
        for col_name, tables_with_col in column_map.items():
            if len(tables_with_col) >= 2:
                # Found a potential join column
                for i, (table1, col1) in enumerate(tables_with_col):
                    for table2, col2 in tables_with_col[i + 1:]:
                        # Check if data types are compatible
                        if self._types_compatible(col1.data_type, col2.data_type):
                            joins.append(JoinEdge(
                                from_table=table1,
                                to_table=table2,
                                from_column=col1.name,
                                to_column=col2.name,
                                join_type="inner",
                                confidence=0.8,
                                evidence=f"Matching column name '{col_name}' found in both tables",
                            ))
        
        return joins
    
    def _types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two column types are compatible for joining."""
        t1 = type1.lower()
        t2 = type2.lower()
        
        # Same type
        if t1 == t2:
            return True
        
        # Numeric types are compatible
        numeric_types = ["int", "bigint", "integer", "float", "double", "decimal", "numeric"]
        is_t1_numeric = any(n in t1 for n in numeric_types)
        is_t2_numeric = any(n in t2 for n in numeric_types)
        if is_t1_numeric and is_t2_numeric:
            return True
        
        # String types are compatible
        string_types = ["varchar", "text", "char"]
        is_t1_string = any(s in t1 for s in string_types)
        is_t2_string = any(s in t2 for s in string_types)
        if is_t1_string and is_t2_string:
            return True
        
        return False
    
    def _find_relevant_schema(
        self,
        intake: IntakeResult,
        tables: list[TableProfile],
    ) -> tuple[list[str], list[str]]:
        """Find tables and columns relevant to the intake result.
        
        IMPROVED: Uses semantic profiler's entity-to-table mapping to
        correctly identify tables (e.g., 'transaction' → test_2_1).
        """
        relevant_tables = []
        relevant_columns = []
        goal_lower = intake.clarified_goal.lower()
        
        # STEP 1: Use semantic profiler if available (BEST approach)
        if self._semantic:
            # Check entity-to-table mappings
            for entity, table_name in self._semantic.entity_to_table.items():
                if entity in goal_lower or entity + 's' in goal_lower:
                    if table_name not in relevant_tables:
                        relevant_tables.append(table_name)
                        
                        # Get semantic info about this table
                        table_info = self._semantic.tables.get(table_name)
                        if table_info:
                            # Add primary key columns
                            for pk in table_info.primary_key_columns:
                                relevant_columns.append(f"{table_name}.{pk}")
                            # Add primary timestamp
                            if table_info.primary_timestamp_column:
                                relevant_columns.append(f"{table_name}.{table_info.primary_timestamp_column}")
                            # Add primary amount
                            if table_info.primary_amount_column:
                                relevant_columns.append(f"{table_name}.{table_info.primary_amount_column}")
            
            # Also check domain mappings
            for domain, domain_tables in self._semantic.domain_to_tables.items():
                if domain in goal_lower:
                    for t in domain_tables:
                        if t not in relevant_tables:
                            relevant_tables.append(t)
        
        # STEP 2: Fallback to keyword matching if no semantic matches
        if not relevant_tables:
            # Check for entity keywords
            for entity, keywords in ENTITY_KEYWORDS.items():
                if any(kw in goal_lower for kw in keywords):
                    # Find tables with columns matching this entity
                    for table in tables:
                        for col in table.columns:
                            if any(kw in col.name.lower() for kw in keywords):
                                if table.name not in relevant_tables:
                                    relevant_tables.append(table.name)
                                relevant_columns.append(f"{table.name}.{col.name}")
                                break
        
        # STEP 3: Extract additional keywords from intake
        keywords = set()
        for metric in intake.metrics:
            keywords.add(metric.name.lower())
            if metric.column_hint:
                keywords.add(metric.column_hint.lower())
        
        for dim in intake.dimensions:
            keywords.add(dim.name.lower())
            if dim.column_hint:
                keywords.add(dim.column_hint.lower())
        
        # Score tables by keyword matches
        for table in tables:
            table_score = 0
            matching_cols = []
            
            # Check table name
            if any(kw in table.name.lower() for kw in keywords):
                table_score += 2
            
            # Check columns
            for col in table.columns:
                col_name_lower = col.name.lower()
                if any(kw in col_name_lower for kw in keywords):
                    table_score += 1
                    matching_cols.append(f"{table.name}.{col.name}")
            
            if table_score > 0:
                if table.name not in relevant_tables:
                    relevant_tables.append(table.name)
                relevant_columns.extend(matching_cols)
        
        # STEP 4: If still no matches, return first table (not all - that's too noisy)
        if not relevant_tables and tables:
            # Pick the largest table as default
            largest = max(tables, key=lambda t: t.row_count)
            relevant_tables.append(largest.name)
        
        return relevant_tables, relevant_columns
    
    def _suggest_metrics(
        self,
        tables: list[TableProfile],
        relevant_tables: list[str],
    ) -> list[str]:
        """Suggest metric columns from relevant tables."""
        metrics = []
        for table in tables:
            if table.name in relevant_tables:
                for col in table.columns:
                    if col.is_likely_metric:
                        metrics.append(f"{table.name}.{col.name}")
        return metrics[:10]  # Limit to 10
    
    def _suggest_dimensions(
        self,
        tables: list[TableProfile],
        relevant_tables: list[str],
    ) -> list[str]:
        """Suggest dimension columns from relevant tables."""
        dimensions = []
        for table in tables:
            if table.name in relevant_tables:
                for col in table.columns:
                    if col.semantic_type == "category":
                        dimensions.append(f"{table.name}.{col.name}")
                    elif col.is_likely_key and not col.name.lower().endswith("_id"):
                        dimensions.append(f"{table.name}.{col.name}")
        return dimensions[:10]
    
    def _suggest_time_column(
        self,
        tables: list[TableProfile],
        relevant_tables: list[str],
    ) -> str | None:
        """Suggest the best time column for filtering.
        
        IMPROVED: Uses semantic profiler's primary_timestamp_column
        which is determined by data quality analysis.
        """
        # STEP 1: Check semantic profiler first (BEST approach)
        if self._semantic:
            for table_name in relevant_tables:
                table_info = self._semantic.tables.get(table_name)
                if table_info and table_info.primary_timestamp_column:
                    return f"{table_name}.{table_info.primary_timestamp_column}"
        
        # STEP 2: Fallback to heuristic detection
        for table in tables:
            if table.name in relevant_tables and table.timestamp_columns:
                # Prefer mt103_created_at for transaction tables
                for col in table.timestamp_columns:
                    col_lower = col.lower()
                    if 'mt103_created_at' in col_lower:
                        return f"{table.name}.{col}"
                
                # Then prefer 'created_at', 'date', 'timestamp' etc.
                for col in table.timestamp_columns:
                    col_lower = col.lower()
                    if any(kw in col_lower for kw in ["created", "date", "timestamp", "time"]):
                        return f"{table.name}.{col}"
                # Return first timestamp column
                return f"{table.name}.{table.timestamp_columns[0]}"
        return None
    
    def get_semantic_context(self) -> str | None:
        """Get semantic context for query generation.
        
        Returns a string that can be injected into LLM prompts to help
        with table selection and column usage.
        """
        if self._semantic:
            return self._semantic.generate_context_prompt()
        return None
