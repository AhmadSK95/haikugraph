"""Card schema definitions using Pydantic."""

from typing import Literal

from pydantic import BaseModel, Field


class TableCard(BaseModel):
    """Card representing a table with semantic annotations."""

    id: str
    card_type: Literal["table"] = "table"
    table: str
    grain: str
    primary_key_candidates: list[str] = Field(default_factory=list)
    time_cols: list[str] = Field(default_factory=list)
    money_cols: list[str] = Field(default_factory=list)
    entity_cols: list[str] = Field(default_factory=list)
    status_cols: list[str] = Field(default_factory=list)
    suggested_metrics: list[str] = Field(default_factory=list)
    gotchas: list[str] = Field(default_factory=list)
    joins_suspected: int = 0


class ColumnCard(BaseModel):
    """Card representing a column with profiling data."""

    id: str
    card_type: Literal["column"] = "column"
    table: str
    column: str
    duckdb_type: str
    null_pct: float
    distinct_count: int
    sample_values: list[str] = Field(default_factory=list)
    semantic_hints: list[str] = Field(default_factory=list)


class RelationCard(BaseModel):
    """Card representing a potential join relationship."""

    id: str
    card_type: Literal["relation"] = "relation"
    left_table: str
    right_table: str
    left_col: str
    right_col: str
    confidence: float
    evidence: dict[str, float | int]
    notes: str
    probe: dict[str, float | int | str] | None = None


class CardIndex(BaseModel):
    """Index of all cards."""

    cards: list[dict[str, str]] = Field(default_factory=list)
    by_table: dict[str, list[str]] = Field(default_factory=dict)
