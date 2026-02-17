"""Generate data cards from profile."""

import json
from pathlib import Path

from haikugraph.cards.schemas import ColumnCard, RelationCard, TableCard


def infer_grain(table_name: str, columns: list[dict], row_count: int) -> str:
    """Infer table grain from column names."""
    col_names = {col["name"].lower() for col in columns}

    if "transaction_id" in col_names:
        return "one row = transaction / transaction-related record"
    if "quote_id" in col_names:
        return "one row = quote"
    if "deal_id" in col_names:
        return "one row = deal"
    if "payee_id" in col_names and row_count < 10000:
        return "one row = payee"
    if "customer_id" in col_names and row_count < 10000:
        return "one row = customer"

    return "one row = record"


def find_primary_key_candidates(columns: list[dict], row_count: int) -> list[str]:
    """Find potential primary key columns."""
    candidates = []

    # Known ID column patterns (case-insensitive)
    known_ids = {
        "sha_id",
        "transaction_id",
        "quote_id",
        "payee_id",
        "deal_id",
        "customer_id",
    }

    for col in columns:
        col_name = col["name"]
        col_name_lower = col_name.lower()
        col_type = col.get("duckdb_type", "").upper()
        null_pct = col.get("null_pct", 100)
        distinct_count = col.get("distinct_count", 0)

        # Must have no nulls
        if null_pct > 0:
            continue

        # Exclude timestamp/date columns unless they're in the known ID list
        # or explicitly end with _id
        is_temporal_type = "TIMESTAMP" in col_type or "DATE" in col_type
        is_temporal_name = (
            col_name_lower.endswith("_at")
            or col_name_lower.endswith("_date")
            or "created_at" in col_name_lower
            or "updated_at" in col_name_lower
            or "expires_at" in col_name_lower
        )
        is_temporal = is_temporal_type or is_temporal_name
        is_known_id = col_name_lower in known_ids
        is_id_column = col_name_lower.endswith("_id")

        if is_temporal and not (is_known_id or is_id_column):
            continue

        # Check if distinct count is close to row count (likely unique)
        if row_count > 0 and distinct_count >= 0.98 * row_count:
            candidates.append(col_name)
        # Or if it's a known ID column
        elif is_known_id:
            candidates.append(col_name)

    return candidates


def find_time_cols(columns: list[dict]) -> list[str]:
    """Find timestamp columns."""
    time_cols = []
    time_patterns = ["_at", "created_at", "updated_at", "expires_at", "_date", "_time"]

    for col in columns:
        col_name = col["name"].lower()
        col_type = col.get("duckdb_type", "").lower()

        # Must be timestamp type
        if "timestamp" not in col_type and "date" not in col_type:
            continue

        # Check name patterns
        if any(pattern in col_name for pattern in time_patterns):
            time_cols.append(col["name"])

    return time_cols


def find_money_cols(columns: list[dict]) -> list[str]:
    """Find money-related columns."""
    money_cols = []
    money_patterns = [
        "amount",
        "charges",
        "gst",
        "tcs",
        "markup",
        "rate",
        "balance",
        "fee",
        "price",
        "cost",
    ]

    for col in columns:
        col_name = col["name"].lower()
        if any(pattern in col_name for pattern in money_patterns):
            money_cols.append(col["name"])

    return money_cols


def find_entity_cols(columns: list[dict]) -> list[str]:
    """Find entity/categorical columns."""
    entity_cols = []
    entity_patterns = [
        "_id",
        "country",
        "state",
        "city",
        "type",
        "platform_name",
        "referral_code",
        "purpose_code",
        "currency",
    ]

    for col in columns:
        col_name = col["name"].lower()
        if any(pattern in col_name for pattern in entity_patterns):
            entity_cols.append(col["name"])

    return entity_cols


def find_status_cols(columns: list[dict]) -> list[str]:
    """Find status columns."""
    return [col["name"] for col in columns if "status" in col["name"].lower()]


def infer_semantic_hints(col_name: str, col_type: str) -> list[str]:
    """Infer semantic hints for a column."""
    hints = []
    name_lower = col_name.lower()

    if "_id" in name_lower or name_lower.endswith("_id"):
        hints.append("identifier")

    # Timestamp: only if ends with _at/_date, or contains specific timestamp patterns
    if (
        name_lower.endswith("_at")
        or name_lower.endswith("_date")
        or "created_at" in name_lower
        or "updated_at" in name_lower
        or "expires_at" in name_lower
        or "timestamp" in name_lower
    ):
        hints.append("timestamp")

    if any(p in name_lower for p in ["amount", "balance", "charges", "fee", "cost"]):
        hints.append("money_or_numeric")
    if "currency" in name_lower:
        hints.append("currency_code")
    if "status" in name_lower:
        hints.append("status")
    if any(p in name_lower for p in ["country", "state", "city"]):
        hints.append("geography")
    if "email" in name_lower:
        hints.append("email")
    if "phone" in name_lower:
        hints.append("phone")

    return hints


def generate_table_card(
    table_name: str, table_profile: dict, all_tables: dict[str, dict]
) -> TableCard:
    """Generate a TableCard from table profile."""
    columns = table_profile["columns"]
    row_count = table_profile["row_count"]

    # Infer grain
    grain = infer_grain(table_name, columns, row_count)

    # Find special columns
    pk_candidates = find_primary_key_candidates(columns, row_count)
    time_cols = find_time_cols(columns)
    money_cols = find_money_cols(columns)
    entity_cols = find_entity_cols(columns)
    status_cols = find_status_cols(columns)

    # Suggested metrics
    suggested_metrics = [f"count({table_name}) -- row count"]
    if money_cols:
        for mcol in money_cols[:3]:  # Limit to first 3
            suggested_metrics.append(f"sum({mcol}) -- total {mcol}")

    # Gotchas
    gotchas = []

    # Check if all columns are VARCHAR (string mode)
    if all(col.get("duckdb_type") == "VARCHAR" for col in columns):
        gotchas.append("string_mode_table: types need refinement")

    # Check for sparse table
    high_null_cols = [col for col in columns if col.get("null_pct", 0) > 50]
    if len(high_null_cols) / len(columns) > 0.3:
        gotchas.append(f"sparse_table_many_nulls: {len(high_null_cols)} cols >50% null")

    # Detect non-unique _id columns (need COUNT DISTINCT when counting)
    non_unique_ids = []
    for col in columns:
        col_name = col["name"].lower()
        distinct_count = col.get("distinct_count", 0)
        if (
            col_name.endswith("_id")
            and row_count > 0
            and distinct_count > 0
            and distinct_count < 0.98 * row_count
        ):
            dup_ratio = round(row_count / distinct_count, 1)
            non_unique_ids.append(f"{col['name']} ({dup_ratio}x duplicates)")
    if non_unique_ids:
        gotchas.append(
            f"non_unique_id_columns: {', '.join(non_unique_ids)} — use COUNT(DISTINCT) when counting"
        )

    # Count suspected joins (will be updated when building relations)
    joins_suspected = 0

    card_id = f"table:{table_name}"

    return TableCard(
        id=card_id,
        table=table_name,
        grain=grain,
        primary_key_candidates=pk_candidates,
        time_cols=time_cols,
        money_cols=money_cols,
        entity_cols=entity_cols,
        status_cols=status_cols,
        suggested_metrics=suggested_metrics,
        gotchas=gotchas,
        joins_suspected=joins_suspected,
    )


def generate_column_card(table_name: str, col_profile: dict, row_count: int = 0) -> ColumnCard:
    """Generate a ColumnCard from column profile."""
    col_name = col_profile["name"]
    card_id = f"column:{table_name}.{col_name}"

    # Get semantic hints
    semantic_hints = infer_semantic_hints(col_name, col_profile.get("duckdb_type", ""))

    # Determine uniqueness: distinct_count >= 98% of row_count
    distinct_count = col_profile.get("distinct_count", 0)
    is_unique = (
        row_count > 0
        and distinct_count > 0
        and distinct_count >= 0.98 * row_count
    )

    return ColumnCard(
        id=card_id,
        table=table_name,
        column=col_name,
        duckdb_type=col_profile.get("duckdb_type", "UNKNOWN"),
        null_pct=col_profile.get("null_pct", 0.0),
        distinct_count=distinct_count,
        is_unique=is_unique,
        sample_values=col_profile.get("sample_values", [])[:5],  # Limit to 5
        semantic_hints=semantic_hints,
    )


def generate_relation_cards(
    all_tables: dict[str, dict], table_cards: dict[str, TableCard]
) -> list[RelationCard]:
    """Generate RelationCards for potential joins."""
    relations = []

    # Known join key patterns
    join_key_names = {
        "customer_id",
        "transaction_id",
        "payee_id",
        "quote_id",
        "deal_id",
        "sha_id",
    }

    # Build column index: column_name -> [(table, col_info), ...]
    col_index: dict[str, list[tuple[str, dict]]] = {}
    for table_name, table_profile in all_tables.items():
        for col in table_profile["columns"]:
            col_name = col["name"]
            if col_name not in col_index:
                col_index[col_name] = []
            col_index[col_name].append((table_name, col))

    # Find shared columns
    for col_name, table_col_pairs in col_index.items():
        # Need at least 2 tables sharing this column
        if len(table_col_pairs) < 2:
            continue

        # Only consider join keys (ends with _id or in known list)
        if not (col_name.endswith("_id") or col_name in join_key_names):
            continue

        # Create relation cards for each pair
        for i, (left_table, left_col) in enumerate(table_col_pairs):
            for right_table, right_col in table_col_pairs[i + 1 :]:
                # Calculate confidence
                confidence = 0.6

                left_row_count = all_tables[left_table]["row_count"]
                right_row_count = all_tables[right_table]["row_count"]

                # Check if right table has this as PK
                right_pk_candidates = table_cards[right_table].primary_key_candidates
                if col_name in right_pk_candidates:
                    confidence += 0.15

                # Check if left has many more rows (many-to-one)
                if right_row_count > 0 and left_row_count > 2 * right_row_count:
                    confidence += 0.15

                # Check distinct counts
                left_distinct = left_col.get("distinct_count", 0)
                right_distinct = right_col.get("distinct_count", 0)
                if right_distinct > 0 and left_distinct >= right_distinct:
                    confidence += 0.1

                # Cap confidence
                confidence = min(confidence, 0.95)

                card_id = f"relation:{left_table}.{col_name}~{right_table}.{col_name}"

                relation = RelationCard(
                    id=card_id,
                    left_table=left_table,
                    right_table=right_table,
                    left_col=col_name,
                    right_col=col_name,
                    confidence=confidence,
                    evidence={
                        "left_distinct": left_distinct,
                        "right_distinct": right_distinct,
                        "left_null_pct": left_col.get("null_pct", 0.0),
                        "right_null_pct": right_col.get("null_pct", 0.0),
                    },
                    notes="needs_probe_join_coverage",
                )

                relations.append(relation)

    return relations


def generate_cards_from_profile(profile_path: Path, out_dir: Path) -> dict:
    """Generate all cards from profile and save to directory."""
    # Load profile
    with open(profile_path) as f:
        profile = json.load(f)

    all_tables = profile["tables"]

    # Generate table cards
    table_cards = {}
    for table_name, table_profile in all_tables.items():
        card = generate_table_card(table_name, table_profile, all_tables)
        table_cards[table_name] = card

    # Generate column cards
    column_cards = []
    for table_name, table_profile in all_tables.items():
        row_count = table_profile.get("row_count", 0)
        for col_profile in table_profile["columns"]:
            card = generate_column_card(table_name, col_profile, row_count)
            column_cards.append(card)

    # Generate relation cards
    relation_cards = generate_relation_cards(all_tables, table_cards)

    # Update joins_suspected count in table cards
    for relation in relation_cards:
        table_cards[relation.left_table].joins_suspected += 1
        table_cards[relation.right_table].joins_suspected += 1

    return {
        "table_cards": list(table_cards.values()),
        "column_cards": column_cards,
        "relation_cards": relation_cards,
    }
