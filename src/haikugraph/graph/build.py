"""Build relationship graph from data cards."""

import json
from pathlib import Path


def load_cards(cards_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load all cards from the cards directory.

    Returns:
        Tuple of (table_cards, column_cards, relation_cards)
    """
    index_path = cards_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    table_cards = []
    column_cards = []
    relation_cards = []

    for card_meta in index["cards"]:
        card_path = cards_dir / card_meta["path"]
        if not card_path.exists():
            continue

        with open(card_path) as f:
            card = json.load(f)

        card_type = card.get("card_type")
        if card_type == "table":
            table_cards.append(card)
        elif card_type == "column":
            column_cards.append(card)
        elif card_type == "relation":
            relation_cards.append(card)

    return table_cards, column_cards, relation_cards


def build_graph(
    table_cards: list[dict],
    column_cards: list[dict],
    relation_cards: list[dict],
    min_confidence: float = 0.5,
    weak_threshold: float = 0.7,
) -> dict:
    """
    Build graph from cards with confidence filtering.

    Args:
        table_cards: List of table card dicts
        column_cards: List of column card dicts
        relation_cards: List of relation card dicts
        min_confidence: Minimum confidence to include relation (default 0.5)
        weak_threshold: Threshold below which relation is marked weak (default 0.7)

    Returns:
        Graph dict with nodes, edges, and metadata
    """
    # Build nodes
    table_nodes = []
    for table_card in table_cards:
        table_nodes.append(
            {
                "id": table_card["table"],
                "type": "table",
                "grain": table_card.get("grain", "unknown"),
                "primary_key_candidates": table_card.get("primary_key_candidates", []),
                "gotchas": table_card.get("gotchas", []),
            }
        )

    column_nodes = []
    for col_card in column_cards:
        column_nodes.append(
            {
                "id": f"{col_card['table']}.{col_card['column']}",
                "type": "column",
                "table": col_card["table"],
                "column": col_card["column"],
                "duckdb_type": col_card.get("duckdb_type", "unknown"),
                "null_pct": col_card.get("null_pct", 0.0),
                "distinct_count": col_card.get("distinct_count", 0),
                "semantic_hints": col_card.get("semantic_hints", []),
            }
        )

    # Build edges from relations with filtering
    edges = []
    filtered_count = 0

    for rel_card in relation_cards:
        confidence = rel_card.get("confidence", 0.0)

        # Filter out low confidence relations
        if confidence < min_confidence:
            filtered_count += 1
            continue

        # Mark as weak if below threshold
        is_weak = confidence < weak_threshold

        # Extract cardinality and sampling info from probe
        probe = rel_card.get("probe", {})
        cardinality = probe.get("estimated_cardinality", "unknown")
        left_sampled = probe.get("left_sampled", False)
        right_sampled = probe.get("right_sampled", False)

        # Get join column name (both sides use same column name)
        join_col = rel_card.get("left_col", "")

        edges.append(
            {
                "id": rel_card["id"],
                "source": rel_card["left_table"],
                "target": rel_card["right_table"],
                "join_column": join_col,
                "confidence": confidence,
                "cardinality": cardinality,
                "is_weak": is_weak,
                "left_sampled": left_sampled,
                "right_sampled": right_sampled,
            }
        )

    # Count strong vs weak edges
    strong_edges = [e for e in edges if not e["is_weak"]]
    weak_edges = [e for e in edges if e["is_weak"]]

    return {
        "nodes": {
            "tables": table_nodes,
            "columns": column_nodes,
        },
        "edges": edges,
        "metadata": {
            "total_tables": len(table_nodes),
            "total_columns": len(column_nodes),
            "total_relations": len(edges),
            "strong_relations": len(strong_edges),
            "weak_relations": len(weak_edges),
            "filtered_relations": filtered_count,
            "min_confidence": min_confidence,
            "weak_threshold": weak_threshold,
        },
    }


def save_graph(graph: dict, output_path: Path) -> None:
    """Save graph to JSON file."""
    with open(output_path, "w") as f:
        json.dump(graph, f, indent=2)
