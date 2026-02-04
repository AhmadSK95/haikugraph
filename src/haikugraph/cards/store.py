"""Card storage and indexing."""

import json
from pathlib import Path
from typing import Any

from haikugraph.cards.schemas import CardIndex


def save_cards(cards: dict[str, list], out_dir: Path) -> CardIndex:
    """
    Save all cards as individual JSON files and build index.

    Args:
        cards: Dictionary with table_cards, column_cards, relation_cards
        out_dir: Output directory for cards

    Returns:
        CardIndex with all card metadata
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    index = CardIndex()

    # Save table cards
    for card in cards["table_cards"]:
        card_path = out_dir / f"{card.id.replace(':', '_').replace('.', '_')}.json"
        with open(card_path, "w") as f:
            json.dump(card.model_dump(), f, indent=2)

        index.cards.append(
            {
                "id": card.id,
                "card_type": "table",
                "title": f"Table: {card.table}",
                "table": card.table,
                "path": str(card_path.name),
            }
        )

        if card.table not in index.by_table:
            index.by_table[card.table] = []
        index.by_table[card.table].append(card.id)

    # Save column cards
    for card in cards["column_cards"]:
        card_path = out_dir / f"{card.id.replace(':', '_').replace('.', '_')}.json"
        with open(card_path, "w") as f:
            json.dump(card.model_dump(), f, indent=2)

        index.cards.append(
            {
                "id": card.id,
                "card_type": "column",
                "title": f"Column: {card.table}.{card.column}",
                "table": card.table,
                "path": str(card_path.name),
            }
        )

        if card.table not in index.by_table:
            index.by_table[card.table] = []
        index.by_table[card.table].append(card.id)

    # Save relation cards
    for card in cards["relation_cards"]:
        safe_id = card.id.replace(":", "_").replace(".", "_").replace("~", "_")
        card_path = out_dir / f"{safe_id}.json"
        with open(card_path, "w") as f:
            json.dump(card.model_dump(), f, indent=2)

        index.cards.append(
            {
                "id": card.id,
                "card_type": "relation",
                "title": (
                    f"Join: {card.left_table}.{card.left_col} ~ {card.right_table}.{card.right_col}"
                ),
                "tables": f"{card.left_table},{card.right_table}",
                "path": str(card_path.name),
            }
        )

        # Add to both tables
        for table in [card.left_table, card.right_table]:
            if table not in index.by_table:
                index.by_table[table] = []
            index.by_table[table].append(card.id)

    # Save index
    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index.model_dump(), f, indent=2)

    return index


def load_card(card_id: str, cards_dir: Path) -> dict[str, Any] | None:
    """Load a card by ID."""
    # Try to find the card file
    filename = card_id.replace(":", "_").replace(".", "_").replace("~", "_") + ".json"
    card_path = cards_dir / filename

    if not card_path.exists():
        return None

    with open(card_path) as f:
        return json.load(f)


def load_index(cards_dir: Path) -> CardIndex | None:
    """Load the card index."""
    index_path = cards_dir / "index.json"

    if not index_path.exists():
        return None

    with open(index_path) as f:
        return CardIndex(**json.load(f))
