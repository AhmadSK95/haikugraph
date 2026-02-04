"""Update relation cards with join probe results."""

import json
from pathlib import Path

import duckdb

from haikugraph.cards.store import load_card, load_index
from haikugraph.graph.probe import probe_relation, update_confidence_from_probe


def probe_and_update_relations(
    db_path: Path,
    cards_dir: Path,
    out_report_path: Path,
    sample_limit: int = 200000,
    max_relations: int | None = None,
) -> dict:
    """
    Probe all relation cards and update them with join metrics.

    Args:
        db_path: Path to DuckDB database
        cards_dir: Path to cards directory
        out_report_path: Path to output join report
        sample_limit: Max rows for sampling (default: 200000)
        max_relations: Max relations to probe (optional, for quick runs)

    Returns:
        Summary dictionary
    """
    # Load card index
    index = load_index(cards_dir)
    if not index:
        raise FileNotFoundError("Card index not found. Run 'haikugraph cards build' first.")

    # Get all relation cards
    relation_cards = [card for card in index.cards if card["card_type"] == "relation"]

    # Limit if requested
    if max_relations:
        relation_cards = relation_cards[:max_relations]

    if not relation_cards:
        return {
            "status": "no_relations",
            "message": "No relation cards found",
            "relations": [],
        }

    # Connect to database
    conn = duckdb.connect(str(db_path), read_only=True)

    results = {"relations": [], "errors": []}

    for card_meta in relation_cards:
        card_id = card_meta["id"]

        try:
            # Load card
            card_data = load_card(card_id, cards_dir)
            if not card_data:
                continue

            # Probe the join
            probe_result = probe_relation(
                conn=conn,
                left_table=card_data["left_table"],
                right_table=card_data["right_table"],
                join_col=card_data["left_col"],
                sample_limit=sample_limit,
            )

            # Update confidence
            old_confidence = card_data["confidence"]
            new_confidence = update_confidence_from_probe(old_confidence, probe_result)

            # Update card
            card_data["probe"] = probe_result
            card_data["confidence"] = round(new_confidence, 2)

            # Update notes with probe summary (bidirectional)
            key_cov_l2r = probe_result.get(
                "key_coverage_distinct_left_to_right",
                probe_result.get("key_coverage_left_to_right", 0),
            )
            key_cov_r2l = probe_result.get("key_coverage_distinct_right_to_left", 0)
            row_join_left = probe_result["row_joinable_pct_left"]
            row_join_right = probe_result.get("row_joinable_pct_right", 0)

            card_data["notes"] = (
                f"probed: key_l2r={key_cov_l2r:.1f}% key_r2l={key_cov_r2l:.1f}% "
                f"row_l={row_join_left:.1f}% row_r={row_join_right:.1f}% "
                f"card={probe_result['estimated_cardinality']}"
            )

            # Save updated card
            card_path = cards_dir / card_meta["path"]
            with open(card_path, "w") as f:
                json.dump(card_data, f, indent=2)

            # Add to results
            results["relations"].append(
                {
                    "id": card_id,
                    "left_table": card_data["left_table"],
                    "right_table": card_data["right_table"],
                    "join_col": card_data["left_col"],
                    "confidence": new_confidence,
                    "key_coverage_l2r": key_cov_l2r,
                    "key_coverage_r2l": key_cov_r2l,
                    "row_joinable_left": row_join_left,
                    "row_joinable_right": row_join_right,
                    "cardinality": probe_result["estimated_cardinality"],
                }
            )

        except Exception as e:
            results["errors"].append({"card_id": card_id, "error": str(e)})

    conn.close()

    # Sort by confidence
    results["relations"].sort(key=lambda x: x["confidence"], reverse=True)

    # Flag low coverage relations (average of both directions)
    results["low_coverage"] = [
        r for r in results["relations"] if (r["key_coverage_l2r"] + r["key_coverage_r2l"]) / 2 < 50
    ]

    # Save report
    with open(out_report_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def format_probe_summary(results: dict, out_report_path: Path) -> str:
    """Format probe results as summary."""
    lines = []

    if results.get("status") == "no_relations":
        lines.append("❌ No relation cards found")
        return "\n".join(lines)

    total = len(results["relations"])
    errors = len(results["errors"])

    lines.append("✅ Join probing complete\n")
    lines.append(f"Relations probed: {total}")
    if errors:
        lines.append(f"Errors: {errors}")

    # Top 5 strongest joins
    if results["relations"]:
        lines.append("\n" + "─" * 70)
        lines.append("Top 5 Strongest Joins:\n")
        for i, rel in enumerate(results["relations"][:5], 1):
            lines.append(
                f"{i}. {rel['left_table']}.{rel['join_col']} ~ "
                f"{rel['right_table']}.{rel['join_col']}"
            )
            lines.append(
                f"   Confidence: {rel['confidence']:.2f} | "
                f"Cov L→R: {rel['key_coverage_l2r']:.1f}% R→L: {rel['key_coverage_r2l']:.1f}% | "
                f"Join L: {rel['row_joinable_left']:.1f}% R: {rel['row_joinable_right']:.1f}% | "
                f"Card: {rel['cardinality']}"
            )

    # Bottom 5 weakest joins
    if len(results["relations"]) > 5:
        lines.append("\n" + "─" * 70)
        lines.append("Bottom 5 Weakest Joins:\n")
        for i, rel in enumerate(results["relations"][-5:], 1):
            lines.append(
                f"{i}. {rel['left_table']}.{rel['join_col']} ~ "
                f"{rel['right_table']}.{rel['join_col']}"
            )
            lines.append(
                f"   Confidence: {rel['confidence']:.2f} | "
                f"Cov L→R: {rel['key_coverage_l2r']:.1f}% R→L: {rel['key_coverage_r2l']:.1f}% | "
                f"Join L: {rel['row_joinable_left']:.1f}% R: {rel['row_joinable_right']:.1f}% | "
                f"Card: {rel['cardinality']}"
            )

    # Low coverage warning
    if results.get("low_coverage"):
        lines.append("\n" + "─" * 70)
        lines.append(f"⚠️  {len(results['low_coverage'])} relations have <50% key coverage")

    lines.append(f"\nReport saved to: {out_report_path.absolute()}")

    return "\n".join(lines)
