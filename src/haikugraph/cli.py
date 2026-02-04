"""CLI entrypoint for haikugraph."""

import json
import sys
from pathlib import Path

import click
import duckdb

from haikugraph import __version__
from haikugraph.cards.generate import generate_cards_from_profile
from haikugraph.cards.store import load_card, load_index, save_cards
from haikugraph.execution import execute_plan, save_result
from haikugraph.graph.build import build_graph, load_cards, save_graph
from haikugraph.graph.update_relations import (
    format_probe_summary,
    probe_and_update_relations,
)
from haikugraph.io.ingest import format_ingestion_summary, ingest_excel_to_duckdb
from haikugraph.io.profile import format_profile_summary, profile_database
from haikugraph.llm.plan_generator import generate_plan
from haikugraph.planning.ambiguity import (
    ambiguity_to_question,
    apply_user_resolution,
    get_unresolved_ambiguities,
)
from haikugraph.planning.plan import (
    build_plan,
    load_cards_data,
    load_graph,
    save_plan,
)


def _interactive_ambiguity_resolution(plan: dict) -> dict:
    """Interactively resolve ambiguities in a plan.

    Args:
        plan: Plan dict with potential ambiguities

    Returns:
        Updated plan with resolved ambiguities
    """
    unresolved = get_unresolved_ambiguities(plan)

    if not unresolved:
        click.echo("\n✅ No unresolved ambiguities found.")
        return plan

    click.echo(f"\n🔍 Found {len(unresolved)} unresolved ambiguity/ambiguities\n")
    click.echo("=" * 70)

    updated_plan = plan
    for i, amb in enumerate(unresolved, 1):
        question_data = ambiguity_to_question(amb)

        click.echo(f"\n[{i}/{len(unresolved)}] {question_data['question']}")
        click.echo("\nOptions:")
        for j, option in enumerate(question_data["options"], 1):
            click.echo(f"  {j}. {option}")

        # Get user input
        while True:
            try:
                choice_input = click.prompt("\nYour choice (number)", type=int)
                if 1 <= choice_input <= len(question_data["options"]):
                    chosen = question_data["options"][choice_input - 1]
                    break
                else:
                    max_choice = len(question_data["options"])
                    click.echo(
                        f"Invalid choice. Please enter a number between 1 and {max_choice}"
                    )
            except (ValueError, click.Abort):
                click.echo("Invalid input. Please enter a number.")

        # Apply resolution
        updated_plan = apply_user_resolution(updated_plan, question_data["issue"], chosen)
        click.echo(f"✓ Selected: {chosen}")

    click.echo("\n" + "=" * 70)
    click.echo("\n✅ All ambiguities resolved!")
    return updated_plan


@click.group()
@click.version_option()
def main():
    """haikugraph - Data assistant with graph-based analysis."""
    pass


@main.command()
@click.option(
    "--data-dir",
    default="./data",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory containing Excel files (default: ./data)",
)
@click.option(
    "--db-path",
    default="./data/haikugraph.duckdb",
    type=click.Path(),
    help="Path to DuckDB database file (default: ./data/haikugraph.duckdb)",
)
@click.option(
    "--sheet",
    default=None,
    help="Sheet name or index to read (default: first sheet)",
)
@click.option(
    "--force",
    is_flag=True,
    default=True,
    help="Overwrite existing tables (default: True)",
)
def ingest(data_dir: str, db_path: str, sheet: str | None, force: bool):
    """Ingest Excel files from data directory into DuckDB."""
    # Convert sheet to int if it's a numeric string
    sheet_arg = None
    if sheet is not None:
        try:
            sheet_arg = int(sheet)
        except ValueError:
            sheet_arg = sheet

    results = ingest_excel_to_duckdb(
        data_dir=Path(data_dir),
        db_path=Path(db_path),
        sheet=sheet_arg,
        force=force,
    )

    summary = format_ingestion_summary(results)
    click.echo(summary)

    # Exit with non-zero if completely failed
    if results["status"] == "failed":
        raise click.Abort()


@main.command()
@click.option(
    "--db-path",
    default="./data/haikugraph.duckdb",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to DuckDB database (default: ./data/haikugraph.duckdb)",
)
@click.option(
    "--out",
    default="./data/profile.json",
    type=click.Path(),
    help="Output JSON file path (default: ./data/profile.json)",
)
@click.option(
    "--sample-rows",
    default=20000,
    type=int,
    help="Max rows for expensive scans (default: 20000)",
)
@click.option(
    "--top-k",
    default=10,
    type=int,
    help="Top values for categorical columns (default: 10)",
)
def profile(db_path: str, out: str, sample_rows: int, top_k: int):
    """Profile DuckDB tables and generate JSON report."""
    try:
        results = profile_database(
            db_path=Path(db_path),
            out_path=Path(out),
            sample_rows=sample_rows,
            top_k=top_k,
        )

        summary = format_profile_summary(results, Path(out))
        click.echo(summary)

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        click.echo(
            "\nRun 'haikugraph ingest' first to create the database.",
            err=True,
        )
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Error during profiling: {e}", err=True)
        raise click.Abort()


@main.group()
def cards():
    """Generate and manage data cards."""
    pass


@cards.command("build")
@click.option(
    "--profile",
    default="./data/profile.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to profile JSON (default: ./data/profile.json)",
)
@click.option(
    "--out-dir",
    default="./data/cards",
    type=click.Path(),
    help="Output directory for cards (default: ./data/cards)",
)
def cards_build(profile: str, out_dir: str):
    """Build data cards from profile."""
    try:
        profile_path = Path(profile)
        out_path = Path(out_dir)

        click.echo("Generating cards...")
        cards_dict = generate_cards_from_profile(profile_path, out_path)

        click.echo("Saving cards...")
        index = save_cards(cards_dict, out_path)

        click.echo("\n✅ Cards generated successfully\n")
        click.echo(f"Output directory: {out_path.absolute()}")
        click.echo(f"Table cards: {len(cards_dict['table_cards'])}")
        click.echo(f"Column cards: {len(cards_dict['column_cards'])}")
        click.echo(f"Relation cards: {len(cards_dict['relation_cards'])}")
        click.echo(f"\nTotal cards: {len(index.cards)}")

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        click.echo(
            "\nRun 'haikugraph profile' first to create the profile.",
            err=True,
        )
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Error generating cards: {e}", err=True)
        raise click.Abort()


@cards.command("list")
@click.option(
    "--cards-dir",
    default="./data/cards",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Cards directory (default: ./data/cards)",
)
def cards_list(cards_dir: str):
    """List all cards with counts."""
    cards_path = Path(cards_dir)
    index = load_index(cards_path)

    if not index:
        click.echo("❌ No card index found. Run 'haikugraph cards build' first.", err=True)
        raise click.Abort()

    # Count by type
    by_type = {"table": 0, "column": 0, "relation": 0}
    for card in index.cards:
        card_type = card.get("card_type")
        if card_type in by_type:
            by_type[card_type] += 1

    click.echo("Card Summary:\n")
    click.echo(f"Table cards: {by_type['table']}")
    click.echo(f"Column cards: {by_type['column']}")
    click.echo(f"Relation cards: {by_type['relation']}")
    click.echo(f"\nTotal: {len(index.cards)}")

    click.echo("\nTables:")
    for table in sorted(index.by_table.keys()):
        card_count = len(index.by_table[table])
        click.echo(f"  • {table:20s} ({card_count} cards)")


@cards.command("show")
@click.argument("card_id")
@click.option(
    "--cards-dir",
    default="./data/cards",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Cards directory (default: ./data/cards)",
)
def cards_show(card_id: str, cards_dir: str):
    """Show a specific card by ID."""
    cards_path = Path(cards_dir)
    card = load_card(card_id, cards_path)

    if not card:
        click.echo(f"❌ Card not found: {card_id}", err=True)
        raise click.Abort()

    click.echo(json.dumps(card, indent=2))


@cards.command("table")
@click.argument("table_name")
@click.option(
    "--cards-dir",
    default="./data/cards",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Cards directory (default: ./data/cards)",
)
def cards_table(table_name: str, cards_dir: str):
    """Show TableCard and top column cards for a table."""
    cards_path = Path(cards_dir)
    index = load_index(cards_path)

    if not index:
        click.echo("❌ No card index found. Run 'haikugraph cards build' first.", err=True)
        raise click.Abort()

    if table_name not in index.by_table:
        click.echo(f"❌ Table not found: {table_name}", err=True)
        raise click.Abort()

    # Load table card
    table_card_id = f"table:{table_name}"
    table_card = load_card(table_card_id, cards_path)

    if table_card:
        click.echo("=" * 70)
        click.echo(f"Table: {table_name}")
        click.echo("=" * 70)
        click.echo(f"\nGrain: {table_card['grain']}")
        pk_list = ", ".join(table_card["primary_key_candidates"]) or "none"
        click.echo(f"Primary key candidates: {pk_list}")
        click.echo(f"Joins suspected: {table_card['joins_suspected']}")

        if table_card["time_cols"]:
            click.echo(f"\nTime columns: {', '.join(table_card['time_cols'])}")
        if table_card["money_cols"]:
            click.echo(f"Money columns: {', '.join(table_card['money_cols'])}")
        if table_card["status_cols"]:
            click.echo(f"Status columns: {', '.join(table_card['status_cols'])}")

        if table_card["gotchas"]:
            click.echo("\nGotchas:")
            for gotcha in table_card["gotchas"]:
                click.echo(f"  ⚠️  {gotcha}")

    # Load and show top column cards
    column_card_ids = [cid for cid in index.by_table[table_name] if cid.startswith("column:")]
    click.echo(f"\n{'-' * 70}")
    click.echo(f"Columns ({len(column_card_ids)} total, showing first 10):\n")

    for card_id in column_card_ids[:10]:
        col_card = load_card(card_id, cards_path)
        if col_card:
            hints = col_card.get("semantic_hints", [])
            hints_str = f" [{', '.join(hints)}]" if hints else ""
            click.echo(
                f"  • {col_card['column']:30s} {col_card['duckdb_type']:12s} "
                f"nulls:{col_card['null_pct']:5.1f}% "
                f"distinct:{col_card['distinct_count']:6d}{hints_str}"
            )


@main.group()
def graph():
    """Build and analyze knowledge graph."""
    pass


@graph.command("build")
@click.option(
    "--cards-dir",
    default="./data/cards",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Cards directory (default: ./data/cards)",
)
@click.option(
    "--out",
    default="./data/graph.json",
    type=click.Path(),
    help="Output graph file (default: ./data/graph.json)",
)
@click.option(
    "--min-confidence",
    default=0.5,
    type=float,
    help="Minimum confidence to include relation (default: 0.5)",
)
@click.option(
    "--weak-threshold",
    default=0.7,
    type=float,
    help="Threshold below which relation is marked weak (default: 0.7)",
)
def graph_build(cards_dir: str, out: str, min_confidence: float, weak_threshold: float):
    """Build relationship graph from data cards."""
    try:
        click.echo("Building graph from cards...\n")

        cards_path = Path(cards_dir)
        out_path = Path(out)

        # Load cards
        table_cards, column_cards, relation_cards = load_cards(cards_path)

        # Build graph
        graph = build_graph(
            table_cards=table_cards,
            column_cards=column_cards,
            relation_cards=relation_cards,
            min_confidence=min_confidence,
            weak_threshold=weak_threshold,
        )

        # Save graph
        save_graph(graph, out_path)

        click.echo("✅ Graph built successfully\n")

        # Print summary
        meta = graph["metadata"]
        click.echo("Nodes:")
        click.echo(f"  Tables: {meta['total_tables']}")
        click.echo(f"  Columns: {meta['total_columns']}")
        click.echo("\nEdges (Relations):")
        click.echo(f"  Total: {meta['total_relations']}")
        click.echo(f"  Strong (>= {weak_threshold}): {meta['strong_relations']}")
        click.echo(f"  Weak (< {weak_threshold}): {meta['weak_relations']}")
        if meta["filtered_relations"] > 0:
            click.echo(f"  Filtered (< {min_confidence}): {meta['filtered_relations']}")

        click.echo(f"\nGraph saved to: {out_path.absolute()}")

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        click.echo("\nRun 'haikugraph cards build' first to generate cards.", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Error building graph: {e}", err=True)
        raise click.Abort()


@graph.command("probe-joins")
@click.option(
    "--db-path",
    default="./data/haikugraph.duckdb",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to DuckDB database (default: ./data/haikugraph.duckdb)",
)
@click.option(
    "--cards-dir",
    default="./data/cards",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Cards directory (default: ./data/cards)",
)
@click.option(
    "--out",
    default="./data/join_report.json",
    type=click.Path(),
    help="Output report path (default: ./data/join_report.json)",
)
@click.option(
    "--sample-limit",
    default=200000,
    type=int,
    help="Max rows for sampling distinct keys (default: 200000)",
)
@click.option(
    "--max-relations",
    default=None,
    type=int,
    help="Max relations to probe for quick runs (optional)",
)
def graph_probe_joins(
    db_path: str, cards_dir: str, out: str, sample_limit: int, max_relations: int | None
):
    """Probe join relationships with bidirectional metrics and sampling."""
    try:
        click.echo("Probing join relationships...\n")

        results = probe_and_update_relations(
            db_path=Path(db_path),
            cards_dir=Path(cards_dir),
            out_report_path=Path(out),
            sample_limit=sample_limit,
            max_relations=max_relations,
        )

        summary = format_probe_summary(results, Path(out))
        click.echo(summary)

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Error during probing: {e}", err=True)
        raise click.Abort()


@main.command()
@click.option(
    "--question",
    "-q",
    required=True,
    type=str,
    help="Natural language question about your data",
)
@click.option(
    "--graph-path",
    default="./data/graph.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to graph.json (default: ./data/graph.json)",
)
@click.option(
    "--cards-dir",
    default="./data/cards",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Cards directory (default: ./data/cards)",
)
@click.option(
    "--out",
    default="./data/plan.json",
    type=click.Path(),
    help="Output plan file (default: ./data/plan.json)",
)
@click.option(
    "--execute",
    is_flag=True,
    default=False,
    help="Execute the plan immediately after generation",
)
@click.option(
    "--db-path",
    default="./data/haikugraph.duckdb",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to DuckDB database (required if --execute is used)",
)
def ask(question: str, graph_path: str, cards_dir: str, out: str, execute: bool, db_path: str):
    """Plan how to answer a question (deterministic, no LLM)."""
    try:
        click.echo(f"Planning how to answer: {question}\n")

        # Load graph and cards
        graph = load_graph(Path(graph_path))
        cards = load_cards_data(Path(cards_dir))

        # Build plan
        plan = build_plan(question, graph, cards)

        # Save plan
        save_plan(plan, Path(out))

        # Print summary
        click.echo("✅ Plan generated\n")
        click.echo(f"Intent: {plan['intent']['type']} (confidence: {plan['intent']['confidence']})")
        click.echo(f"\nEntities detected: {len(plan['entities_detected'])}")
        for entity in plan["entities_detected"]:
            tables = set(ref.split(".")[0] for ref in entity["mapped_to"])
            click.echo(f"  • {entity['name']:15s} → {', '.join(list(tables)[:3])}")

        if plan["metrics_requested"]:
            click.echo(f"\nMetrics requested: {len(plan['metrics_requested'])}")
            for metric in plan["metrics_requested"]:
                click.echo(f"  • {metric['aggregation']}({', '.join(metric['mapped_columns'])})")

        if plan["constraints"]:
            click.echo(f"\nConstraints: {len(plan['constraints'])}")
            for constraint in plan["constraints"]:
                click.echo(f"  • {constraint['type']}: {constraint['expression']}")

        click.echo(f"\nSubquestions: {len(plan['subquestions'])}")
        for sq in plan["subquestions"]:
            click.echo(f"  [{sq['id']}] {sq['description']}")
            click.echo(f"      Tables: {', '.join(sq['tables'])}")
            click.echo(f"      Columns: {', '.join(sq['columns'][:5])}")
            if sq["columns"][5:]:
                click.echo(f"              ... and {len(sq['columns'][5:])} more")
            if sq.get("group_by"):
                click.echo(f"      Group by: {', '.join(sq['group_by'])}")
            if sq.get("aggregations"):
                aggs_str = ", ".join(f"{a['agg']}({a['col']})" for a in sq["aggregations"])
                click.echo(f"      Aggregations: {aggs_str}")

        if plan["join_paths"]:
            click.echo(f"\nJoin paths: {len(plan['join_paths'])}")
            for jp in plan["join_paths"][:5]:
                click.echo(
                    f"  • {jp['from']:12s} → {jp['to']:12s} "
                    f"via {', '.join(jp['via'])} "
                    f"(conf: {jp['confidence']:.2f}, {jp['cardinality']})"
                )

        if plan["ambiguities"]:
            click.echo(f"\n⚠️  Ambiguities: {len(plan['ambiguities'])}")
            for amb in plan["ambiguities"]:
                click.echo(f"  • {amb['issue']}")
                click.echo(f"    Options: {', '.join(amb['options'])}")
                click.echo(f"    Recommended: {amb['recommended']}")

        click.echo(f"\nOverall confidence: {plan['plan_confidence']:.2f}")
        click.echo(f"\nPlan saved to: {Path(out).absolute()}")

        # Execute if requested
        if execute:
            click.echo("\n" + "=" * 70)
            click.echo("Executing plan...\n")
            try:
                result = execute_plan(plan, Path(db_path))
                result_path = Path(out).parent / "result.json"
                save_result(result, result_path)

                click.echo("✅ Execution complete\n")
                click.echo(result["final_summary"])
                click.echo(f"\nResult saved to: {result_path.absolute()}")
            except Exception as exec_e:
                click.echo(f"❌ Error during execution: {exec_e}", err=True)
        else:
            # Hint about execution
            click.echo("\n💡 This is a plan only - no data has been queried yet.")
            click.echo("    Run with --execute to execute immediately, or use 'haikugraph run'")

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        click.echo("\nRun 'haikugraph graph build' first to generate the graph.", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Error generating plan: {e}", err=True)
        raise click.Abort()


@main.command("ask-llm")
@click.option(
    "--question",
    "-q",
    required=True,
    type=str,
    help="Natural language question about your data",
)
@click.option(
    "--db-path",
    default="./data/haikugraph.duckdb",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to DuckDB database (default: ./data/haikugraph.duckdb)",
)
@click.option(
    "--out",
    default="./data/plan_llm.json",
    type=click.Path(),
    help="Output plan file (default: ./data/plan_llm.json)",
)
@click.option(
    "--model",
    default=None,
    type=str,
    help="LLM model to use (default: gpt-4o-mini)",
)
@click.option(
    "--execute",
    is_flag=True,
    default=False,
    help="Execute the plan immediately after generation",
)
@click.option(
    "--interactive",
    is_flag=True,
    default=False,
    help="Interactively resolve ambiguities before execution",
)
def ask_llm(
    question: str, db_path: str, out: str, model: str | None, execute: bool, interactive: bool
):
    """Generate a plan using LLM and optionally execute it.

    Requires OPENAI_API_KEY environment variable to be set.
    Use --interactive to resolve ambiguities interactively.
    """
    try:
        click.echo(f"Generating plan with LLM for: {question}\n")

        # Generate plan using LLM
        plan = generate_plan(
            question=question,
            db_path=Path(db_path),
            model=model,
        )

        # Save plan
        save_plan(plan, Path(out))

        # Print summary
        click.echo("✅ LLM Plan generated\n")
        click.echo(
            f"Intent: {plan.get('intent', {}).get('type', 'unknown')} "
            f"(confidence: {plan.get('intent', {}).get('confidence', 0.0):.2f})"
        )

        if plan.get("entities_detected"):
            click.echo(f"\nEntities detected: {len(plan['entities_detected'])}")
            for entity in plan["entities_detected"][:5]:
                click.echo(f"  • {entity['name']}")

        if plan.get("metrics_requested"):
            click.echo(f"\nMetrics requested: {len(plan['metrics_requested'])}")
            for metric in plan["metrics_requested"][:5]:
                click.echo(f"  • {metric['name']} ({metric.get('aggregation', 'N/A')})")

        click.echo(f"\nSubquestions: {len(plan['subquestions'])}")
        for sq in plan["subquestions"]:
            click.echo(f"  [{sq['id']}] {sq.get('description', 'N/A')}")
            click.echo(f"      Tables: {', '.join(sq['tables'])}")

        if plan.get("ambiguities"):
            click.echo(f"\n⚠️  Ambiguities: {len(plan['ambiguities'])}")
            for amb in plan["ambiguities"]:
                click.echo(f"  • {amb['issue']}")
                click.echo(f"    Recommended: {amb.get('recommended', 'N/A')}")

        click.echo(f"\nOverall confidence: {plan.get('plan_confidence', 0.0):.2f}")
        click.echo(f"\nPlan saved to: {Path(out).absolute()}")

        # Interactive ambiguity resolution if requested
        if interactive:
            plan = _interactive_ambiguity_resolution(plan)
            # Save updated plan with resolutions
            save_plan(plan, Path(out))
            click.echo(f"\nUpdated plan with resolutions saved to: {Path(out).absolute()}")

        # Execute if requested
        if execute:
            click.echo("\n" + "=" * 70)
            click.echo("Executing plan...\n")
            try:
                result = execute_plan(plan, Path(db_path))
                result_path = Path(out).parent / "result_llm.json"
                save_result(result, result_path)

                click.echo("✅ Execution complete\n")
                click.echo(result["final_summary"])
                click.echo(f"\nResult saved to: {result_path.absolute()}")
            except Exception as exec_e:
                click.echo(f"❌ Error during execution: {exec_e}", err=True)
        else:
            click.echo("\n💡 This is a plan only - no data has been queried yet.")
            click.echo("    Run with --execute to execute immediately, or use 'haikugraph run'")

    except ValueError as e:
        click.echo(f"❌ Error: {e}", err=True)
        if "OPENAI_API_KEY" in str(e):
            click.echo("\nPlease set the OPENAI_API_KEY environment variable.", err=True)
            click.echo("Example: export OPENAI_API_KEY='your-api-key-here'", err=True)
        raise click.Abort()
    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        click.echo("\nRun 'haikugraph ingest' first to create the database.", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Error generating plan: {e}", err=True)
        raise click.Abort()


@main.command()
@click.option(
    "--plan",
    default="./data/plan.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to plan JSON (default: ./data/plan.json)",
)
@click.option(
    "--db-path",
    default="./data/haikugraph.duckdb",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to DuckDB database (default: ./data/haikugraph.duckdb)",
)
@click.option(
    "--out",
    default="./data/result.json",
    type=click.Path(),
    help="Output result file (default: ./data/result.json)",
)
def run(plan: str, db_path: str, out: str):
    """Execute a plan and query the database."""
    try:
        click.echo(f"Executing plan: {plan}\n")

        # Load plan
        with open(plan) as f:
            plan_data = json.load(f)

        # Execute plan
        result = execute_plan(plan_data, Path(db_path))

        # Save result
        save_result(result, Path(out))

        # Print summary
        click.echo("✅ Execution complete\n")
        click.echo(result["final_summary"])

        if result.get("resolutions"):
            click.echo(f"\n⚙️  Resolutions: {len(result['resolutions'])}")
            for res in result["resolutions"]:
                click.echo(f"  • {res['issue']}")
                click.echo(f"    Chose: {res['chosen']}")

        # Show applied resolutions summary (if present)
        applied_summary = result.get("applied_resolutions_summary", [])
        if applied_summary:
            click.echo(f"\n🧭 Applied Resolutions: {len(applied_summary)}")
            for line in applied_summary:
                click.echo(f"  • {line}")

        click.echo("\nSubquestion Results:")
        for sq_result in result["subquestion_results"]:
            status_icon = "✅" if sq_result["status"] == "success" else "❌"
            click.echo(
                f"  {status_icon} [{sq_result['id']}] "
                f"{sq_result.get('description', '')} - "
                f"{sq_result['row_count']} rows"
            )

            # Show preview of first few rows
            if sq_result["preview_rows"]:
                click.echo(f"\n  SQL: {sq_result.get('sql', 'N/A')}")
                click.echo(f"\n  Preview (first {min(5, len(sq_result['preview_rows']))} rows):")
                for i, row in enumerate(sq_result["preview_rows"][:5]):
                    if i == 0:
                        # Print header
                        click.echo(f"    {', '.join(row.keys())}")
                    # Print values
                    values = [str(v) for v in row.values()]
                    click.echo(f"    {', '.join(values)}")
                click.echo()

        click.echo(f"Result saved to: {Path(out).absolute()}")

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        click.echo("\nRun 'haikugraph ask' first to generate a plan.", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Error during execution: {e}", err=True)
        raise click.Abort()


@main.command()
@click.option(
    "--data-dir",
    default="./data",
    help="Data directory to check (default: ./data)",
)
@click.option(
    "--db-path",
    default="./data/haikugraph.duckdb",
    help="Database path to check (default: ./data/haikugraph.duckdb)",
)
def doctor(data_dir: str, db_path: str):
    """Check haikugraph environment and configuration."""
    click.echo("🔍 haikugraph doctor\n")

    # Python executable
    click.echo(f"Python executable: {sys.executable}")

    # haikugraph version
    click.echo(f"haikugraph version: {__version__}")

    # DuckDB version
    click.echo(f"DuckDB version: {duckdb.__version__}")

    # Data directory
    data_path = Path(data_dir)
    data_exists = data_path.exists() and data_path.is_dir()
    status_data = "✅" if data_exists else "❌"
    click.echo(f"\n{status_data} Data directory: {data_path.absolute()}")
    if data_exists:
        excel_files = list(data_path.glob("*.xlsx")) + list(data_path.glob("*.xls"))
        click.echo(f"   Excel files found: {len(excel_files)}")

    # Database path
    db_file = Path(db_path)
    db_exists = db_file.exists() and db_file.is_file()
    status_db = "✅" if db_exists else "❌"
    click.echo(f"\n{status_db} Database: {db_file.absolute()}")
    if db_exists:
        try:
            conn = duckdb.connect(str(db_file), read_only=True)
            tables = conn.execute("SHOW TABLES").fetchall()
            click.echo(f"   Tables: {len(tables)}")
            conn.close()
        except Exception as e:
            click.echo(f"   Error reading database: {e}")

    click.echo("\n✅ Environment check complete")


if __name__ == "__main__":
    main()
