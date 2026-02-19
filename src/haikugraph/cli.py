"""CLI entrypoint for haikugraph."""

import json
import os
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
from haikugraph.io.smart_ingest import (
    smart_ingest_excel_to_duckdb,
    format_smart_ingestion_summary,
)
from haikugraph.io.document_ingest import ingest_documents_to_duckdb
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


def _parse_sheet_arg(sheet: str | None) -> str | int | None:
    """Convert numeric sheet index strings to int for pandas."""
    if sheet is None:
        return None
    try:
        return int(sheet)
    except ValueError:
        return sheet


def _run_unified_ingest(data_dir: str, db_path: str, sheet: str | None, force: bool) -> None:
    """Run the single supported ingest pipeline (smart ingest)."""
    sheet_arg = _parse_sheet_arg(sheet)
    results = smart_ingest_excel_to_duckdb(
        data_dir=Path(data_dir),
        db_path=Path(db_path),
        sheet=sheet_arg,
        force=force,
    )
    click.echo(format_smart_ingestion_summary(results))
    if results["status"] == "failed":
        raise click.Abort()


@main.command()
@click.option(
    "--data-dir",
    default="./data",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory containing Excel files (default: ./data)",
)
@click.option(
    "--db-path",
    default="./data/haikugraph.db",
    type=click.Path(),
    help="Path to DuckDB database file (default: ./data/haikugraph.db)",
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
    """Ingest Excel files into DuckDB using the unified smart-ingest pipeline."""
    _run_unified_ingest(data_dir, db_path, sheet, force)


@main.command("ingest-docs")
@click.option(
    "--docs-dir",
    default="./data",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory containing document files (.txt/.md/.pdf/.docx).",
)
@click.option(
    "--db-path",
    default="./data/haikugraph.db",
    type=click.Path(),
    help="Target DuckDB path (default: ./data/haikugraph.db).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="If set, clears existing datada_documents before ingesting.",
)
def ingest_docs(docs_dir: str, db_path: str, force: bool):
    """Ingest text-heavy documents into datada_documents for evidence retrieval."""
    result = ingest_documents_to_duckdb(
        docs_dir=Path(docs_dir),
        db_path=Path(db_path),
        force=force,
    )
    if result.get("success"):
        click.echo(f"✅ {result.get('message')}")
    else:
        click.echo(f"❌ {result.get('message')}", err=True)
        raise click.Abort()


@main.command("use-db")
@click.option(
    "--db-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to an existing DuckDB database file.",
)
@click.option(
    "--env-file",
    default=".env",
    type=click.Path(file_okay=True, dir_okay=False),
    help="Env file to update with HG_DB_PATH (default: ./.env)",
)
def use_db(db_path: str, env_file: str):
    """Use an existing DuckDB database directly (skip Excel ingestion)."""
    source = Path(db_path).expanduser().resolve()

    # Validate DB readability early.
    try:
        conn = duckdb.connect(str(source), read_only=True)
        conn.execute("SELECT 1").fetchone()
        conn.close()
    except Exception as exc:
        click.echo(f"❌ Unable to read DuckDB file at {source}: {exc}", err=True)
        raise click.Abort()

    env_path = Path(env_file)
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    entry = f"HG_DB_PATH={source}"
    updated = False
    out_lines: list[str] = []
    for line in lines:
        if line.startswith("HG_DB_PATH="):
            out_lines.append(entry)
            updated = True
        else:
            out_lines.append(line)
    if not updated:
        out_lines.append(entry)

    env_path.write_text("\n".join(out_lines).strip() + "\n")

    click.echo("✅ dataDa is now configured to use an existing DuckDB source.")
    click.echo(f"DB: {source}")
    click.echo(f"Updated: {env_path}")
    click.echo("Next: restart the API/UI (`./run.sh`) so the new DB path is picked up.")


@main.command()
@click.option(
    "--db-path",
    default="./data/haikugraph.db",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to DuckDB database (default: ./data/haikugraph.db)",
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
    default="./data/haikugraph.db",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to DuckDB database (default: ./data/haikugraph.db)",
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
    default="./data/haikugraph.db",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to DuckDB database (required if --execute is used)",
)
@click.option(
    "--use-llm-resolver",
    is_flag=True,
    default=False,
    help="Enable LLM resolver for ambiguous mentions (requires API key)",
)
@click.option(
    "--followup",
    is_flag=True,
    default=False,
    help="Treat as follow-up question (loads previous plan for context)",
)
def ask(question: str, graph_path: str, cards_dir: str, out: str, execute: bool, db_path: str, use_llm_resolver: bool, followup: bool):
    """Plan how to answer a question (deterministic + optional LLM resolver)."""
    try:
        click.echo(f"Planning how to answer: {question}\n")

        # Load graph and cards
        graph = load_graph(Path(graph_path))
        cards = load_cards_data(Path(cards_dir))

        # Auto-detect follow-up questions only when explicitly enabled.
        auto_followup_enabled = os.environ.get("HG_AUTO_FOLLOWUP", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        prev_plan_path = Path(out)
        if auto_followup_enabled and not followup and prev_plan_path.exists():
            from haikugraph.planning.followups import classify_followup
            
            try:
                with open(prev_plan_path) as f:
                    prev_plan = json.load(f)
                
                prev_question = prev_plan.get("original_question", "")
                if prev_question:
                    classification = classify_followup(question, prev_question, prev_plan)
                    
                    # Auto-enable followup if confidence is high enough
                    if classification["is_followup"] and classification["confidence"] >= 0.8:
                        followup = True
                        click.echo(f"🔍 Auto-detected as follow-up: {classification['type']} (confidence: {classification['confidence']:.2f})\n")
            except Exception:
                # If auto-detection fails, proceed as new question
                pass

        # Handle followup questions
        if followup:
            from haikugraph.planning.followups import classify_followup, patch_plan
            
            # Load previous plan
            prev_plan_path = Path(out)
            if not prev_plan_path.exists():
                click.echo("⚠️  No previous plan found. Building new plan instead.", err=True)
                followup = False
            else:
                with open(prev_plan_path) as f:
                    prev_plan = json.load(f)
                
                prev_question = prev_plan.get("original_question", "")
                
                # Classify if this is actually a followup
                classification = classify_followup(question, prev_question, prev_plan)
                
                if classification["is_followup"]:
                    click.echo(f"🔗 Follow-up detected: {classification['type']} (confidence: {classification['confidence']:.2f})")
                    plan = patch_plan(prev_plan, classification, question, cards, use_llm=use_llm_resolver)
                    click.echo("   ✅ Plan patched with previous context\n")
                else:
                    click.echo("⚠️  Question doesn't seem like a follow-up. Building new plan.\n")
                    followup = False
        
        # Build new plan if not followup
        if not followup:
            plan = build_plan(question, graph, cards)
        
        # Enhance with LLM resolver if requested
        if use_llm_resolver:
            from haikugraph.planning.llm_resolver import enhance_plan_with_llm
            click.echo("🔍 Using LLM resolver for ambiguous mentions...")
            plan = enhance_plan_with_llm(plan, cards, enable_llm=True)
            if plan.get("llm_resolutions"):
                click.echo(f"   ✅ LLM resolved {len(plan['llm_resolutions'])} mentions\n")

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
    default="./data/haikugraph.db",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to DuckDB database (default: ./data/haikugraph.db)",
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


@main.command("ask-a6")
@click.option(
    "--question",
    "-q",
    required=True,
    type=str,
    help="Natural language question about your data",
)
@click.option(
    "--db-path",
    default="./data/haikugraph.db",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to DuckDB database (default: ./data/haikugraph.db)",
)
@click.option(
    "--out",
    default="./data/plan_a6.json",
    type=click.Path(),
    help="Output plan file (default: ./data/plan_a6.json)",
)
def ask_a6(question: str, db_path: str, out: str):
    """Generate and execute plan using local Ollama models (A6 POC).

    Uses split LLM approach:
    - Planner LLM: generates strict JSON Plan
    - Narrator LLM: explains results in natural language

    Requires Ollama to be running with models pulled.
    Set HG_PLANNER_MODEL and HG_NARRATOR_MODEL env vars to customize.
    """
    try:
        from haikugraph.llm.plan_generator import introspect_schema
        from haikugraph.planning.llm_planner import generate_or_patch_plan
        from haikugraph.explain.narrator import narrate

        click.echo(f"🤖 Generating plan with Ollama for: {question}\n")

        # Introspect schema
        schema_text = introspect_schema(Path(db_path))

        # Generate plan using Ollama planner
        plan = generate_or_patch_plan(
            question=question,
            schema=schema_text,
        )

        # Save plan
        save_plan(plan, Path(out))

        # Print plan summary
        click.echo("✅ Plan generated with Ollama\n")
        click.echo(f"Subquestions: {len(plan['subquestions'])}")
        for sq in plan["subquestions"]:
            click.echo(f"  [{sq['id']}] Tables: {', '.join(sq['tables'])}")

        if plan.get("constraints"):
            click.echo(f"\nConstraints: {len(plan['constraints'])}")
            for constraint in plan["constraints"]:
                applies_to = constraint.get("applies_to", "all")
                click.echo(
                    f"  • {constraint['type']}: {constraint['expression']} "
                    f"(applies_to: {applies_to})"
                )

        click.echo(f"\nPlan saved to: {Path(out).absolute()}")

        # Execute plan
        click.echo("\n" + "=" * 70)
        click.echo("Executing plan...\n")

        result = execute_plan(plan, Path(db_path))

        # Format results and metadata for narrator
        results_dict = {}
        meta_dict = {}
        for sq_result in result["subquestion_results"]:
            sq_id = sq_result["id"]
            results_dict[sq_id] = sq_result.get("preview_rows", [])
            meta_dict[sq_id] = sq_result.get("metadata", {})

        # Generate narrative explanation
        click.echo("Generating explanation...\n")
        explanation = narrate(
            question=question,
            plan=plan,
            results=results_dict,
            meta=meta_dict,
            subquestion_results=result["subquestion_results"],
        )

        # Print explanation
        click.echo("=" * 70)
        click.echo("📊 Explanation:\n")
        click.echo(explanation)
        click.echo("\n" + "=" * 70)

        # Save result
        result_path = Path(out).parent / "result_a6.json"
        save_result(result, result_path)
        click.echo(f"\nResult saved to: {result_path.absolute()}")

    except ConnectionError as e:
        import os
        from haikugraph.llm.router import DEFAULT_PLANNER_MODEL, DEFAULT_NARRATOR_MODEL
        
        click.echo(f"❌ Connection Error: {e}", err=True)
        click.echo("\nMake sure Ollama is running:", err=True)
        click.echo("  macOS: Start Ollama app or run 'ollama serve'", err=True)
        
        # Show actual model names from env vars (or defaults)
        planner_model = os.environ.get("HG_PLANNER_MODEL", DEFAULT_PLANNER_MODEL)
        narrator_model = os.environ.get("HG_NARRATOR_MODEL", DEFAULT_NARRATOR_MODEL)
        
        click.echo("\nPull required models:", err=True)
        click.echo(f"  ollama pull {planner_model}", err=True)
        click.echo(f"  ollama pull {narrator_model}", err=True)
        click.echo("\nOr set custom models with env vars:", err=True)
        click.echo("  export HG_PLANNER_MODEL=your-planner-model", err=True)
        click.echo("  export HG_NARRATOR_MODEL=your-narrator-model", err=True)
        raise click.Abort()
    except ValueError as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()
    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        click.echo("\nRun 'haikugraph ingest' first to create the database.", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        import traceback
        traceback.print_exc()
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
    default="./data/haikugraph.db",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to DuckDB database (default: ./data/haikugraph.db)",
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
    default="./data/haikugraph.db",
    help="Database path to check (default: ./data/haikugraph.db)",
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


@main.group()
def rules():
    """Manage data rules configuration."""
    pass


@rules.command("show")
@click.option(
    "--data-dir",
    default="./data",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Data directory containing rules.yaml (default: ./data)",
)
def rules_show(data_dir: str):
    """Display current data rules configuration."""
    from haikugraph.rules import load_rules
    
    try:
        config = load_rules(Path(data_dir) / "rules.yaml")
        if not config:
            click.echo("No rules loaded (rules.yaml missing or empty).")
            return
        click.echo("✅ Rules loaded\n")
        click.echo(json.dumps(config, indent=2))
    except Exception as e:
        click.echo(f"❌ Error loading rules: {e}", err=True)
        raise click.Abort()


@rules.command("validate")
@click.option(
    "--rules-path",
    default="./data/rules.yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to rules file (default: ./data/rules.yaml)",
)
def rules_validate(rules_path: str):
    """Validate a rules configuration file."""
    from haikugraph.rules import load_rules

    config = load_rules(Path(rules_path))
    if not isinstance(config, dict):
        click.echo("❌ Invalid rules format: expected a mapping/object", err=True)
        raise click.Abort()

    for key in ("entity_rules", "column_rules", "global_rules"):
        value = config.get(key, {})
        if value is not None and not isinstance(value, dict):
            click.echo(f"❌ Invalid rules format: '{key}' must be an object", err=True)
            raise click.Abort()

    click.echo("✅ Rules file is valid.")


@rules.command("init")
@click.option(
    "--data-dir",
    default="./data",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Data directory for rules.yaml (default: ./data)",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing rules.yaml",
)
def rules_init(data_dir: str, force: bool):
    """Create a starter rules.yaml file with examples."""
    rules_path = Path(data_dir) / "rules.yaml"
    
    if rules_path.exists() and not force:
        click.echo(f"❌ Rules file already exists: {rules_path}", err=True)
        click.echo("   Use --force to overwrite.", err=True)
        raise click.Abort()
    
    # Create data directory if needed
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write starter template
    starter_content = '''# HaikuGraph Data Rules Configuration
# ====================================
# Define business rules that are automatically applied when querying data.
# See GETTING_STARTED.md for detailed documentation.

version: "1.0"

# Entity Rules - organize rules by business entity
entity_rules:
  # Example: transaction entity
  # transaction:
  #   description: "Financial transactions"
  #   tables:
  #     - transactions_table
  #   validity:
  #     - column: status_column
  #       condition: IS NOT NULL
  #       reason: "Only complete transactions are valid"
  #   default_filters:
  #     - column: is_test
  #       condition: "= false"
  #       override_keywords: ["test", "all data"]

# Column Rules - rules for specific columns
column_rules: {}

# Global Rules - apply to all queries
global_rules: {}
'''
    
    rules_path.write_text(starter_content)
    click.echo(f"✅ Created rules file: {rules_path}")
    click.echo("\n📝 Edit the file to add your business rules.")
    click.echo("   Run 'haikugraph rules show' to view current configuration.")


@main.command("ask-demo")
@click.argument("question")
@click.option(
    "--db-path",
    default="./data/haikugraph.db",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to DuckDB database",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Print intent + plan + SQL",
)
@click.option(
    "--no-intent",
    is_flag=True,
    default=False,
    help="Skip A8 intent classification",
)
@click.option(
    "--raw",
    is_flag=True,
    default=False,
    help="Print raw execution results only",
)
def ask_demo(question: str, db_path: str, debug: bool, no_intent: bool, raw: bool):
    """A10: End-to-end demo of Question → Intent → Plan → Execute → Narrate.
    
    This command demonstrates the complete HaikuGraph pipeline:
    1. Intent Classification (A8)
    2. Plan Generation (A6-A7)
    3. SQL Execution (A4-A5)
    4. Narration (A9)
    
    Example:
        haikugraph ask-demo "What is total revenue?"
        haikugraph ask-demo "Revenue by barber" --debug
        haikugraph ask-demo "Compare this month vs last month" --raw
    """
    import sys
    import traceback
    from pathlib import Path
    
    from haikugraph.planning.intent import classify_intent
    from haikugraph.llm.plan_generator import introspect_schema
    from haikugraph.planning.llm_planner import generate_or_patch_plan
    from haikugraph.execution import execute_plan
    from haikugraph.explain.narrator import narrate_results
    
    # Print question
    if not raw:
        click.echo("\n" + "=" * 70)
        click.echo(f"Question: {question}")
        click.echo("=" * 70)
    
    intent = None
    plan = None
    result = None
    
    try:
        # ======================================================================
        # STAGE 1: Intent Classification (A8)
        # ======================================================================
        if not no_intent:
            if debug:
                click.echo("\n[1/4] Classifying intent...")
            
            try:
                intent = classify_intent(question)
                
                if debug:
                    click.echo("\n✅ Intent:")
                    click.echo(f"  Type: {intent.type.value}")
                    click.echo(f"  Confidence: {intent.confidence:.2f}")
                    click.echo(f"  Rationale: {intent.rationale}")
                    click.echo(f"  Requires comparison: {intent.requires_comparison}")
            
            except Exception as e:
                # Intent classification failed - non-fatal
                if debug:
                    click.echo(f"\n⚠️  Intent classification failed (non-fatal): {e}", err=True)
                    click.echo("   Continuing without intent context...", err=True)
                intent = None
        
        # ======================================================================
        # STAGE 2: Plan Generation (A6-A7)
        # ======================================================================
        if debug:
            click.echo("\n[2/4] Generating plan...")
        
        try:
            # Introspect schema
            schema_text = introspect_schema(Path(db_path))
            
            # Generate plan
            plan = generate_or_patch_plan(
                question=question,
                schema=schema_text,
                intent=intent,
            )
            
            if debug:
                click.echo("\n✅ Plan:")
                click.echo(json.dumps(plan, indent=2))
        
        except Exception as e:
            # Planner failed - FATAL
            click.echo(f"\n❌ Planner failed: {e}", err=True)
            if debug:
                traceback.print_exc()
            sys.exit(1)
        
        # ======================================================================
        # STAGE 3: SQL Execution (A4-A5)
        # ======================================================================
        if debug:
            click.echo("\n[3/4] Executing SQL...")
        
        try:
            result = execute_plan(plan, Path(db_path))
            
            if debug:
                click.echo("\n✅ Execution Results:")
                for sq_result in result["subquestion_results"]:
                    sq_id = sq_result["id"]
                    status = sq_result["status"]
                    sql = sq_result.get("sql", "N/A")
                    
                    status_icon = "✅" if status == "success" else "❌"
                    click.echo(f"\n  {status_icon} {sq_id}:")
                    click.echo(f"     SQL: {sql}")
                    
                    if status == "success":
                        click.echo(f"     Rows: {sq_result['row_count']}")
                        if sq_result.get("preview_rows"):
                            click.echo(f"     Sample: {sq_result['preview_rows'][:3]}")
                    else:
                        click.echo(f"     Error: {sq_result.get('error', 'Unknown')}")  
        
        except Exception as e:
            # Execution failed - FATAL
            click.echo(f"\n❌ Execution failed: {e}", err=True)
            if debug:
                traceback.print_exc()
            sys.exit(2)
        
        # ======================================================================
        # RAW MODE: Just print results and exit
        # ======================================================================
        if raw:
            click.echo(json.dumps(result, indent=2, default=str))
            sys.exit(0)
        
        # ======================================================================
        # STAGE 4: Narration (A9)
        # ======================================================================
        if debug:
            click.echo("\n[4/4] Generating narrative explanation...")
        
        try:
            # Convert execution results to narrator format
            narrator_results = {}
            for sq_result in result["subquestion_results"]:
                sq_id = sq_result["id"]
                
                if sq_result["status"] == "success":
                    narrator_results[sq_id] = {
                        "rows": sq_result.get("preview_rows", []),
                        "columns": list(sq_result["preview_rows"][0].keys()) if sq_result.get("preview_rows") else [],
                        "row_count": sq_result["row_count"],
                    }
                else:
                    narrator_results[sq_id] = {
                        "rows": [],
                        "columns": [],
                        "row_count": 0,
                        "error": sq_result.get("error", "Unknown error"),
                    }
            
            # Call narrator
            explanation = narrate_results(
                original_question=question,
                intent=intent,
                plan=plan,
                results=narrator_results,
            )
            
            # Print answer
            click.echo("\n" + "="  * 70)
            click.echo("Answer:")
            click.echo("=" * 70)
            click.echo(f"\n{explanation}\n")
            
            sys.exit(0)
        
        except Exception as e:
            # Narration failed - FATAL (but show raw results)
            click.echo(f"\n❌ Narration failed: {e}", err=True)
            if debug:
                traceback.print_exc()
            
            # Fallback: show raw results
            click.echo("\n📊 Raw Results:", err=True)
            for sq_result in result["subquestion_results"]:
                sq_id = sq_result["id"]
                status_icon = "✅" if sq_result["status"] == "success" else "❌"
                click.echo(f"\n  {status_icon} {sq_id}: {sq_result['row_count']} rows", err=True)
                if sq_result.get("preview_rows"):
                    click.echo(f"     {sq_result['preview_rows'][:3]}", err=True)
            
            sys.exit(3)
    
    except KeyboardInterrupt:
        click.echo("\n\n⚠️  Interrupted by user", err=True)
        sys.exit(130)
    
    except Exception as e:
        click.echo(f"\n❌ Unexpected error: {e}", err=True)
        if debug:
            traceback.print_exc()
        sys.exit(255)


if __name__ == "__main__":
    main()
