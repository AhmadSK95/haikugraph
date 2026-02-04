"""Question to subquestion graph planner (deterministic, no LLM)."""

import json
import re
from pathlib import Path


def build_plan(question: str, graph: dict, cards: dict) -> dict:
    """
    Convert natural language question into structured subquestion plan.

    Args:
        question: Natural language question
        graph: Loaded graph.json structure
        cards: Dict with table_cards, column_cards, relation_cards

    Returns:
        Plan dict with subquestions, entities, metrics, and join paths
    """
    question_lower = question.lower()

    # Detect intent
    intent = detect_intent(question_lower)

    # Detect entities (tables, columns)
    entities = detect_entities(question_lower, graph, cards)

    # Detect requested metrics
    metrics = detect_metrics(question_lower, cards)

    # Detect constraints (filters)
    constraints = detect_constraints(question_lower, cards)

    # Build subquestions
    subquestions = build_subquestions(question_lower, entities, metrics, constraints, graph, cards)

    # Find required join paths
    join_paths = find_join_paths(subquestions, graph)

    # Detect ambiguities
    ambiguities = detect_ambiguities(entities, metrics, subquestions, cards)

    # Calculate overall plan confidence
    plan_confidence = calculate_plan_confidence(
        intent, entities, metrics, subquestions, ambiguities
    )

    return {
        "original_question": question,
        "intent": intent,
        "entities_detected": entities,
        "metrics_requested": metrics,
        "constraints": constraints,
        "subquestions": subquestions,
        "join_paths": join_paths,
        "ambiguities": ambiguities,
        "plan_confidence": plan_confidence,
    }


def detect_intent(question: str) -> dict:
    """Classify question intent using keyword matching."""
    intent_patterns = {
        "metric": [
            r"\btotal\b",
            r"\bsum\b",
            r"\bcount\b",
            r"\baverage\b",
            r"\bhow many\b",
            r"\bhow much\b",
        ],
        "comparison": [r"\bcompare\b", r"\bvs\b", r"\bdifference\b", r"\bbetween\b"],
        "diagnostic": [r"\bwhy\b", r"\bfailed\b", r"\berror\b", r"\bissue\b"],
        "lookup": [r"\bshow\b", r"\blist\b", r"\bfind\b", r"\bget\b"],
        "trend": [r"\bover time\b", r"\btrend\b", r"\bgrowth\b", r"\bchange\b"],
        "breakdown": [r"\bby\b", r"\bper\b", r"\bgroup\b", r"\bsegment\b"],
    }

    matches = {}
    for intent_type, patterns in intent_patterns.items():
        score = sum(1 for p in patterns if re.search(p, question))
        if score > 0:
            matches[intent_type] = score

    if not matches:
        return {"type": "lookup", "confidence": 0.3}

    # Return highest scoring intent
    best_intent = max(matches.items(), key=lambda x: x[1])
    confidence = min(0.95, 0.5 + (best_intent[1] * 0.15))

    return {"type": best_intent[0], "confidence": round(confidence, 2)}


def detect_entities(question: str, graph: dict, cards: dict) -> list[dict]:
    """Detect table/column entities referenced in question."""
    entities = []
    columns = graph["nodes"]["columns"]

    # Common entity keywords
    entity_keywords = {
        "customer": ["customer", "user", "client"],
        "transaction": ["transaction", "txn", "payment"],
        "payee": ["payee", "recipient", "beneficiary"],
        "quote": ["quote", "rate", "estimate"],
        "deal": ["deal", "booking"],
        "refund": ["refund", "reversal"],
    }

    for entity_name, keywords in entity_keywords.items():
        for keyword in keywords:
            if keyword in question:
                # Find matching columns
                matching_cols = []
                for col in columns:
                    col_name = col["column"].lower()
                    if entity_name in col_name or keyword in col_name:
                        matching_cols.append(f"{col['table']}.{col['column']}")

                if matching_cols:
                    # Calculate confidence based on keyword match strength
                    confidence = 0.9 if keyword == entity_name else 0.7
                    entities.append(
                        {
                            "name": entity_name,
                            "mapped_to": matching_cols[:5],  # Limit to 5
                            "confidence": confidence,
                        }
                    )
                    break  # Only one match per entity

    return entities


def detect_metrics(question: str, cards: dict) -> list[dict]:
    """Detect metric requests (aggregations)."""
    metrics = []

    # Aggregation patterns
    agg_patterns = {
        "sum": [r"\btotal\b", r"\bsum\b", r"\baggregate\b"],
        "count": [r"\bcount\b", r"\bnumber of\b", r"\bhow many\b"],
        "avg": [r"\baverage\b", r"\bmean\b", r"\bavg\b"],
        "max": [r"\bmaximum\b", r"\bmax\b", r"\bhighest\b"],
        "min": [r"\bminimum\b", r"\bmin\b", r"\blowest\b"],
    }

    # Money column patterns
    money_keywords = ["amount", "payment", "price", "cost", "value", "total"]

    for agg_type, patterns in agg_patterns.items():
        if any(re.search(p, question) for p in patterns):
            # Find money columns
            for col_card in cards.get("column_cards", []):
                col_name = col_card.get("column", "").lower()
                semantic_hints = col_card.get("semantic_hints", [])

                if (
                    any(kw in col_name for kw in money_keywords)
                    or "money_or_numeric" in semantic_hints
                ):
                    metric_name = f"{agg_type}_{col_card['column']}"
                    metrics.append(
                        {
                            "name": metric_name,
                            "mapped_columns": [col_card["column"]],
                            "aggregation": agg_type,
                            "confidence": 0.8,
                        }
                    )
                    break  # Only one metric per aggregation type

    return metrics


def detect_constraints(question: str, cards: dict) -> list[dict]:
    """Detect filter constraints in question."""
    constraints = []

    # Status constraints
    status_keywords = ["success", "failed", "pending", "completed", "active"]
    for keyword in status_keywords:
        if keyword in question:
            # Find status columns
            for col_card in cards.get("column_cards", []):
                if "status" in col_card.get("column", "").lower():
                    constraints.append(
                        {
                            "type": "status",
                            "expression": f"{col_card['column']} = '{keyword.upper()}'",
                            "confidence": 0.7,
                        }
                    )
                    break

    # Time constraints
    time_keywords = ["today", "yesterday", "week", "month", "year", "recent"]
    if any(kw in question for kw in time_keywords):
        for col_card in cards.get("column_cards", []):
            if "timestamp" in col_card.get("semantic_hints", []):
                constraints.append(
                    {
                        "type": "time",
                        "expression": f"{col_card['column']} >= DATE_SUB(NOW(), INTERVAL 1 MONTH)",
                        "confidence": 0.6,
                    }
                )
                break

    return constraints


def build_subquestions(
    question: str, entities: list, metrics: list, constraints: list, graph: dict, cards: dict
) -> list[dict]:
    """Build executable subquestions from detected components."""
    subquestions = []

    if not entities and not metrics:
        # Generic lookup
        # Pick first table with most columns
        tables = sorted(
            graph["nodes"]["tables"],
            key=lambda t: len(t.get("primary_key_candidates", [])),
            reverse=True,
        )
        if tables:
            table = tables[0]
            subquestions.append(
                {
                    "id": "SQ1",
                    "description": f"Retrieve data from {table['id']}",
                    "tables": [table["id"]],
                    "columns": table.get("primary_key_candidates", [])[:3],
                    "required_joins": [],
                    "confidence": 0.5,
                }
            )
        return subquestions

    # Build subquestion per entity-metric combination
    sq_id = 1
    for entity in entities:
        # Get primary table for this entity
        primary_tables = set()
        for col_ref in entity["mapped_to"]:
            table_name = col_ref.split(".")[0]
            primary_tables.add(table_name)

        for table_name in list(primary_tables)[:1]:  # Use first table
            # Get columns needed
            columns = []
            for col_ref in entity["mapped_to"]:
                if col_ref.startswith(table_name + "."):
                    columns.append(col_ref.split(".")[1])

            # Add metric columns
            for metric in metrics:
                columns.extend(metric["mapped_columns"])

            # Remove duplicates
            columns = list(dict.fromkeys(columns))

            description = f"Analyze {entity['name']}"
            if metrics:
                description += f" with {metrics[0]['aggregation']}"

            subquestions.append(
                {
                    "id": f"SQ{sq_id}",
                    "description": description,
                    "tables": [table_name],
                    "columns": columns[:10],  # Limit columns
                    "required_joins": [],
                    "confidence": round(min(entity["confidence"], 0.9), 2),
                }
            )
            sq_id += 1

    # If no subquestions yet but have metrics
    if not subquestions and metrics:
        for metric in metrics[:1]:  # One subquestion per metric
            # Find tables with this column
            matching_tables = []
            for col_card in cards.get("column_cards", []):
                if col_card["column"] in metric["mapped_columns"]:
                    matching_tables.append(col_card["table"])

            if matching_tables:
                table_name = matching_tables[0]
                subquestions.append(
                    {
                        "id": f"SQ{sq_id}",
                        "description": f"Compute {metric['name']}",
                        "tables": [table_name],
                        "columns": metric["mapped_columns"],
                        "required_joins": [],
                        "confidence": metric["confidence"],
                    }
                )
                sq_id += 1

    return subquestions


def find_join_paths(subquestions: list, graph: dict) -> list[dict]:
    """Find required join paths between tables in subquestions."""
    join_paths = []

    # Get all unique tables across subquestions
    all_tables = set()
    for sq in subquestions:
        all_tables.update(sq["tables"])

    if len(all_tables) <= 1:
        return []

    # Find edges connecting these tables
    edges = graph.get("edges", [])
    for edge in edges:
        if edge["source"] in all_tables and edge["target"] in all_tables:
            join_paths.append(
                {
                    "from": edge["source"],
                    "to": edge["target"],
                    "via": [edge["join_column"]],
                    "confidence": edge["confidence"],
                    "cardinality": edge["cardinality"],
                }
            )

    # Sort by confidence (prefer strong joins)
    join_paths.sort(key=lambda x: x["confidence"], reverse=True)

    return join_paths


def detect_ambiguities(
    entities: list, metrics: list, subquestions: list, cards: dict
) -> list[dict]:
    """Detect ambiguous mappings that need clarification."""
    ambiguities = []

    # Check for multi-table entity mappings
    for entity in entities:
        tables = set(ref.split(".")[0] for ref in entity["mapped_to"])
        if len(tables) > 1:
            ambiguities.append(
                {
                    "issue": f"Entity '{entity['name']}' found in multiple tables",
                    "options": list(tables),
                    "recommended": list(tables)[0],  # First by default
                    "confidence": 0.6,
                }
            )

    # Check for duplicate metric columns
    for metric in metrics:
        matching_tables = []
        for col_card in cards.get("column_cards", []):
            if col_card["column"] in metric["mapped_columns"]:
                matching_tables.append(col_card["table"])

        if len(set(matching_tables)) > 1:
            ambiguities.append(
                {
                    "issue": f"Multiple tables contain {metric['mapped_columns'][0]}",
                    "options": list(set(matching_tables)),
                    "recommended": matching_tables[0],
                    "confidence": 0.6,
                }
            )

    return ambiguities


def calculate_plan_confidence(
    intent: dict, entities: list, metrics: list, subquestions: list, ambiguities: list
) -> float:
    """Calculate overall plan confidence score."""
    scores = []

    # Intent confidence
    scores.append(intent["confidence"])

    # Entity confidence
    if entities:
        scores.append(sum(e["confidence"] for e in entities) / len(entities))

    # Metric confidence
    if metrics:
        scores.append(sum(m["confidence"] for m in metrics) / len(metrics))

    # Subquestion confidence
    if subquestions:
        scores.append(sum(sq["confidence"] for sq in subquestions) / len(subquestions))

    # Penalize for ambiguities
    ambiguity_penalty = len(ambiguities) * 0.05

    base_score = sum(scores) / len(scores) if scores else 0.3
    final_score = max(0.05, base_score - ambiguity_penalty)

    return round(final_score, 2)


def save_plan(plan: dict, output_path: Path) -> None:
    """Save plan to JSON file."""
    with open(output_path, "w") as f:
        json.dump(plan, f, indent=2)


def load_graph(graph_path: Path) -> dict:
    """Load graph.json."""
    with open(graph_path) as f:
        return json.load(f)


def load_cards_data(cards_dir: Path) -> dict:
    """Load all cards into memory."""
    from haikugraph.graph.build import load_cards

    table_cards, column_cards, relation_cards = load_cards(cards_dir)

    return {
        "table_cards": table_cards,
        "column_cards": column_cards,
        "relation_cards": relation_cards,
    }
