"""Question to subquestion graph planner (deterministic, no LLM) - HARDENED."""

import json
import re
from collections import deque
from pathlib import Path


def _sorted_unique(iterable) -> list[str]:
    """Return sorted unique list from any iterable."""
    return sorted(set(iterable))


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

    # Detect requested metrics WITH table qualification
    metrics = detect_metrics(question_lower, cards, question, entities)

    # Get context tables from entities and metrics
    context_tables = set()
    for entity in entities:
        for ref in entity["mapped_to"]:
            context_tables.add(ref.split(".")[0])
    for metric in metrics:
        for ref in metric["mapped_columns"]:
            if "." in ref:
                context_tables.add(ref.split(".")[0])
    
    # Determine primary table (first from metrics, then entities)
    primary_table = None
    if metrics and metrics[0]["mapped_columns"]:
        primary_table = metrics[0]["mapped_columns"][0].split(".")[0]
    elif entities and entities[0]["mapped_to"]:
        primary_table = entities[0]["mapped_to"][0].split(".")[0]

    # Detect constraints (filters) - context-aware
    constraints = detect_constraints(
        question_lower, cards, context_tables, intent, metrics, entities, primary_table
    )

    # Build subquestions with group_by support
    subquestions = build_subquestions(
        question_lower, entities, metrics, constraints, graph, cards, intent
    )

    # Comparison plans require scoped time markers per subquestion for schema validation.
    if intent.get("type") == "comparison" and subquestions:
        has_scoped_time = any(
            c.get("type") == "time" and c.get("applies_to")
            for c in constraints
        )
        if not has_scoped_time:
            for sq in subquestions:
                sq_id = sq.get("id")
                tf = sq.get("time_filter")
                if sq_id and tf and tf.get("period"):
                    constraints.append(
                        {
                            "type": "time",
                            "expression": str(tf["period"]),
                            "applies_to": sq_id,
                            "confidence": 0.9,
                        }
                    )

    # Find required join paths (BFS multi-hop)
    join_paths = find_join_paths(subquestions, graph)

    # Detect ambiguities
    ambiguities = detect_ambiguities(entities, metrics, subquestions, cards, question, constraints)

    # Calculate overall plan confidence
    plan_confidence = calculate_plan_confidence(
        intent, entities, metrics, subquestions, ambiguities
    )

    # Sort all lists for determinism
    entities = sorted(entities, key=lambda e: e["name"])
    for entity in entities:
        entity["mapped_to"] = _sorted_unique(entity["mapped_to"])

    for ambiguity in ambiguities:
        ambiguity["options"] = _sorted_unique(ambiguity["options"])

    # Sort join paths by confidence desc, then lexicographically
    join_paths = sorted(
        join_paths,
        key=lambda jp: (-jp["confidence"], jp["from"], jp["to"], ",".join(jp["via"])),
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

    for entity_name, keywords in sorted(entity_keywords.items()):
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
                            "mapped_to": _sorted_unique(matching_cols),
                            "confidence": confidence,
                        }
                    )
                    break  # Only one match per entity

    return entities


def detect_metrics(question: str, cards: dict, original_question: str, entities: list = None) -> list[dict]:
    """Detect metric requests with table-qualified columns and better ranking.
    
    Args:
        question: Lowercase question text
        cards: Schema cards
        original_question: Original question (for keyword matching)
        entities: Detected entities (for COUNT metric resolution)
    """
    metrics = []

    # Aggregation patterns
    agg_patterns = {
        "sum": [r"\btotal\b", r"\bsum\b", r"\baggregate\b", r"\brevenue\b", r"\bvolume\b", r"\bsales\b"],
        "count": [r"\bcount\b", r"\bnumber of\b", r"\bhow many\b"],
        "avg": [r"\baverage\b", r"\bmean\b", r"\bavg\b"],
        "max": [r"\bmaximum\b", r"\bmax\b", r"\bhighest\b"],
        "min": [r"\bminimum\b", r"\bmin\b", r"\blowest\b"],
    }

    # Domain keywords for ranking
    question_keywords = {
        "payment": ["payment", "pay", "paid"],
        "refund": ["refund", "reversal"],
        "quote": ["quote", "rate"],
    }

    for agg_type, patterns in agg_patterns.items():
        if any(re.search(p, question) for p in patterns):
            # For COUNT, prefer ID columns from entities, not money columns
            if agg_type == "count" and entities:
                # Use entity columns (like transaction_id)
                # IMPORTANT: Use DISTINCT to count unique entities, not all rows
                # (handles cases where one entity has multiple related rows)
                entity = entities[0]
                for ref in entity["mapped_to"]:
                    table, col = ref.split(".")
                    metrics.append({
                        "name": f"count_distinct_{col}",
                        "mapped_columns": [ref],
                        "aggregation": "count_distinct",  # Use count_distinct aggregation
                        "confidence": 0.8,
                    })
                    break  # Just use first entity column
                break  # Done with this agg type
            
            # For SUM/AVG/etc., score all candidate columns
            candidates = []
            for col_card in cards.get("column_cards", []):
                col_name = col_card.get("column", "").lower()
                table = col_card.get("table", "")
                semantic_hints = col_card.get("semantic_hints", [])

                # Skip if not money-like
                money_keywords = ["amount", "price", "cost", "value", "total", "charge"]
                is_timestamp_like = (
                    "timestamp" in semantic_hints
                    or col_name.endswith("_at")
                    or "date" in col_name
                    or "time" in col_name
                )
                is_money = any(kw in col_name for kw in money_keywords) or (
                    "money_or_numeric" in semantic_hints and not is_timestamp_like
                )
                if not is_money:
                    continue

                score = 0
                if "payment amount" in question and "payment_amount" in col_name:
                    score += 10
                if "amount" in question and "amount" in col_name:
                    score += 4
                # +3 if domain keyword in column name
                for domain, keywords in question_keywords.items():
                    if any(kw in original_question.lower() for kw in keywords):
                        if domain in col_name:
                            score += 3
                            break

                # +2 if semantic hint
                if "money_or_numeric" in semantic_hints:
                    score += 2

                # +1 if contains money keyword
                if any(kw in col_name for kw in money_keywords):
                    score += 1

                candidates.append(
                    {
                        "score": score,
                        "table": table,
                        "column": col_card["column"],
                        "qualified": f"{table}.{col_card['column']}",
                    }
                )

            # Pick best candidate
            if candidates:
                best = max(candidates, key=lambda c: (c["score"], c["qualified"]))
                metric_name = f"{agg_type}_{best['column']}"
                metrics.append(
                    {
                        "name": metric_name,
                        "mapped_columns": [best["qualified"]],
                        "aggregation": agg_type,
                        "confidence": 0.8,
                    }
                )
                break  # Only one metric per agg type

    # Fallback for implicit amount-style metrics (common in comparison prompts).
    if not metrics and any(token in question for token in ["amount", "revenue", "volume", "sales"]):
        candidates = []
        for col_card in cards.get("column_cards", []):
            col_name = col_card.get("column", "").lower()
            semantic_hints = col_card.get("semantic_hints", [])
            is_timestamp_like = (
                "timestamp" in semantic_hints
                or col_name.endswith("_at")
                or "date" in col_name
                or "time" in col_name
            )
            if "money_or_numeric" not in semantic_hints and not any(
                kw in col_name for kw in ["amount", "price", "cost", "value", "charge"]
            ):
                continue
            if is_timestamp_like:
                continue
            score = 0
            if "payment amount" in question and "payment_amount" in col_name:
                score += 10
            if "amount" in col_name:
                score += 2
            if "payment" in question and "payment" in col_name:
                score += 3
            if "money_or_numeric" in semantic_hints:
                score += 1
            qualified = f"{col_card.get('table', '')}.{col_card.get('column', '')}"
            candidates.append((score, qualified))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            _, best = candidates[0]
            metrics.append(
                {
                    "name": f"sum_{best.split('.')[-1]}",
                    "mapped_columns": [best],
                    "aggregation": "sum",
                    "confidence": 0.7,
                }
            )

    return metrics


def extract_time_periods(question: str) -> tuple[str, str] | None:
    """Extract two time periods from comparison questions.
    
    Returns:
        Tuple of (period1, period2) or None if not a time comparison
    """
    # Month patterns: "September 2025 vs October 2025" / "Sep 2025 vs Oct 2025"
    month_alias = {
        "jan": "january",
        "january": "january",
        "feb": "february",
        "february": "february",
        "mar": "march",
        "march": "march",
        "apr": "april",
        "april": "april",
        "may": "may",
        "jun": "june",
        "june": "june",
        "jul": "july",
        "july": "july",
        "aug": "august",
        "august": "august",
        "sep": "september",
        "sept": "september",
        "september": "september",
        "oct": "october",
        "october": "october",
        "nov": "november",
        "november": "november",
        "dec": "december",
        "december": "december",
    }
    month_pattern = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{4})"
    months = re.findall(month_pattern, question.lower())
    
    if len(months) >= 2:
        month1_raw, year1 = months[0]
        month2_raw, year2 = months[1]
        month1 = month_alias.get(month1_raw, month1_raw)
        month2 = month_alias.get(month2_raw, month2_raw)
        return (f"{month1}_{year1}", f"{month2}_{year2}")
    
    # Year patterns: "2024 vs 2025"
    year_pattern = r"\b(20\d{2})\b"
    years = re.findall(year_pattern, question)
    
    if len(years) >= 2:
        return (f"year_{years[0]}", f"year_{years[1]}")
    
    # Relative periods: "this month vs last month"
    if re.search(r"this\s+month.*(?:vs|versus).*last\s+month", question):
        return ("this_month", "last_month")
    
    if re.search(r"this\s+year.*(?:vs|versus).*last\s+year", question):
        return ("this_year", "last_year")
    
    return None


def detect_constraints(
    question: str,
    cards: dict,
    context_tables: set,
    intent: dict,
    metrics: list,
    entities: list,
    primary_table: str = None,
) -> list[dict]:
    """Detect filter constraints with context-aware status column selection.
    
    Args:
        primary_table: The primary table being queried (from metrics/entities), used to prioritize columns
    """
    constraints = []

    # Status constraints - context-aware
    status_keywords = ["success", "failed", "pending", "completed", "active"]
    for keyword in status_keywords:
        if keyword in question:
            # Score all status columns
            candidates = []
            for col_card in cards.get("column_cards", []):
                col_name = col_card.get("column", "").lower()
                table = col_card.get("table", "")

                if "status" not in col_name:
                    continue

                score = 0
                # +3 if column name contains question keyword
                question_keywords = ["payment", "refund", "quote", "transaction", "txn"]
                for kw in question_keywords:
                    if kw in question and kw in col_name:
                        score += 3
                        break

                # +2 if table in context
                if table in context_tables:
                    score += 2

                # +1 base for having "status"
                score += 1

                candidates.append(
                    {
                        "score": score,
                        "table": table,
                        "column": col_card["column"],
                    }
                )

            if candidates:
                # Pick highest score; tie-break by table, then column
                best = max(candidates, key=lambda c: (c["score"], c["table"], c["column"]))
                constraints.append(
                    {
                        "type": "status",
                        "expression": f"{best['table']}.{best['column']} = '{keyword.upper()}'",
                        "confidence": 0.7,
                    }
                )
                break

    # Time constraints - specific months
    month_pattern = r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
    month_match = re.search(month_pattern, question)
    
    if month_match:
        month_token = month_match.group(1)
        month_name = {
            "jan": "january",
            "january": "january",
            "feb": "february",
            "february": "february",
            "mar": "march",
            "march": "march",
            "apr": "april",
            "april": "april",
            "may": "may",
            "jun": "june",
            "june": "june",
            "jul": "july",
            "july": "july",
            "aug": "august",
            "august": "august",
            "sep": "september",
            "sept": "september",
            "september": "september",
            "oct": "october",
            "october": "october",
            "nov": "november",
            "november": "november",
            "dec": "december",
            "december": "december",
        }[month_token]
        month_num = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }[month_name]
        
        # Find timestamp column in context tables
        # Prioritize: primary_table > created_at > updated_at > other timestamps
        timestamp_candidates = []
        for col_card in cards.get("column_cards", []):
            if "timestamp" in col_card.get("semantic_hints", []):
                table = col_card.get("table", "")
                # Skip if we have context tables and this isn't one of them
                if context_tables and table not in context_tables:
                    continue
                col = col_card.get("column", "")
                # Score columns by preference
                score = 0
                if col == "created_at":
                    score = 3
                elif col == "updated_at":
                    score = 2
                elif "created" in col:
                    score = 1
                # STRONG preference for primary table if specified
                if primary_table and table == primary_table:
                    score += 100
                # Bonus if table is in context
                elif table in context_tables:
                    score += 10
                timestamp_candidates.append((score, table, col))
        
        if timestamp_candidates:
            # Pick highest scoring timestamp column
            timestamp_candidates.sort(reverse=True)
            _, table, col = timestamp_candidates[0]
            constraints.append({
                "type": "time_month",
                "expression": f"{table}.{col} month={month_num}",
                "month": month_num,
                "month_name": month_name,
                "column": col,
                "table": table,
                "confidence": 0.9,
            })
    
    # Time constraints - relative periods (today, this month, last week, etc.)
    # Check for specific patterns
    relative_time_patterns = {
        r"\btoday\b": "today",
        r"\byesterday\b": "yesterday",
        r"\bthis\s+week\b": "this_week",
        r"\blast\s+week\b": "last_week",
        r"\bthis\s+month\b": "this_month",
        r"\blast\s+month\b": "last_month",
        r"\bthis\s+year\b": "this_year",
        r"\blast\s+year\b": "last_year",
        r"\blast\s+(\d+)\s+days?\b": "last_N_days",  # e.g., "last 7 days", "last 30 days"
    }
    
    detected_period = None
    days_count = None
    
    # Find which pattern matches
    for pattern, period_name in relative_time_patterns.items():
        match = re.search(pattern, question)
        if match:
            detected_period = period_name
            # Extract days count for "last N days" pattern
            if period_name == "last_N_days":
                days_count = match.group(1)
            break
    
    if detected_period and not month_match:
        # Find timestamp column in context tables (same logic as month detection)
        timestamp_candidates = []
        for col_card in cards.get("column_cards", []):
            if "timestamp" in col_card.get("semantic_hints", []):
                table = col_card.get("table", "")
                # Skip if we have context tables and this isn't one of them
                if context_tables and table not in context_tables:
                    continue
                col = col_card.get("column", "")
                # Score columns by preference
                score = 0
                if col == "created_at":
                    score = 3
                elif col == "updated_at":
                    score = 2
                elif "created" in col:
                    score = 1
                # STRONG preference for primary table if specified
                if primary_table and table == primary_table:
                    score += 100
                # Bonus if table is in context
                elif table in context_tables:
                    score += 10
                timestamp_candidates.append((score, table, col))
        
        if timestamp_candidates:
            # Pick highest scoring timestamp column
            timestamp_candidates.sort(reverse=True)
            _, table, col = timestamp_candidates[0]
            
            # Build expression based on period type
            if days_count:
                expression = f"{table}.{col} in last_{days_count}_days"
            else:
                expression = f"{table}.{col} in {detected_period}"
            
            constraints.append({
                "type": "time_relative",
                "expression": expression,
                "period": detected_period,
                "days": days_count,
                "column": col,
                "table": table,
                "confidence": 0.8,
            })
    
    # Column value constraints (e.g., "with mt103", "where status = X")
    # Look for patterns like "with <value>" or "where <column> <value>"
    value_patterns = [
        r"\bwith\s+(\w+)",
        r"\bhas\s+(\w+)",
        r"\bcontaining\s+(\w+)",
    ]
    
    for pattern in value_patterns:
        match = re.search(pattern, question)
        if match:
            value = match.group(1)
            
            # Find columns that might contain this value
            # Look for columns with similar names or in the context
            for col_card in cards.get("column_cards", []):
                col_name = col_card.get("column", "").lower()
                table = col_card.get("table", "")
                
                # Check if value appears in column name (e.g., mt103 -> mt103_document_id)
                if value in col_name:
                    constraints.append({
                        "type": "value_filter",
                        "expression": f"{table}.{col_card['column']} IS NOT NULL",
                        "value": value,
                        "column": col_card['column'],
                        "table": table,
                        "confidence": 0.8,
                    })
                    break

    # Normalize explicit domain tokens even without "with/has" wording.
    # Example: "How many MT103 transactions in December?"
    if "mt103" in question and not any("mt103" in str(c.get("expression", "")).lower() for c in constraints):
        mt_candidates: list[tuple[int, str, str]] = []
        for col_card in cards.get("column_cards", []):
            col_name = col_card.get("column", "").lower()
            if "mt103" not in col_name:
                continue
            table = col_card.get("table", "")
            if context_tables and table not in context_tables:
                continue
            score = 0
            if "document_id" in col_name:
                score += 3
            if "created_at" in col_name:
                score += 2
            if primary_table and table == primary_table:
                score += 5
            mt_candidates.append((score, table, col_card["column"]))

        if mt_candidates:
            mt_candidates.sort(reverse=True)
            _, table, column = mt_candidates[0]
            constraints.append(
                {
                    "type": "value_filter",
                    "expression": f"{table}.{column} IS NOT NULL",
                    "operator": "is_not_null",
                    "column": column,
                    "table": table,
                    "confidence": 0.85,
                }
            )

    return constraints


def build_subquestions(
    question: str,
    entities: list,
    metrics: list,
    constraints: list,
    graph: dict,
    cards: dict,
    intent: dict,
) -> list[dict]:
    """Build executable subquestions with group_by support for breakdown queries."""
    subquestions = []

    # Check if this is a comparison query
    if intent["type"] == "comparison":
        time_periods = extract_time_periods(question)
        
        if time_periods and metrics:
            # Build two subquestions for comparison
            period1, period2 = time_periods
            
            # Get metric details
            metric = metrics[0]
            table_name = metric["mapped_columns"][0].split(".")[0]
            col_name = metric["mapped_columns"][0].split(".")[1]
            
            # Find timestamp column in this table
            timestamp_col = None
            ts_candidates: list[tuple[int, str]] = []
            for col_card in cards.get("column_cards", []):
                if (col_card.get("table") == table_name and 
                    "timestamp" in col_card.get("semantic_hints", [])):
                    ts_col_name = col_card.get("column", "")
                    score = 0
                    if ts_col_name == "created_at":
                        score += 5
                    elif ts_col_name == "updated_at":
                        score += 3
                    elif "created" in ts_col_name:
                        score += 2
                    else:
                        score += 1
                    ts_candidates.append((score, ts_col_name))
            if ts_candidates:
                ts_candidates.sort(reverse=True)
                timestamp_col = ts_candidates[0][1]
            
            # Create SQ1_current (first period)
            subquestions.append({
                "id": "SQ1_current",
                "description": f"Compute {metric['aggregation']}({col_name}) for {period1}",
                "tables": [table_name],
                "columns": [col_name],
                "aggregations": [{"agg": metric["aggregation"], "col": col_name}],
                "time_filter": {"column": timestamp_col, "period": period1} if timestamp_col else None,
                "required_joins": [],
                "confidence": metric["confidence"],
            })
            
            # Create SQ2_comparison (second period)
            subquestions.append({
                "id": "SQ2_comparison",
                "description": f"Compute {metric['aggregation']}({col_name}) for {period2}",
                "tables": [table_name],
                "columns": [col_name],
                "aggregations": [{"agg": metric["aggregation"], "col": col_name}],
                "time_filter": {"column": timestamp_col, "period": period2} if timestamp_col else None,
                "required_joins": [],
                "confidence": metric["confidence"],
            })
            
            return subquestions
    
    # Check if this is a breakdown/aggregation query
    has_breakdown = intent["type"] in ["breakdown", "metric"] and (
        re.search(r"\bby\b|\bper\b", question)
    )

    if not entities and not metrics:
        # Generic lookup
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
                    "columns": sorted(table.get("primary_key_candidates", [])[:3]),
                    "required_joins": [],
                    "confidence": 0.5,
                }
            )
        return subquestions

    # For breakdown queries, try to create a single grouped subquestion
    if has_breakdown and entities and metrics:
        metric = metrics[0] if metrics else None

        if metric:
            metric_table = metric["mapped_columns"][0].split(".")[0]
            metric_col = metric["mapped_columns"][0].split(".")[1]

            # ── Collect ALL requested dimensions, not just the first entity ──
            # 1. Start with entity-based dimensions (existing logic)
            group_cols: list[str] = []
            group_col_set: set[str] = set()
            for ent in entities:
                ent_tables = set(ref.split(".")[0] for ref in ent["mapped_to"])
                if metric_table in ent_tables:
                    col = next(
                        (
                            ref.split(".")[1]
                            for ref in ent["mapped_to"]
                            if ref.startswith(metric_table + ".")
                        ),
                        None,
                    )
                    if col and col not in group_col_set and col != metric_col:
                        group_cols.append(col)
                        group_col_set.add(col)

            # 2. Detect additional dimension keywords from the question by
            #    matching tokens against actual column names in the metric table.
            columns = graph["nodes"]["columns"]
            table_cols = [c for c in columns if c["table"] == metric_table]
            q_tokens = set(re.findall(r"[a-z][a-z0-9_]*", question))

            # Also detect "month", "week", "year", "day" as time dimensions
            time_dim_map = {
                "month": "__month__",
                "monthly": "__month__",
                "week": "__week__",
                "weekly": "__week__",
                "year": "__year__",
                "yearly": "__year__",
                "day": "__day__",
                "daily": "__day__",
                "quarter": "__quarter__",
                "quarterly": "__quarter__",
            }
            for token in q_tokens:
                if token in time_dim_map and time_dim_map[token] not in group_col_set:
                    group_cols.append(time_dim_map[token])
                    group_col_set.add(time_dim_map[token])

            # Match question tokens against column names (fuzzy: token in column name)
            for col_info in table_cols:
                col_name = col_info["column"].lower()
                for token in q_tokens:
                    if len(token) >= 4 and token in col_name and col_name not in group_col_set and col_name != metric_col:
                        group_cols.append(col_name)
                        group_col_set.add(col_name)
                        break

            if group_cols:
                dim_names = ", ".join(group_cols)
                all_cols = sorted(set(group_cols + [metric_col]))
                subquestions.append(
                    {
                        "id": "SQ1",
                        "description": (
                            f"Compute {metric['aggregation']}({metric_col}) "
                            f"per {dim_names}"
                        ),
                        "tables": [metric_table],
                        "columns": all_cols,
                        "group_by": group_cols,
                        "aggregations": [{"agg": metric["aggregation"], "col": metric_col}],
                        "required_joins": [],
                        "confidence": round(
                            min((e["confidence"] for e in entities), default=0.7) * 0.9, 2
                        ),
                    }
                )
                return subquestions

    # Otherwise, build subquestion per entity
    sq_id = 1
    for entity in entities:
        # Get primary table for this entity
        primary_tables = sorted(set(ref.split(".")[0] for ref in entity["mapped_to"]))

        for table_name in primary_tables[:1]:  # Use first table
            # Get columns needed
            columns = []
            for col_ref in entity["mapped_to"]:
                if col_ref.startswith(table_name + "."):
                    columns.append(col_ref.split(".")[1])

            # Check if we have metrics for this table
            aggregations = []
            for metric in metrics:
                for mcol in metric["mapped_columns"]:
                    if mcol.startswith(table_name + "."):
                        col = mcol.split(".")[1]
                        columns.append(col)
                        aggregations.append({"agg": metric["aggregation"], "col": col})

            # Remove duplicates and sort
            columns = _sorted_unique(columns)

            description = f"Analyze {entity['name']}"
            if metrics:
                description += f" with {metrics[0]['aggregation']}"

            sq_dict = {
                "id": f"SQ{sq_id}",
                "description": description,
                "tables": [table_name],
                "columns": columns[:10],  # Limit columns
                "required_joins": [],
                "confidence": round(min(entity["confidence"], 0.9), 2),
            }
            
            # Add aggregations if present
            if aggregations:
                sq_dict["aggregations"] = aggregations
            
            subquestions.append(sq_dict)
            sq_id += 1

    # If no subquestions yet but have metrics
    if not subquestions and metrics:
        for metric in metrics[:1]:
            # Get table from qualified column
            table_name = metric["mapped_columns"][0].split(".")[0]
            col_name = metric["mapped_columns"][0].split(".")[1]

            subquestions.append(
                {
                    "id": f"SQ{sq_id}",
                    "description": f"Compute {metric['name']}",
                    "tables": [table_name],
                    "columns": [col_name],
                    "aggregations": [{"agg": metric["aggregation"], "col": col_name}],
                    "required_joins": [],
                    "confidence": metric["confidence"],
                }
            )
            sq_id += 1

    return subquestions


def find_join_paths(subquestions: list, graph: dict) -> list[dict]:
    """Find join paths using BFS for multi-hop support."""
    join_paths = []

    # Get all unique tables across subquestions
    all_tables = set()
    for sq in subquestions:
        all_tables.update(sq["tables"])

    if len(all_tables) <= 1:
        return []

    # Build adjacency list from edges
    adj = {}
    edge_info = {}
    for edge in graph.get("edges", []):
        src, tgt = edge["source"], edge["target"]
        if src not in adj:
            adj[src] = []
        if tgt not in adj:
            adj[tgt] = []
        adj[src].append(tgt)
        adj[tgt].append(src)  # Undirected

        # Store edge info for both directions
        key1 = (src, tgt)
        key2 = (tgt, src)
        edge_info[key1] = edge
        edge_info[key2] = edge

    # BFS to find shortest paths between all table pairs
    table_list = sorted(all_tables)
    for i, start in enumerate(table_list):
        for end in table_list[i + 1 :]:
            path = bfs_shortest_path(start, end, adj)
            if path:
                # Extract join info from path
                via_cols = []
                min_confidence = 1.0
                cardinality = "multi_hop" if len(path) > 2 else None

                for j in range(len(path) - 1):
                    edge_key = (path[j], path[j + 1])
                    if edge_key in edge_info:
                        edge = edge_info[edge_key]
                        via_cols.append(edge["join_column"])
                        min_confidence = min(min_confidence, edge["confidence"])
                        if cardinality is None:
                            cardinality = edge["cardinality"]

                join_paths.append(
                    {
                        "from": path[0],
                        "to": path[-1],
                        "via": via_cols,
                        "confidence": round(min_confidence, 2),
                        "cardinality": cardinality,
                    }
                )

    return join_paths


def bfs_shortest_path(start: str, end: str, adj: dict) -> list[str] | None:
    """BFS to find shortest path between two nodes."""
    if start not in adj or end not in adj:
        return None

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()
        if node == end:
            return path

        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


def detect_ambiguities(
    entities: list, metrics: list, subquestions: list, cards: dict, question: str = "", constraints: list = None
) -> list[dict]:
    """Detect ambiguous mappings that need clarification.
    
    Args:
        entities: Detected entities
        metrics: Detected metrics
        subquestions: Built subquestions
        cards: Schema cards
        question: Original question (used to detect explicit table references)
        constraints: Detected constraints (for context-aware resolution)
    """
    ambiguities = []
    
    if constraints is None:
        constraints = []

    # Get metric tables and constraint tables for context-aware resolution
    metric_tables = set()
    for metric in metrics:
        for mcol in metric["mapped_columns"]:
            if "." in mcol:
                metric_tables.add(mcol.split(".")[0])
    
    constraint_tables = set()
    for constraint in constraints:
        if "table" in constraint:
            constraint_tables.add(constraint["table"])
    
    # Combine for total context
    context_tables = metric_tables | constraint_tables

    # Check for multi-table entity mappings
    for entity in entities:
        tables = _sorted_unique(ref.split(".")[0] for ref in entity["mapped_to"])
        if len(tables) > 1:
            # Check if user explicitly mentioned a table in the question
            recommended = _find_explicit_table_reference(question.lower(), tables)
            if not recommended:
                # Prefer tables that contain the metrics or constraints being queried
                if context_tables:
                    matching = [t for t in tables if t in context_tables]
                    if matching:
                        recommended = matching[0]
                        confidence = 0.85  # High confidence based on context
                    else:
                        recommended = tables[0]  # First alphabetically as fallback
                        confidence = 0.6
                else:
                    recommended = tables[0]  # First alphabetically as fallback
                    confidence = 0.6
            else:
                confidence = 0.9  # High confidence when explicitly mentioned
            
            ambiguities.append(
                {
                    "issue": f"Entity '{entity['name']}' found in multiple tables",
                    "options": tables,
                    "recommended": recommended,
                    "confidence": confidence,
                }
            )

    # Check for duplicate metric columns
    for metric in metrics:
        matching_tables = []
        for mcol in metric["mapped_columns"]:
            if "." in mcol:
                matching_tables.append(mcol.split(".")[0])

        # Also check unqualified scenario
        for col_card in cards.get("column_cards", []):
            for mcol in metric["mapped_columns"]:
                col_name = mcol.split(".")[-1] if "." in mcol else mcol
                if col_card["column"] == col_name:
                    matching_tables.append(col_card["table"])

        unique_tables = _sorted_unique(matching_tables)
        if len(unique_tables) > 1:
            # Check if user explicitly mentioned a table in the question
            recommended = _find_explicit_table_reference(question.lower(), unique_tables)
            if not recommended:
                recommended = unique_tables[0]  # First alphabetically as fallback
                confidence = 0.6
            else:
                confidence = 0.9  # High confidence when explicitly mentioned
            
            ambiguities.append(
                {
                    "issue": f"Multiple tables contain {metric['mapped_columns'][0]}",
                    "options": unique_tables,
                    "recommended": recommended,
                    "confidence": confidence,
                }
            )

    return ambiguities


def _find_explicit_table_reference(question: str, table_options: list[str]) -> str | None:
    """Check if user explicitly mentioned one of the table options in their question.
    
    Args:
        question: Lowercase question text
        table_options: List of possible table names
    
    Returns:
        Table name if found, None otherwise
    """
    for table in table_options:
        # Check for exact table name or common patterns like "from table_name"
        if table.lower() in question:
            return table
    return None


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
        json.dump(plan, f, indent=2, sort_keys=True)


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
