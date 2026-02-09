"""LLM-powered resolver for ambiguous query mentions using Haiku cards.

This module resolves ambiguous mentions (e.g., "mt103", "high-value", "India") 
by using LLM with full schema card context to determine:
1. Is mention a column name, data value, or semantic concept?
2. Which column(s) should be used?
3. What filter operator to apply?
"""

import json
import re
from typing import Any

from haikugraph.llm.router import call_llm


def extract_non_column_mentions(question: str, cards: dict) -> list[dict]:
    """Extract mentions that don't directly map to column names.
    
    These are ambiguous values that need LLM resolution.
    
    Args:
        question: User's question
        cards: Schema cards
        
    Returns:
        List of ambiguous mention dicts with context
    """
    question_lower = question.lower()
    ambiguous = []
    
    # Get all actual column names for exact matching
    column_names = set()
    for col_card in cards.get("column_cards", []):
        column_names.add(col_card["column"].lower())
    
    # Pattern 1: "with X" or "has X" where X is not a column
    with_pattern = r"\b(?:with|has|containing|having)\s+([a-zA-Z0-9_]+)"
    for match in re.finditer(with_pattern, question_lower):
        value = match.group(1)
        if value not in column_names and value not in ["a", "an", "the"]:
            ambiguous.append({
                "mention": value,
                "context": match.group(0),
                "type": "value_reference",
                "position": match.start()
            })
    
    # Pattern 1b: "X transactions/customers/payments" where X is not a column
    # Matches patterns like "MT103 transactions", "India customers", "failed payments"
    prefix_pattern = r"\b([A-Z][A-Z0-9]{2,})\s+(transactions?|customers?|payments?|orders?|records?)"
    for match in re.finditer(prefix_pattern, question):  # Case sensitive for codes like MT103
        value = match.group(1).lower()
        if value not in column_names:
            ambiguous.append({
                "mention": value,
                "context": match.group(0),
                "type": "value_reference",
                "position": match.start()
            })
    
    # Pattern 2: Semantic concepts (high/low value, recent, active, etc.)
    semantic_patterns = [
        r"\b(high|low|large|small)[-\s]?(value|amount|cost|price)",
        r"\b(recent|latest|new|old|active|inactive)",
    ]
    
    for pattern in semantic_patterns:
        for match in re.finditer(pattern, question_lower):
            ambiguous.append({
                "mention": match.group(0),
                "context": match.group(0),
                "type": "semantic_concept",
                "position": match.start()
            })
    
    # Pattern 3: Geographic/categorical values (country names, status values, etc.)
    # These are harder to detect deterministically, so we'll let LLM handle them
    # For now, just flag potential categorical values after "in", "from", "to"
    location_pattern = r"\b(?:in|from|to)\s+([A-Z][a-z]+)"
    for match in re.finditer(location_pattern, question):  # Case sensitive!
        value = match.group(1)
        if value not in ["December", "January", "February"]:  # Not months
            ambiguous.append({
                "mention": value,
                "context": match.group(0),
                "type": "categorical_value",
                "position": match.start()
            })
    
    return ambiguous


def format_cards_for_llm(cards: dict, context_tables: set = None, ambiguous_mentions: list = None) -> str:
    """Format card information for LLM context.
    
    Args:
        cards: Full cards dict
        context_tables: Optional set of tables to focus on
        ambiguous_mentions: Optional list of mentions to prioritize related columns
        
    Returns:
        Formatted string with relevant card info
    """
    lines = []
    
    # Table cards
    table_cards = cards.get("table_cards", [])
    if context_tables:
        table_cards = [tc for tc in table_cards if tc["table"] in context_tables]
    
    for tc in table_cards[:5]:  # Limit to avoid token bloat
        lines.append(f"📊 Table: {tc['table']}")
        lines.append(f"   Grain: {tc.get('grain', 'unknown')}")
        if tc.get("primary_key_candidates"):
            lines.append(f"   Primary keys: {', '.join(tc['primary_key_candidates'][:3])}")
        if tc.get("time_cols"):
            lines.append(f"   Time columns: {', '.join(tc['time_cols'][:5])}")
        if tc.get("money_cols"):
            lines.append(f"   Money columns: {', '.join(tc['money_cols'][:5])}")
        if tc.get("gotchas"):
            for gotcha in tc["gotchas"][:2]:
                lines.append(f"   ⚠️  {gotcha}")
        lines.append("")
    
    # Column cards - show relevant ones
    column_cards = cards.get("column_cards", [])
    if context_tables:
        column_cards = [cc for cc in column_cards if cc["table"] in context_tables]
    
    # Prioritize columns that match ambiguous mentions
    if ambiguous_mentions:
        mention_texts = [m.get("mention", "").lower() for m in ambiguous_mentions]
        
        # Split into priority (matching mentions) and rest
        priority_cols = []
        other_cols = []
        for cc in column_cards:
            col_lower = cc["column"].lower()
            if any(mention in col_lower for mention in mention_texts):
                priority_cols.append(cc)
            else:
                other_cols.append(cc)
        
        # Take all priority columns + remaining up to 30 total
        column_cards = priority_cols + other_cols[:max(0, 30 - len(priority_cols))]
    else:
        column_cards = column_cards[:30]  # Limit columns
    
    lines.append("📋 Columns:")
    for cc in column_cards:
        hints = cc.get("semantic_hints", [])
        hints_str = f" [{', '.join(hints)}]" if hints else ""
        null_info = f" ({cc.get('null_pct', 0):.0f}% NULL)" if cc.get("null_pct", 0) > 50 else ""
        
        lines.append(
            f"   • {cc['table']}.{cc['column']} "
            f"({cc.get('duckdb_type', 'unknown')}){hints_str}{null_info}"
        )
    
    return "\n".join(lines)


def resolve_with_llm(
    question: str, 
    ambiguous_mentions: list[dict],
    cards: dict,
    context_tables: set = None
) -> list[dict]:
    """Use LLM to resolve ambiguous mentions to specific columns and operators.
    
    Args:
        question: User's original question
        ambiguous_mentions: List of ambiguous mentions from extract_non_column_mentions
        cards: Full schema cards
        context_tables: Tables already in context (from entities/metrics)
        
    Returns:
        List of resolution dicts with column, operator, value, reasoning
    """
    if not ambiguous_mentions:
        return []
    
    # Format schema context with prioritized columns
    schema_context = format_cards_for_llm(cards, context_tables, ambiguous_mentions)
    
    
    # Build prompt
    prompt = f"""You are a database schema resolver. Your job is to map ambiguous mentions in user questions to specific database columns and filter operations.

USER QUESTION:
"{question}"

AMBIGUOUS MENTIONS TO RESOLVE:
{json.dumps(ambiguous_mentions, indent=2)}

DATABASE SCHEMA (from profiled Haiku cards):
{schema_context}

YOUR TASK:
For each ambiguous mention, determine:
1. What type of mention is it? (data_value, semantic_concept, categorical_value)
2. Which column(s) in the schema relate to this mention?
3. What SQL operator should be used? (IS NOT NULL, =, >, <, LIKE, IN, BETWEEN)
4. What value(s) if any? (for =, >, <, etc.)
5. Your reasoning

IMPORTANT GUIDELINES:
- Use ONLY columns that exist in the schema above
- Consider semantic hints (e.g., "identifier", "timestamp", "money_or_numeric")
- Consider NULL rates (high NULL rate might mean column tracks presence/absence of something)
- For value checks like "mt103", if there's a related ID column with high NULLs, use IS NOT NULL
- For semantic concepts like "high-value", suggest reasonable numeric thresholds
- Be conservative - if unsure, explain why in reasoning

OUTPUT FORMAT (valid JSON only):
{{
  "resolutions": [
    {{
      "mention": "mt103",
      "mention_type": "data_value",
      "column": "mt103_document_id",
      "table": "test_1_1_merged",
      "operator": "IS NOT NULL",
      "value": null,
      "reasoning": "mt103 is a SWIFT document type. The mt103_document_id column is an identifier with 97.8% NULL rate, suggesting it tracks presence/absence of MT103 documents. IS NOT NULL filters for transactions that have MT103."
    }}
  ]
}}

RESPOND WITH ONLY THE JSON, NO OTHER TEXT:"""

    try:
        response = call_llm([
            {"role": "system", "content": "You are a precise database schema resolver. Always output valid JSON."},
            {"role": "user", "content": prompt}
        ], role="planner", max_tokens=2000)
        
        # Parse JSON response
        # Handle markdown code blocks if present
        response_text = response.strip()
        if response_text.startswith("```"):
            # Extract JSON from code block
            lines = response_text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)
        
        result = json.loads(response_text)
        resolutions = result.get("resolutions", [])
        return resolutions
        
    except json.JSONDecodeError as e:
        print(f"⚠️  LLM resolver failed to parse JSON: {e}")
        print(f"Response was: {response[:200]}")
        return []
    except Exception as e:
        print(f"⚠️  LLM resolver error: {e}")
        return []


def apply_resolutions_to_constraints(
    existing_constraints: list[dict],
    resolutions: list[dict]
) -> list[dict]:
    """Merge LLM resolutions into existing constraint list.
    
    Args:
        existing_constraints: Constraints from deterministic planner
        resolutions: Resolutions from LLM
        
    Returns:
        Enhanced constraints list
    """
    # Add new constraints from LLM resolutions
    enhanced = existing_constraints.copy()
    
    for resolution in resolutions:
        operator = resolution.get("operator", "=")
        column = resolution.get("column")
        table = resolution.get("table")
        value = resolution.get("value")
        
        if not column or not table:
            continue
        
        # Build constraint based on operator
        if operator == "IS NOT NULL":
            constraint = {
                "type": "value_filter",
                "expression": f"{table}.{column} IS NOT NULL",
                "column": column,
                "table": table,
                "confidence": 0.9,
                "source": "llm_resolver",
                "reasoning": resolution.get("reasoning", "")
            }
        elif operator in ("=", ">", "<", ">=", "<=", "!="):
            if value is not None:
                # Quote strings, keep numbers as-is
                if isinstance(value, str):
                    value_str = f"'{value}'"
                else:
                    value_str = str(value)
                
                constraint = {
                    "type": "value_filter",
                    "expression": f"{table}.{column} {operator} {value_str}",
                    "column": column,
                    "table": table,
                    "operator": operator,
                    "value": value,
                    "confidence": 0.85,
                    "source": "llm_resolver",
                    "reasoning": resolution.get("reasoning", "")
                }
            else:
                continue  # Skip if no value provided
        elif operator == "LIKE":
            if value:
                constraint = {
                    "type": "value_filter",
                    "expression": f"{table}.{column} LIKE '%{value}%'",
                    "column": column,
                    "table": table,
                    "confidence": 0.8,
                    "source": "llm_resolver",
                    "reasoning": resolution.get("reasoning", "")
                }
            else:
                continue
        else:
            # Unsupported operator, skip
            continue
        
        enhanced.append(constraint)
    
    return enhanced


def enhance_plan_with_llm(
    plan: dict,
    cards: dict,
    enable_llm: bool = True
) -> dict:
    """Main entry point: enhance deterministic plan with LLM resolver.
    
    Args:
        plan: Plan from deterministic planner
        cards: Schema cards
        enable_llm: Whether to use LLM (can disable for testing/cost)
        
    Returns:
        Enhanced plan with LLM-resolved constraints
    """
    if not enable_llm:
        return plan
    
    question = plan.get("original_question", "")
    
    # Extract ambiguous mentions
    ambiguous = extract_non_column_mentions(question, cards)
    
    if not ambiguous:
        # No ambiguities, return as-is
        return plan
    
    # Get context tables from existing entities/metrics
    context_tables = set()
    for entity in plan.get("entities_detected", []):
        for ref in entity.get("mapped_to", []):
            if "." in ref:
                context_tables.add(ref.split(".")[0])
    
    for metric in plan.get("metrics_requested", []):
        for ref in metric.get("mapped_columns", []):
            if "." in ref:
                context_tables.add(ref.split(".")[0])
    
    # Resolve with LLM
    resolutions = resolve_with_llm(question, ambiguous, cards, context_tables)
    
    if not resolutions:
        # LLM failed or returned nothing, return original plan
        return plan
    
    # Apply resolutions to constraints
    enhanced_constraints = apply_resolutions_to_constraints(
        plan.get("constraints", []),
        resolutions
    )
    
    # Update plan
    enhanced_plan = plan.copy()
    enhanced_plan["constraints"] = enhanced_constraints
    enhanced_plan["llm_resolutions"] = resolutions  # Track what LLM added
    
    return enhanced_plan
