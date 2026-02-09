"""FastAPI backend for HaikuGraph web UI.

Provides HTTP API that wraps the existing intent → planner → executor → narrator pipeline.
Returns stable JSON contract for UI consumption.
"""

import json
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from haikugraph.planning.intent import classify_intent, Intent
from haikugraph.llm.plan_generator import introspect_schema
from haikugraph.planning.llm_planner import generate_or_patch_plan
from haikugraph.planning.plan import build_plan, load_graph, load_cards_data  # Hybrid approach
from haikugraph.execution import execute_plan
from haikugraph.explain.narrator import narrate_results

app = FastAPI(title="HaikuGraph API", version="1.0.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    """Request to ask a question."""
    question: str = Field(..., min_length=1, description="Natural language question")


class SubquestionResult(BaseModel):
    """Result of a single subquestion execution."""
    id: str
    description: str
    sql: str
    status: str
    row_count: int
    columns: list[str]
    preview_rows: list[dict[str, Any]]
    metadata: dict[str, Any]
    
    # Visualization hints
    display_hint: str | None = None
    chart_type: str | None = None
    x_axis: str | None = None
    y_axis: str | None = None
    units: str | None = None


class ComparisonData(BaseModel):
    """Normalized comparison structure (A11)."""
    metric: str
    current_period: str
    current_value: float | None
    comparison_period: str
    comparison_value: float | None
    delta: float | None
    delta_pct: float | None
    direction: str


class AskResponse(BaseModel):
    """Unified response for question answering."""
    final_answer: str
    
    # Intent
    intent: dict[str, Any] | None = None
    
    # Plan
    plan: dict[str, Any]
    
    # Queries
    queries: list[str]
    
    # Results
    results: list[SubquestionResult]
    
    # Comparison (if applicable)
    comparison: ComparisonData | None = None
    
    # Metadata
    metadata: dict[str, Any]
    
    # Warnings/Errors
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# Global DB path (can be configured via env var or startup param)
DB_PATH = Path("./data/haikugraph.duckdb")


def _infer_visualization_hints(sq_result: dict, intent: Intent | None) -> dict[str, Any]:
    """Infer visualization hints from subquestion results and intent.
    
    Returns dict with: display_hint, chart_type, x_axis, y_axis, units
    """
    hints = {
        "display_hint": "table",  # default
        "chart_type": None,
        "x_axis": None,
        "y_axis": None,
        "units": None,
    }
    
    rows = sq_result.get("preview_rows", [])
    columns = sq_result.get("columns", [])
    row_count = sq_result.get("row_count", 0)
    
    if not rows or row_count == 0:
        hints["display_hint"] = "text"
        return hints
    
    # Single row, single value → number display
    if row_count == 1 and len(columns) == 1:
        hints["display_hint"] = "number"
        hints["units"] = _infer_units(columns[0])
        return hints
    
    # Two columns (dimension + value) → bar chart
    if len(columns) == 2 and row_count > 1:
        hints["display_hint"] = "bar_chart"
        hints["chart_type"] = "bar"
        hints["x_axis"] = columns[0]
        hints["y_axis"] = columns[1]
        hints["units"] = _infer_units(columns[1])
        return hints
    
    # Three+ columns or many rows → table
    if len(columns) >= 3 or row_count > 20:
        hints["display_hint"] = "table"
        return hints
    
    # Time series pattern (date/time/month/year + value) → line chart
    if len(columns) >= 2:
        first_col_lower = columns[0].lower()
        # Check for time bucket columns or date columns
        is_time_series = any(word in first_col_lower for word in [
            "date", "time", "month", "year", "day", "week", "quarter"
        ])
        
        # Also check if first column looks like a timestamp
        if not is_time_series and rows and len(rows) > 0:
            first_val = rows[0].get(columns[0])
            if first_val and isinstance(first_val, str):
                # Check if it looks like a date (YYYY-MM or YYYY-MM-DD)
                import re
                if re.match(r"\d{4}-\d{2}", str(first_val)):
                    is_time_series = True
        
        if is_time_series:
            hints["display_hint"] = "line_chart"
            hints["chart_type"] = "line"
            hints["x_axis"] = columns[0]
            hints["y_axis"] = columns[1]
            hints["units"] = _infer_units(columns[1])
            return hints
    
    # Default to table
    hints["display_hint"] = "table"
    return hints


def _infer_units(col_name: str) -> str | None:
    """Infer units from column name."""
    col_lower = col_name.lower()
    
    if any(word in col_lower for word in ["amount", "price", "cost", "revenue", "total", "payment"]):
        return "currency"
    
    if any(word in col_lower for word in ["count", "number", "qty", "quantity"]):
        return "count"
    
    if any(word in col_lower for word in ["percent", "pct", "rate"]):
        return "percentage"
    
    return None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "db_exists": DB_PATH.exists()}


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Answer a natural language question about data.
    
    Pipeline:
    1. Intent classification (A8)
    2. Plan generation (A6-A7)
    3. SQL execution (A4-A5)
    4. Narration (A9)
    5. Visualization hints
    """
    start_time = time.time()
    
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Database not found at {DB_PATH}. Run 'haikugraph ingest' first."
        )
    
    warnings = []
    errors = []
    intent = None
    plan = None
    result = None
    
    try:
        # Stage 1: Intent classification (non-fatal if fails)
        intent_start = time.time()
        try:
            intent = classify_intent(question)
        except Exception as e:
            warnings.append(f"Intent classification failed: {str(e)[:200]}")
            intent = None
        intent_duration = time.time() - intent_start
        
        # Stage 2: Plan generation (hybrid: LLM with deterministic fallback)
        plan_start = time.time()
        try:
            # Try LLM planner first (with better model)
            schema_text = introspect_schema(DB_PATH)
            
            try:
                # Set better model for planner via environment
                import os
                original_model = os.environ.get("HG_PLANNER_MODEL")
                os.environ["HG_PLANNER_MODEL"] = "llama3.1:8b"  # Better model
                
                plan = generate_or_patch_plan(
                    question=question,
                    schema=schema_text,
                    intent=intent,
                )
                
                # DEBUG: Log generated plan
                print(f"\n{'='*80}")
                print(f"LLM PLANNER OUTPUT:")
                print(f"{'='*80}")
                print(f"Question: {question}")
                print(f"\nGenerated Plan:\n{json.dumps(plan, indent=2)}")
                print(f"{'='*80}\n")
                
                # Restore original model
                if original_model:
                    os.environ["HG_PLANNER_MODEL"] = original_model
                else:
                    os.environ.pop("HG_PLANNER_MODEL", None)
                
                # Validate plan has constraints if question mentions time/filters
                if "december" in question.lower() or "january" in question.lower():
                    if not plan.get("constraints"):
                        warnings.append("LLM planner missed constraints, falling back to deterministic")
                        raise ValueError("Missing expected constraints")
                        
            except Exception as llm_error:
                # Fallback to deterministic planner
                warnings.append(f"LLM planner failed: {str(llm_error)[:100]}, using deterministic fallback")
                
                graph_path = Path("./data/graph.json")
                cards_dir = Path("./data/cards")
                
                if not graph_path.exists():
                    raise FileNotFoundError(f"Graph file not found at {graph_path}")
                
                graph = load_graph(graph_path)
                cards = load_cards_data(cards_dir)
                
                plan = build_plan(question, graph, cards)
        except Exception as e:
            errors.append(f"Plan generation failed: {str(e)[:200]}")
            raise HTTPException(status_code=500, detail=f"Planning failed: {str(e)}")
        plan_duration = time.time() - plan_start
        
        # Stage 3: SQL execution (with fallback to deterministic planner)
        exec_start = time.time()
        try:
            result = execute_plan(plan, DB_PATH)
            
            # Check if any subquestions failed due to column/table errors
            has_schema_errors = False
            for sq_result in result.get("subquestion_results", []):
                if sq_result["status"] != "success":
                    error_msg = sq_result.get("error", "")
                    if "does not have a column" in error_msg or "Binder Error" in error_msg:
                        has_schema_errors = True
                        break
            
            # Fallback to deterministic planner if LLM plan had schema errors
            if has_schema_errors and not plan.get("__is_deterministic"):
                warnings.append("LLM plan had schema errors, falling back to deterministic planner")
                
                graph_path = Path("./data/graph.json")
                cards_dir = Path("./data/cards")
                
                if graph_path.exists():
                    graph = load_graph(graph_path)
                    cards = load_cards_data(cards_dir)
                    plan = build_plan(question, graph, cards)
                    plan["__is_deterministic"] = True  # Mark to prevent infinite loop
                    
                    # Re-execute with deterministic plan
                    result = execute_plan(plan, DB_PATH)
                    warnings.append("Successfully recovered using deterministic planner")
        except Exception as e:
            errors.append(f"Execution failed: {str(e)[:200]}")
            raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")
        exec_duration = time.time() - exec_start
        
        # Check for failed subquestions
        for sq_result in result["subquestion_results"]:
            if sq_result["status"] != "success":
                errors.append(f"Subquestion {sq_result['id']} failed: {sq_result.get('error', 'Unknown')[:200]}")
        
        # Stage 4: Narration
        narration_start = time.time()
        try:
            # Convert execution results to narrator format
            narrator_results = {}
            for sq_result in result["subquestion_results"]:
                sq_id = sq_result["id"]
                if sq_result["status"] == "success":
                    narrator_results[sq_id] = {
                        "rows": sq_result.get("preview_rows", []),
                        "columns": sq_result.get("columns", []),
                        "row_count": sq_result["row_count"],
                    }
                else:
                    narrator_results[sq_id] = {
                        "rows": [],
                        "columns": [],
                        "row_count": 0,
                        "error": sq_result.get("error", "Unknown error"),
                    }
            
            # Call narrator with comparison if present
            comparison_dict = result.get("comparison")
            final_answer = narrate_results(
                original_question=question,
                intent=intent,
                plan=plan,
                results=narrator_results,
                comparison=comparison_dict,
            )
        except Exception as e:
            warnings.append(f"Narration failed: {str(e)[:200]}")
            # Fallback to simple summary
            final_answer = result.get("final_summary", "Query executed successfully but explanation failed.")
        narration_duration = time.time() - narration_start
        
        # Stage 5: Add visualization hints
        enriched_results = []
        for sq_result in result["subquestion_results"]:
            viz_hints = _infer_visualization_hints(sq_result, intent)
            enriched_results.append(
                SubquestionResult(
                    id=sq_result["id"],
                    description=sq_result.get("description", ""),
                    sql=sq_result.get("sql", ""),
                    status=sq_result["status"],
                    row_count=sq_result["row_count"],
                    columns=sq_result.get("columns", []),
                    preview_rows=sq_result.get("preview_rows", []),
                    metadata=sq_result.get("metadata", {}),
                    display_hint=viz_hints["display_hint"],
                    chart_type=viz_hints["chart_type"],
                    x_axis=viz_hints["x_axis"],
                    y_axis=viz_hints["y_axis"],
                    units=viz_hints["units"],
                )
            )
        
        # Extract queries
        queries = [sq.sql for sq in enriched_results if sq.sql]
        
        # Build metadata
        total_duration = time.time() - start_time
        metadata = {
            "execution_time_ms": int(total_duration * 1000),
            "intent_time_ms": int(intent_duration * 1000) if intent else None,
            "plan_time_ms": int(plan_duration * 1000),
            "exec_time_ms": int(exec_duration * 1000),
            "narration_time_ms": int(narration_duration * 1000),
            "total_rows": sum(sq.row_count for sq in enriched_results),
            "tables_used": list(set(table for sq in plan.get("subquestions", []) for table in sq.get("tables", []))),
        }
        
        # Extract comparison if present
        comparison_data = None
        if result.get("comparison"):
            comp = result["comparison"]
            comparison_data = ComparisonData(
                metric=comp.get("metric", ""),
                current_period=comp.get("current", {}).get("period", ""),
                current_value=comp.get("current", {}).get("value"),
                comparison_period=comp.get("comparison", {}).get("period", ""),
                comparison_value=comp.get("comparison", {}).get("value"),
                delta=comp.get("delta"),
                delta_pct=comp.get("delta_pct"),
                direction=comp.get("direction", "flat"),
            )
        
        # Build intent dict
        intent_dict = None
        if intent:
            intent_dict = {
                "type": intent.type.value,
                "confidence": intent.confidence,
                "rationale": intent.rationale,
                "requires_comparison": intent.requires_comparison,
            }
        
        return AskResponse(
            final_answer=final_answer,
            intent=intent_dict,
            plan=plan,
            queries=queries,
            results=enriched_results,
            comparison=comparison_data,
            metadata=metadata,
            warnings=warnings,
            errors=errors,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        error_detail = str(e)[:500]
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {error_detail}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
