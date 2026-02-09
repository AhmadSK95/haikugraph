"""
CLI test runner for HaikuGraph hardening.

Runs questions through the actual CLI, captures plan + SQL + results + errors.
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import re


@dataclass
class CLITestResult:
    """Result of running a question through the CLI"""
    question: str
    exit_code: int
    stdout: str
    stderr: str
    
    # Parsed artifacts
    plan: Optional[Dict[str, Any]] = None
    sql: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Extracted metadata for oracle checks
    intent_type: Optional[str] = None
    has_group_by: Optional[bool] = None
    has_distinct: Optional[bool] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization"""
        return asdict(self)


def run_question_through_cli(
    question: str,
    db_path: Path,
    data_dir: Path,
    planner: str = "deterministic",
    timeout: int = 30,
) -> CLITestResult:
    """
    Run a question through the HaikuGraph CLI.
    
    Args:
        question: Natural language question
        db_path: Path to DuckDB database
        data_dir: Path to data directory (for plan/result files)
        planner: "deterministic" or "llm"
        timeout: Command timeout in seconds
    
    Returns:
        CLITestResult with captured artifacts
    """
    # Create temp files for plan and result
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as plan_file:
        plan_path = Path(plan_file.name)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as result_file:
        result_path = Path(result_file.name)
    
    try:
        # Build CLI command
        if planner == "deterministic":
            cmd = [
                "haikugraph", "ask",
                "--question", question,
                "--db-path", str(db_path),
                "--out", str(plan_path),
                "--execute",
            ]
        elif planner == "llm":
            cmd = [
                "haikugraph", "ask-a6",
                "--question", question,
                "--db-path", str(db_path),
                "--out", str(plan_path),
            ]
        else:
            raise ValueError(f"Unknown planner: {planner}")
        
        # Run command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(data_dir.parent),  # Run from repo root
        )
        
        # Parse artifacts
        plan = None
        sql = None
        result_data = None
        error = None
        
        # Try to load plan file
        if plan_path.exists():
            try:
                with open(plan_path, 'r') as f:
                    plan = json.load(f)
            except Exception as e:
                error = f"Failed to parse plan.json: {e}"
        
        # Try to load result file (for deterministic planner)
        if planner == "deterministic":
            result_file_path = data_dir / "result.json"
            if result_file_path.exists():
                try:
                    with open(result_file_path, 'r') as f:
                        result_data = json.load(f)
                except Exception as e:
                    error = f"Failed to parse result.json: {e}"
        
        # Extract SQL from stdout or result
        sql = extract_sql_from_output(result.stdout, result_data)
        
        # Extract metadata
        intent_type = None
        has_group_by = None
        has_distinct = None
        row_count = None
        column_count = None
        
        if plan:
            intent_type = plan.get("intent", {}).get("type")
        
        if sql:
            has_group_by = "GROUP BY" in sql.upper()
            has_distinct = "COUNT(DISTINCT" in sql.upper() or "COUNT (DISTINCT" in sql.upper()
        
        if result_data and result_data.get("subquestion_results"):
            for sq_result in result_data["subquestion_results"]:
                if sq_result["status"] == "success":
                    row_count = sq_result.get("row_count", 0)
                    column_count = len(sq_result.get("columns", []))
                    break
        
        # Check for errors
        if result.exit_code != 0:
            error = result.stderr or "Command failed with non-zero exit code"
        elif "❌" in result.stdout or "Error" in result.stdout:
            error = extract_error_from_output(result.stdout)
        
        return CLITestResult(
            question=question,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            plan=plan,
            sql=sql,
            result=result_data,
            error=error,
            intent_type=intent_type,
            has_group_by=has_group_by,
            has_distinct=has_distinct,
            row_count=row_count,
            column_count=column_count,
        )
    
    except subprocess.TimeoutExpired:
        return CLITestResult(
            question=question,
            exit_code=-1,
            stdout="",
            stderr="",
            error=f"Command timed out after {timeout} seconds",
        )
    
    except Exception as e:
        return CLITestResult(
            question=question,
            exit_code=-1,
            stdout="",
            stderr="",
            error=f"Exception running command: {e}",
        )
    
    finally:
        # Cleanup temp files
        try:
            plan_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)
        except:
            pass


def extract_sql_from_output(stdout: str, result_data: Optional[Dict] = None) -> Optional[str]:
    """
    Extract SQL query from stdout or result data.
    
    Priority:
    1. result.json subquestion_results[].sql
    2. stdout "Generated SQL:" section
    """
    # Try result data first (most reliable)
    if result_data and result_data.get("subquestion_results"):
        for sq_result in result_data["subquestion_results"]:
            if sq_result.get("sql"):
                return sq_result["sql"]
    
    # Try stdout parsing
    sql_match = re.search(r"Generated SQL:\s*\n(.+?)(?:\n={40,}|\n\n|$)", stdout, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()
    
    return None


def extract_error_from_output(stdout: str) -> Optional[str]:
    """Extract error message from stdout"""
    # Look for error patterns
    error_patterns = [
        r"❌\s+Error:?\s*(.+?)(?:\n|$)",
        r"Error:\s*(.+?)(?:\n|$)",
        r"Failed:\s*(.+?)(?:\n|$)",
    ]
    
    for pattern in error_patterns:
        match = re.search(pattern, stdout, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def parse_plan_from_stdout(stdout: str) -> Optional[Dict]:
    """Try to extract plan JSON from stdout (for --debug mode)"""
    # Look for JSON blocks
    json_match = re.search(r"\{.*\"intent\".*\}", stdout, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    return None


def get_actual_time_filter_sql(sql: Optional[str]) -> Optional[str]:
    """
    Extract time filter SQL from generated query.
    
    Returns the WHERE clause related to time filtering.
    """
    if not sql:
        return None
    
    # Look for common time filter patterns
    time_patterns = [
        r"EXTRACT\(MONTH FROM .+?\) = \d+",
        r"EXTRACT\(YEAR FROM .+?\) = \d+",
        r"date_trunc\('[^']+', .+?\)",
        r".+? >= CURRENT_DATE - INTERVAL '\d+ days?'",
        r".+? >= DATE '.+?'",
    ]
    
    filters = []
    for pattern in time_patterns:
        matches = re.findall(pattern, sql, re.IGNORECASE)
        filters.extend(matches)
    
    return " AND ".join(filters) if filters else None


if __name__ == "__main__":
    # Test the CLI runner
    from pathlib import Path
    
    db_path = Path("./data/haikugraph.duckdb")
    data_dir = Path("./data")
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run 'haikugraph ingest' first")
        exit(1)
    
    # Test question
    test_question = "How many transactions in December?"
    
    print(f"Testing CLI runner with: {test_question}\\n")
    print("=" * 80)
    
    result = run_question_through_cli(
        question=test_question,
        db_path=db_path,
        data_dir=data_dir,
        planner="deterministic",
    )
    
    print(f"Exit code: {result.exit_code}")
    print(f"Intent: {result.intent_type}")
    print(f"Has GROUP BY: {result.has_group_by}")
    print(f"Has DISTINCT: {result.has_distinct}")
    print(f"Row count: {result.row_count}")
    print(f"Error: {result.error}")
    print(f"\\nSQL:\\n{result.sql}")
