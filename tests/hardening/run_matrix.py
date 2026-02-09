"""
Main test harness for HaikuGraph hardening.

Runs combinatorial question matrix through CLI, checks invariants, and generates report.
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from tests.hardening.question_generator import (
    generate_question_matrix,
    QuestionSpec,
)
from tests.hardening.cli_runner import (
    run_question_through_cli,
    CLITestResult,
)
from tests.hardening.oracle import (
    check_oracle_invariants,
    classify_failure,
    OracleViolation,
)


def run_hardening_matrix(
    db_path: Path,
    data_dir: Path,
    max_questions: int = 200,
    planner: str = "deterministic",
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Run the complete hardening matrix.
    
    Args:
        db_path: Path to DuckDB database
        data_dir: Path to data directory
        max_questions: Maximum number of questions to test
        planner: "deterministic" or "llm"
        output_dir: Directory to save results (default: ./reports/hardening_TIMESTAMP)
    
    Returns:
        Summary dict with results and failure buckets
    """
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./reports/hardening_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("HaikuGraph Hardening Matrix")
    print("=" * 80)
    print(f"Database: {db_path}")
    print(f"Planner: {planner}")
    print(f"Max questions: {max_questions}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    print()
    
    # Generate question matrix
    print("Generating question matrix...")
    questions = generate_question_matrix(
        max_questions=max_questions,
        include_comparisons=True,
        include_breakdowns=True,
        include_filters=True,
    )
    print(f"Generated {len(questions)} questions\\n")
    
    # Run each question
    results: List[Dict[str, Any]] = []
    failure_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    total = len(questions)
    passed = 0
    failed = 0
    
    for i, spec in enumerate(questions, 1):
        question_text = spec.to_natural_language()
        
        print(f"[{i}/{total}] Testing: {question_text[:70]}...")
        
        # Run through CLI
        cli_result = run_question_through_cli(
            question=question_text,
            db_path=db_path,
            data_dir=data_dir,
            planner=planner,
            timeout=30,
        )
        
        # Check invariants
        violations = check_oracle_invariants(spec, cli_result)
        
        # Classify result
        bucket = classify_failure(violations)
        
        if bucket == "success":
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = f"❌ FAIL ({bucket})"
        
        print(f"    {status}")
        
        # Store result
        result_entry = {
            "index": i,
            "question": question_text,
            "spec": {
                "intent": spec.intent.value,
                "metric": spec.metric,
                "time_window": spec.time_window.value if spec.time_window else None,
                "breakdown": spec.breakdown.value if spec.breakdown else None,
                "expected_shape": spec.expected_shape,
                "expected_group_by": spec.expected_group_by,
                "expected_distinct": spec.expected_distinct,
            },
            "result": {
                "exit_code": cli_result.exit_code,
                "error": cli_result.error,
                "sql": cli_result.sql,
                "intent_type": cli_result.intent_type,
                "has_group_by": cli_result.has_group_by,
                "has_distinct": cli_result.has_distinct,
                "row_count": cli_result.row_count,
                "column_count": cli_result.column_count,
            },
            "violations": [
                {
                    "type": v.violation_type.value,
                    "description": v.description,
                    "expected": v.expected,
                    "actual": v.actual,
                    "severity": v.severity,
                }
                for v in violations
            ],
            "bucket": bucket,
        }
        
        results.append(result_entry)
        
        # Add to failure bucket
        if bucket != "success":
            failure_buckets[bucket].append(result_entry)
        
        # Progress update every 20 questions
        if i % 20 == 0:
            print(f"    Progress: {i}/{total} ({passed} passed, {failed} failed)\\n")
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total: {total}")
    print(f"Passed: {passed} ({100*passed//total}%)")
    print(f"Failed: {failed} ({100*failed//total}%)")
    print()
    
    # Print failure buckets
    print("Failure Buckets:")
    print("-" * 80)
    for bucket, entries in sorted(failure_buckets.items(), key=lambda x: -len(x[1])):
        count = len(entries)
        pct = 100 * count // total
        print(f"  {bucket:30s}: {count:3d} ({pct:2d}%)")
    
    # Generate summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "planner": planner,
        "total_questions": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total > 0 else 0,
        "failure_buckets": {
            bucket: len(entries)
            for bucket, entries in failure_buckets.items()
        },
        "db_path": str(db_path),
        "output_dir": str(output_dir),
    }
    
    # Save results
    print()
    print("=" * 80)
    print("Saving results...")
    
    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")
    
    # Save all results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results: {results_path}")
    
    # Save failures only
    failures_path = output_dir / "failures.json"
    failures = [r for r in results if r["bucket"] != "success"]
    with open(failures_path, 'w') as f:
        json.dump(failures, f, indent=2)
    print(f"  Failures: {failures_path}")
    
    # Save per-bucket files
    buckets_dir = output_dir / "buckets"
    buckets_dir.mkdir(exist_ok=True)
    for bucket, entries in failure_buckets.items():
        bucket_path = buckets_dir / f"{bucket}.json"
        with open(bucket_path, 'w') as f:
            json.dump(entries, f, indent=2)
        print(f"  Bucket {bucket}: {bucket_path}")
    
    print()
    print("=" * 80)
    print("✅ Hardening matrix complete!")
    print("=" * 80)
    
    return summary


def main():
    """CLI entry point for hardening matrix"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run HaikuGraph hardening matrix")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("./data/haikugraph.duckdb"),
        help="Path to DuckDB database",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Path to data directory",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=200,
        help="Maximum number of questions to test",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="deterministic",
        choices=["deterministic", "llm"],
        help="Planner to use",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.db_path.exists():
        print(f"Error: Database not found: {args.db_path}", file=sys.stderr)
        print("Run 'haikugraph ingest' first to create the database", file=sys.stderr)
        sys.exit(1)
    
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Run matrix
    summary = run_hardening_matrix(
        db_path=args.db_path,
        data_dir=args.data_dir,
        max_questions=args.max_questions,
        planner=args.planner,
        output_dir=args.output_dir,
    )
    
    # Exit with non-zero if failures
    if summary["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
