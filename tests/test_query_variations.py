"""Test suite for query language variations.

Tests that different phrasings of the same query produce consistent results.
"""

import json
import subprocess
from pathlib import Path

import pytest


# Test data: groups of equivalent queries
QUERY_VARIATIONS = [
    {
        "concept": "mt103_in_december",
        "expected_result_range": (1800, 1900),  # Expected count range
        "variations": [
            "How many transactions with mt103 in December?",
            "Count transactions with MT103 in December",
            "How many MT103 transactions in December?",
            "Show count of transactions having mt103 in December",
            "December transactions with mt103 - how many?",
        ]
    },
    {
        "concept": "payment_amount_september_vs_october",
        "expected_metric": "payment_amount",
        "variations": [
            "Compare total payment amount in September 2025 vs October 2025",
            "Payment amount September 2025 vs October 2025",
            "Total payment amount: September 2025 compared to October 2025",
            "How does payment amount in Sep 2025 compare to Oct 2025?",
        ]
    },
    {
        "concept": "recent_transactions",
        "expected_time_filter": True,
        "variations": [
            "Show recent transactions",
            "List latest transactions",
            "Display new transactions",
        ]
    }
]


def run_query(question: str, use_llm: bool = False) -> dict:
    """Run a query and return the result.
    
    Args:
        question: Question to ask
        use_llm: Whether to use LLM resolver
        
    Returns:
        Result dict with SQL, row_count, etc.
    """
    cmd = [
        "haikugraph",
        "ask",
        "--question", question,
        "--execute"
    ]
    
    if use_llm:
        cmd.append("--use-llm-resolver")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=False
        )
        
        # Load result.json
        result_path = Path(__file__).parent.parent / "data" / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                return json.load(f)
        else:
            return {"error": "No result file", "stdout": result.stdout, "stderr": result.stderr}
            
    except Exception as e:
        return {"error": str(e)}


def extract_count_from_result(result: dict) -> int | None:
    """Extract count value from result."""
    try:
        sq_results = result.get("subquestion_results", [])
        if sq_results:
            preview = sq_results[0].get("preview_rows", [])
            if preview:
                # Look for count_* column
                row = preview[0]
                for key, value in row.items():
                    if key.startswith("count_"):
                        return int(value)
    except Exception:
        return None
    return None


def extract_sql_from_result(result: dict) -> str | None:
    """Extract SQL query from result."""
    try:
        sq_results = result.get("subquestion_results", [])
        if sq_results:
            return sq_results[0].get("sql", "")
    except Exception:
        return None
    return None


def compare_results(result1: dict, result2: dict) -> dict:
    """Compare two query results for consistency.
    
    Returns:
        Dict with comparison metrics
    """
    sql1 = extract_sql_from_result(result1)
    sql2 = extract_sql_from_result(result2)
    
    count1 = extract_count_from_result(result1)
    count2 = extract_count_from_result(result2)
    
    # Normalize SQL for comparison (remove whitespace differences)
    def normalize_sql(sql):
        if not sql:
            return ""
        return " ".join(sql.upper().split())
    
    sql1_norm = normalize_sql(sql1)
    sql2_norm = normalize_sql(sql2)
    
    return {
        "sql_identical": sql1_norm == sql2_norm,
        "sql1": sql1,
        "sql2": sql2,
        "count_identical": count1 == count2 if (count1 is not None and count2 is not None) else None,
        "count1": count1,
        "count2": count2,
        "consistent": (sql1_norm == sql2_norm) or (count1 == count2 if count1 and count2 else False)
    }


@pytest.mark.parametrize("variation_group", QUERY_VARIATIONS)
def test_query_variations_deterministic(variation_group):
    """Test that query variations produce consistent results (deterministic planner)."""
    variations = variation_group["variations"]
    concept = variation_group["concept"]
    
    # Run all variations
    results = []
    for question in variations:
        result = run_query(question, use_llm=False)
        results.append((question, result))
    
    # Compare first result with all others
    base_question, base_result = results[0]
    
    for i, (question, result) in enumerate(results[1:], 1):
        comparison = compare_results(base_result, result)
        
        # Log comparison
        print(f"\n{concept}: Comparing variation {i+1}")
        print(f"  Base: {base_question}")
        print(f"  Var{i+1}: {question}")
        print(f"  SQL identical: {comparison['sql_identical']}")
        print(f"  Count identical: {comparison['count_identical']}")
        
        # Assert consistency
        assert comparison["consistent"], (
            f"Inconsistent results for {concept}:\n"
            f"  Base: {base_question}\n"
            f"  Var: {question}\n"
            f"  Base SQL: {comparison['sql1']}\n"
            f"  Var SQL: {comparison['sql2']}\n"
            f"  Base count: {comparison['count1']}\n"
            f"  Var count: {comparison['count2']}"
        )


@pytest.mark.parametrize("variation_group", QUERY_VARIATIONS)
def test_query_variations_with_llm(variation_group):
    """Test that query variations produce consistent results (with LLM resolver)."""
    variations = variation_group["variations"]
    concept = variation_group["concept"]
    
    # Run all variations with LLM
    results = []
    for question in variations:
        result = run_query(question, use_llm=True)
        results.append((question, result))
    
    # Compare first result with all others
    base_question, base_result = results[0]
    
    for i, (question, result) in enumerate(results[1:], 1):
        comparison = compare_results(base_result, result)
        
        # Log comparison
        print(f"\n{concept} (LLM): Comparing variation {i+1}")
        print(f"  Base: {base_question}")
        print(f"  Var{i+1}: {question}")
        print(f"  SQL identical: {comparison['sql_identical']}")
        print(f"  Count identical: {comparison['count_identical']}")
        
        # With LLM, we expect even better consistency
        assert comparison["consistent"], (
            f"Inconsistent results for {concept} with LLM:\n"
            f"  Base: {base_question}\n"
            f"  Var: {question}\n"
            f"  Base SQL: {comparison['sql1']}\n"
            f"  Var SQL: {comparison['sql2']}\n"
            f"  Base count: {comparison['count1']}\n"
            f"  Var count: {comparison['count2']}"
        )


def test_deterministic_vs_llm_consistency():
    """Test that deterministic and LLM approaches produce similar results for clear queries."""
    clear_query = "How many transactions with mt103 in December?"
    
    result_det = run_query(clear_query, use_llm=False)
    result_llm = run_query(clear_query, use_llm=True)
    
    comparison = compare_results(result_det, result_llm)
    
    print(f"\nDeterministic vs LLM comparison:")
    print(f"  Query: {clear_query}")
    print(f"  Det count: {comparison['count1']}")
    print(f"  LLM count: {comparison['count2']}")
    print(f"  Consistent: {comparison['consistent']}")
    
    # For clear queries, both should produce same results
    assert comparison["count_identical"], (
        f"Deterministic and LLM produced different counts:\n"
        f"  Det: {comparison['count1']}\n"
        f"  LLM: {comparison['count2']}\n"
        f"  Det SQL: {comparison['sql1']}\n"
        f"  LLM SQL: {comparison['sql2']}"
    )


def test_expected_ranges():
    """Test that results fall within expected ranges."""
    for group in QUERY_VARIATIONS:
        if "expected_result_range" in group:
            concept = group["concept"]
            min_expected, max_expected = group["expected_result_range"]
            
            # Test first variation
            question = group["variations"][0]
            result = run_query(question, use_llm=False)
            count = extract_count_from_result(result)
            
            assert count is not None, f"No count returned for {concept}"
            assert min_expected <= count <= max_expected, (
                f"Count {count} outside expected range [{min_expected}, {max_expected}] "
                f"for {concept}"
            )
            
            print(f"\n{concept}: count={count} (expected {min_expected}-{max_expected}) ✓")


if __name__ == "__main__":
    # Run tests manually
    print("="*70)
    print("QUERY VARIATION TESTS")
    print("="*70)
    
    # Test expected ranges first
    print("\n\n1. Testing expected ranges...")
    print("-"*70)
    test_expected_ranges()
    
    # Test each group
    for i, group in enumerate(QUERY_VARIATIONS, 1):
        print(f"\n\n{i}. Testing {group['concept']}")
        print("-"*70)
        test_query_variations_deterministic(group)
    
    print("\n\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70)
