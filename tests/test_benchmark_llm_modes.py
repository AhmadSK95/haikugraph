"""Benchmark: Anthropic vs OpenAI vs Local Ollama vs Deterministic comparison.

Runs 18 ground-truth queries against the ``known_data_db`` fixture in each
available LLM mode, then scores **correctness**, **confidence calibration**,
**latency**, and **narrative quality** to produce a composite "Intelligence
Score" and LLM-uplift metric.

Usage::

    # Deterministic only (no external deps):
    pytest tests/test_benchmark_llm_modes.py -k deterministic -v

    # With Ollama running:
    pytest tests/test_benchmark_llm_modes.py -k "deterministic or local" -v

    # Full benchmark (all 4 modes):
    pytest tests/test_benchmark_llm_modes.py -v

    # Skip benchmark in normal test runs:
    pytest tests/ -m "not benchmark"
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from haikugraph.api.server import create_app

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

MODES = ("deterministic", "local", "openai", "anthropic")

# Weights for composite Intelligence Score
W_CORRECTNESS = 0.50
W_CONFIDENCE = 0.15
W_LATENCY = 0.15
W_NARRATIVE = 0.20

LATENCY_CEILING_MS = 30_000  # 30 s → latency score = 0

# ---------------------------------------------------------------------------
# Ground-truth query specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuerySpec:
    """Defines a benchmark query with its expected answer."""

    id: str
    category: str
    goal: str
    # Expected ground-truth value(s).  For exact numbers supply a tuple of
    # acceptable string representations.  For "row-count" checks supply an
    # int (minimum expected rows) prefixed with "rows>=".
    expected: str | tuple[str, ...]

    @property
    def is_row_count_check(self) -> bool:
        return isinstance(self.expected, str) and self.expected.startswith("rows>=")

    @property
    def min_rows(self) -> int:
        assert self.is_row_count_check
        return int(self.expected.split(">=")[1])


QUERIES: list[QuerySpec] = [
    # --- Simple counts (4) ---
    QuerySpec("cnt_tx", "simple_counts",
              "How many transactions are there?", ("8",)),
    QuerySpec("cnt_cust", "simple_counts",
              "How many customers are there?", ("5",)),
    QuerySpec("cnt_quote", "simple_counts",
              "How many quotes are there?", ("4",)),
    QuerySpec("cnt_book", "simple_counts",
              "How many bookings are there?", ("3",)),
    # --- Aggregations (4) ---
    QuerySpec("agg_tx_total", "aggregations",
              "What is the total payment amount across all transactions?",
              ("15950", "15,950", "15950.0", "15950.00")),
    QuerySpec("agg_book_total", "aggregations",
              "What is the total booked amount across all bookings?",
              ("3000", "3,000", "3000.0", "3000.00")),
    QuerySpec("agg_quote_total", "aggregations",
              "What is the total amount to be paid across all quotes?",
              ("6500", "6,500", "6500.0", "6500.00")),
    QuerySpec("agg_tx_avg", "aggregations",
              "What is the average payment amount per transaction?",
              ("1993", "1994", "1993.75", "1,993.75", "1993.8")),
    # --- Boolean filters (3) ---
    QuerySpec("bool_mt103", "boolean_filters",
              "How many transactions have an MT103?",
              ("3",)),
    QuerySpec("bool_refund", "boolean_filters",
              "How many transactions have a refund?",
              ("2",)),
    QuerySpec("bool_univ", "boolean_filters",
              "How many customers are universities?",
              ("2",)),
    # --- Grouping (3) ---
    QuerySpec("grp_platform", "grouping",
              "Show transaction count by platform",
              "rows>=3"),
    QuerySpec("grp_country", "grouping",
              "Show customer count by country",
              "rows>=3"),
    QuerySpec("grp_deal", "grouping",
              "Show booking count by deal type",
              "rows>=2"),
    # --- Time filters (2) ---
    QuerySpec("time_dec_cnt", "time_filters",
              "How many transactions were created in December 2025?",
              ("5",)),
    QuerySpec("time_dec_total", "time_filters",
              "What is the total payment amount for transactions in December 2025?",
              ("10750", "10,750", "10750.0", "10750.00")),
    # --- Complex (2) ---
    QuerySpec("cplx_amt_plat", "complex",
              "Show total payment amount by platform",
              "rows>=3"),
    QuerySpec("cplx_mt103_plat", "complex",
              "Show MT103 transaction count by platform",
              "rows>=1"),
]


# ---------------------------------------------------------------------------
# QueryResult dataclass — captures every metric for one query × mode
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    query_id: str = ""
    category: str = ""
    goal: str = ""
    mode: str = ""

    # Correctness
    success: bool = False
    answer_correct: bool = False
    correctness_detail: str = ""  # "exact", "partial", "wrong", "error"

    # Confidence
    confidence_score: float = 0.0
    confidence_level: str = ""

    # LLM metadata
    llm_intake_used: bool = False
    llm_narrative_used: bool = False
    llm_effective: bool = False
    provider: str | None = None
    intent_model: str | None = None
    narrator_model: str | None = None

    # Performance
    latency_ms: float = 0.0
    execution_time_ms: float = 0.0

    # Narrative quality
    narrative_length: int = 0
    narrative_has_markdown: bool = False
    sample_rows_count: int = 0

    # Trace
    agent_trace_steps: int = 0
    warnings: list[str] = field(default_factory=list)
    sanity_checks_passed: int = 0
    sanity_checks_total: int = 0

    # Raw answer for debugging
    raw_answer: str = ""


# ---------------------------------------------------------------------------
# Module-scoped results store — shared across all test classes
# ---------------------------------------------------------------------------

_benchmark_results: dict[str, list[QueryResult]] = {
    "deterministic": [],
    "local": [],
    "openai": [],
    "anthropic": [],
}

# Provider availability flags (populated by first fixture use)
_provider_available: dict[str, bool] = {
    "deterministic": True,
    "local": False,
    "openai": False,
    "anthropic": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_providers(client: TestClient) -> None:
    """Probe /api/assistant/providers and cache availability."""
    resp = client.get("/api/assistant/providers")
    if resp.status_code == 200:
        data = resp.json()
        checks = data.get("checks", {})
        _provider_available["local"] = (
            checks.get("ollama", {}).get("available", False)
        )
        _provider_available["openai"] = (
            checks.get("openai", {}).get("available", False)
        )
        _provider_available["anthropic"] = (
            checks.get("anthropic", {}).get("available", False)
        )


def _extract_numbers(text: str) -> list[str]:
    """Pull all number-like tokens from a string (ignores commas inside numbers)."""
    # Match integers and decimals, possibly with commas as thousands separators
    return re.findall(r"[\d,]+\.?\d*", text.replace(",", ""))


def _check_correctness(spec: QuerySpec, answer: str, sample_rows: list[dict]) -> tuple[bool, str]:
    """Return (is_correct, detail) for a query result against ground truth."""
    if spec.is_row_count_check:
        n = len(sample_rows)
        if n >= spec.min_rows:
            return True, f"exact (rows={n}>={spec.min_rows})"
        return False, f"wrong (rows={n}<{spec.min_rows})"

    # Exact value match: check if any expected value appears in the answer
    assert isinstance(spec.expected, tuple)
    answer_lower = answer.lower().replace(",", "")
    for ev in spec.expected:
        ev_clean = ev.replace(",", "")
        if ev_clean in answer_lower:
            return True, "exact"

    # Partial: check if any number in the answer is close to expected
    answer_numbers = _extract_numbers(answer)
    for ev in spec.expected:
        ev_clean = ev.replace(",", "")
        for an in answer_numbers:
            try:
                if abs(float(an) - float(ev_clean)) / max(float(ev_clean), 1) < 0.02:
                    return True, "exact"
            except ValueError:
                continue

    # Check sample_rows for the value
    for row in sample_rows:
        for v in row.values():
            v_str = str(v).replace(",", "")
            for ev in spec.expected:
                ev_clean = ev.replace(",", "")
                try:
                    if abs(float(v_str) - float(ev_clean)) / max(float(ev_clean), 1) < 0.02:
                        return True, "exact"
                except (ValueError, ZeroDivisionError):
                    if ev_clean == v_str:
                        return True, "exact"

    return False, "wrong"


def _score_correctness(detail: str) -> float:
    if detail.startswith("exact"):
        return 1.0
    if detail.startswith("partial"):
        return 0.5
    return 0.0


def _score_confidence_calibration(correct: bool, conf_score: float) -> float:
    """Score confidence calibration.

    Correct + high confidence → 1.0
    Correct + low confidence  → 0.5
    Wrong   + low confidence  → 0.7 (appropriately uncertain)
    Wrong   + high confidence → 0.0 (over-confident and wrong)
    """
    high = conf_score >= 0.7
    if correct and high:
        return 1.0
    if correct and not high:
        return 0.5
    if not correct and not high:
        return 0.7
    return 0.0  # wrong + high


def _score_latency(latency_ms: float) -> float:
    return max(0.0, 1.0 - latency_ms / LATENCY_CEILING_MS)


def _score_narrative(result: QueryResult) -> float:
    """Score narrative quality (0-1 scale).

    Has number     +0.40
    Markdown       +0.20
    Good length    +0.10  (50-2000 chars)
    Evidence       +0.15  (sanity checks passed > 0)
    No warnings    +0.15
    """
    score = 0.0
    answer = result.raw_answer

    # Has a relevant number
    if _extract_numbers(answer):
        score += 0.40

    # Markdown formatting (headers, bold, tables, bullets)
    if any(c in answer for c in ("**", "##", "| ", "- ", "* ")):
        score += 0.20

    # Good length
    if 50 <= result.narrative_length <= 2000:
        score += 0.10

    # Evidence (sanity checks passed)
    if result.sanity_checks_passed > 0:
        score += 0.15

    # No warnings
    if not result.warnings:
        score += 0.15

    return min(score, 1.0)


def _composite_score(results: list[QueryResult]) -> dict[str, float]:
    """Compute composite intelligence score for a list of results."""
    if not results:
        return {
            "correctness": 0.0,
            "confidence": 0.0,
            "latency": 0.0,
            "narrative": 0.0,
            "composite": 0.0,
        }
    correctness_scores = [_score_correctness(r.correctness_detail) for r in results]
    confidence_scores = [
        _score_confidence_calibration(r.answer_correct, r.confidence_score)
        for r in results
    ]
    latency_scores = [_score_latency(r.latency_ms) for r in results]
    narrative_scores = [_score_narrative(r) for r in results]

    avg_c = statistics.mean(correctness_scores)
    avg_cf = statistics.mean(confidence_scores)
    avg_l = statistics.mean(latency_scores)
    avg_n = statistics.mean(narrative_scores)

    composite = (
        W_CORRECTNESS * avg_c
        + W_CONFIDENCE * avg_cf
        + W_LATENCY * avg_l
        + W_NARRATIVE * avg_n
    )
    return {
        "correctness": round(avg_c, 4),
        "confidence": round(avg_cf, 4),
        "latency": round(avg_l, 4),
        "narrative": round(avg_n, 4),
        "composite": round(composite, 4),
    }


def _run_benchmark_query(
    client: TestClient,
    spec: QuerySpec,
    mode: str,
) -> QueryResult:
    """Execute a single benchmark query and capture all metrics."""
    result = QueryResult(
        query_id=spec.id,
        category=spec.category,
        goal=spec.goal,
        mode=mode,
    )

    start = time.perf_counter()
    try:
        resp = client.post(
            "/api/assistant/query",
            json={"goal": spec.goal, "llm_mode": mode},
        )
        elapsed = (time.perf_counter() - start) * 1000
        result.latency_ms = round(elapsed, 2)

        if resp.status_code != 200:
            result.correctness_detail = "error"
            result.raw_answer = f"HTTP {resp.status_code}"
            return result

        data = resp.json()
        result.success = data.get("success", False)
        result.raw_answer = data.get("answer_markdown", "")
        result.confidence_score = float(data.get("confidence_score", 0.0))
        result.confidence_level = data.get("confidence", "")
        result.execution_time_ms = float(data.get("execution_time_ms") or 0.0)
        result.narrative_length = len(result.raw_answer)
        result.narrative_has_markdown = any(
            c in result.raw_answer for c in ("**", "##", "| ", "- ", "* ")
        )
        result.sample_rows_count = len(data.get("sample_rows", []))

        # Trace
        result.agent_trace_steps = len(data.get("agent_trace", []))
        result.warnings = data.get("warnings", [])
        checks = data.get("sanity_checks", [])
        result.sanity_checks_total = len(checks)
        result.sanity_checks_passed = sum(1 for c in checks if c.get("passed"))

        # LLM metadata from runtime
        rt = data.get("runtime", {})
        result.llm_intake_used = bool(rt.get("llm_intake_used", False))
        result.llm_narrative_used = bool(rt.get("llm_narrative_used", False))
        result.llm_effective = bool(rt.get("llm_effective", False))
        result.provider = rt.get("provider")
        result.intent_model = rt.get("intent_model")
        result.narrator_model = rt.get("narrator_model")

        # Correctness
        sample_rows = data.get("sample_rows", [])
        correct, detail = _check_correctness(spec, result.raw_answer, sample_rows)
        result.answer_correct = correct
        result.correctness_detail = detail

    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        result.latency_ms = round(elapsed, 2)
        result.correctness_detail = "error"
        result.raw_answer = str(exc)

    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bench_client(known_data_db):
    """Module-scoped TestClient for benchmarks."""
    app = create_app(db_path=known_data_db)
    client = TestClient(app)
    _check_providers(client)
    return client


# ---------------------------------------------------------------------------
# Override known_data_db to module scope for benchmark efficiency
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def known_data_db():
    """Module-scoped version of known_data_db for benchmark reuse."""
    import tempfile
    import duckdb

    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = Path(f.name)
    db_path.unlink()

    conn = duckdb.connect(str(db_path))

    # Schema
    conn.execute("""
        CREATE TABLE test_1_1_merged (
            transaction_id VARCHAR, customer_id VARCHAR, payee_id VARCHAR,
            platform_name VARCHAR, state VARCHAR, txn_flow VARCHAR,
            payment_status VARCHAR, mt103_created_at VARCHAR,
            created_at VARCHAR, updated_at VARCHAR,
            payment_amount DOUBLE, deal_details_amount DOUBLE,
            amount_collected DOUBLE, refund_refund_id VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE test_3_1 (
            quote_id VARCHAR, customer_id VARCHAR,
            source_currency VARCHAR, destination_currency VARCHAR,
            exchange_rate DOUBLE, total_amount_to_be_paid DOUBLE,
            total_additional_charges DOUBLE, forex_markup DOUBLE,
            created_at VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE test_4_1 (
            payee_id VARCHAR, customer_id VARCHAR, is_university BOOLEAN,
            type VARCHAR, status VARCHAR, created_at VARCHAR,
            address_country VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE test_5_1 (
            deal_id VARCHAR, quote_id VARCHAR, booked_amount DOUBLE,
            rate DOUBLE, deal_type VARCHAR, customer_id VARCHAR,
            payee_id VARCHAR, created_at VARCHAR, updated_at VARCHAR
        )
    """)

    # Transactions (8 rows)
    transactions = [
        ("kt_001", "kc_01", "kp_01", "B2C-APP", "NY", "flow_a", "completed",
         "2025-12-01 10:00:00", "2025-12-01 08:00:00", "2025-12-01 10:30:00",
         1000.0, 1000.0, 1000.0, None),
        ("kt_002", "kc_02", "kp_02", "B2C-WEB", "CA", "flow_b", "completed",
         None, "2025-12-03 11:00:00", "2025-12-03 12:00:00",
         2000.0, 2000.0, 2000.0, None),
        ("kt_003", "kc_01", "kp_01", "B2C-APP", "TX", "flow_c", "completed",
         "2025-11-15 09:00:00", "2025-11-15 07:00:00", "2025-11-15 09:30:00",
         1500.0, 1500.0, 1500.0, None),
        ("kt_004", "kc_03", "kp_03", "B2B", "FL", "flow_a", "pending",
         None, "2025-12-05 14:00:00", "2025-12-05 14:30:00",
         3000.0, 3000.0, 3000.0, None),
        ("kt_005", "kc_02", "kp_02", "B2C-WEB", "IL", "flow_b", "completed",
         "2025-11-20 14:30:00", "2025-11-20 13:00:00", "2025-11-20 15:00:00",
         2500.0, 2500.0, 2500.0, None),
        ("kt_006", "kc_04", "kp_04", "B2C-APP", "NY", "flow_a", "completed",
         None, "2025-12-08 09:00:00", "2025-12-08 09:30:00",
         500.0, 500.0, 500.0, "ref_001"),
        ("kt_007", "kc_05", "kp_05", "B2C-WEB", "CA", "flow_c", "processing",
         None, "2025-11-25 16:00:00", "2025-11-25 16:30:00",
         1200.0, 1200.0, 1200.0, None),
        ("kt_008", "kc_03", "kp_03", "B2B", "TX", "flow_b", "completed",
         None, "2025-12-10 10:00:00", "2025-12-10 10:30:00",
         4250.0, 4250.0, 4250.0, "ref_002"),
    ]
    for row in transactions:
        conn.execute(
            "INSERT INTO test_1_1_merged VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            list(row),
        )

    # Customers (5 rows)
    customers = [
        ("kp_01", "kc_01", True,  "education",  "active", "2025-01-10 08:00:00", "US"),
        ("kp_02", "kc_02", False, "individual", "active", "2025-02-15 09:00:00", "US"),
        ("kp_03", "kc_03", True,  "education",  "active", "2025-03-20 10:00:00", "UK"),
        ("kp_04", "kc_04", False, "business",   "active", "2025-04-25 11:00:00", "US"),
        ("kp_05", "kc_05", False, "retail",     "active", "2025-05-30 12:00:00", "IN"),
    ]
    for row in customers:
        conn.execute("INSERT INTO test_4_1 VALUES (?,?,?,?,?,?,?)", list(row))

    # Quotes (4 rows)
    quotes = [
        ("kq_01", "kc_01", "USD", "INR", 83.25,  1000.0, 12.0, 3.0, "2025-12-02 10:00:00"),
        ("kq_02", "kc_02", "EUR", "USD", 1.10,   2000.0, 8.0,  2.5, "2025-12-04 11:00:00"),
        ("kq_03", "kc_03", "GBP", "INR", 105.50, 3000.0, 15.0, 4.0, "2025-11-18 12:00:00"),
        ("kq_04", "kc_04", "USD", "EUR", 0.92,   500.0,  5.0,  1.5, "2025-12-09 09:00:00"),
    ]
    for row in quotes:
        conn.execute("INSERT INTO test_3_1 VALUES (?,?,?,?,?,?,?,?,?)", list(row))

    # Bookings (3 rows)
    bookings = [
        ("kd_01", "kq_01", 500.0,  1.02, "spot",    "kc_01", "kp_01",
         "2025-12-03 08:00:00", "2025-12-03 08:30:00"),
        ("kd_02", "kq_02", 1000.0, 1.05, "forward", "kc_02", "kp_02",
         "2025-12-05 09:00:00", "2025-12-05 09:30:00"),
        ("kd_03", "kq_03", 1500.0, 1.08, "spot",    "kc_03", "kp_03",
         "2025-11-19 10:00:00", "2025-11-19 10:30:00"),
    ]
    for row in bookings:
        conn.execute("INSERT INTO test_5_1 VALUES (?,?,?,?,?,?,?,?,?)", list(row))

    conn.close()
    yield db_path
    db_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestBenchmarkDeterministic:
    """Baseline: deterministic (no LLM) mode — always runs."""

    @pytest.mark.parametrize("spec", QUERIES, ids=[q.id for q in QUERIES])
    def test_query(self, bench_client: TestClient, spec: QuerySpec) -> None:
        result = _run_benchmark_query(bench_client, spec, "deterministic")
        _benchmark_results["deterministic"].append(result)

        # Deterministic mode must succeed
        assert result.success, (
            f"[deterministic] {spec.id} failed: {result.raw_answer[:200]}"
        )

        # Verify runtime mode was not silently upgraded
        assert result.provider is None or result.provider == "", (
            f"[deterministic] {spec.id} unexpected provider: {result.provider}"
        )


@pytest.mark.benchmark
class TestBenchmarkLocal:
    """Local Ollama mode — skips if Ollama is unavailable."""

    @pytest.mark.parametrize("spec", QUERIES, ids=[q.id for q in QUERIES])
    def test_query(self, bench_client: TestClient, spec: QuerySpec) -> None:
        if not _provider_available["local"]:
            pytest.skip("Ollama not available")

        result = _run_benchmark_query(bench_client, spec, "local")

        # Detect silent fallback to deterministic
        rt_mode = result.provider
        if rt_mode is None and _provider_available["local"]:
            result.warnings.append("silent_fallback_to_deterministic")

        _benchmark_results["local"].append(result)


@pytest.mark.benchmark
class TestBenchmarkOpenAI:
    """OpenAI mode — skips if OpenAI API key is missing."""

    @pytest.mark.parametrize("spec", QUERIES, ids=[q.id for q in QUERIES])
    def test_query(self, bench_client: TestClient, spec: QuerySpec) -> None:
        if not _provider_available["openai"]:
            pytest.skip("OpenAI not available")

        result = _run_benchmark_query(bench_client, spec, "openai")

        # Detect silent fallback
        rt_mode = result.provider
        if rt_mode is None and _provider_available["openai"]:
            result.warnings.append("silent_fallback_to_deterministic")

        _benchmark_results["openai"].append(result)


@pytest.mark.benchmark
class TestBenchmarkAnthropic:
    """Anthropic Claude mode — skips if Anthropic API key is missing."""

    @pytest.mark.parametrize("spec", QUERIES, ids=[q.id for q in QUERIES])
    def test_query(self, bench_client: TestClient, spec: QuerySpec) -> None:
        if not _provider_available["anthropic"]:
            pytest.skip("Anthropic not available")

        result = _run_benchmark_query(bench_client, spec, "anthropic")

        # Detect silent fallback
        rt_mode = result.provider
        if rt_mode is None and _provider_available["anthropic"]:
            result.warnings.append("silent_fallback_to_deterministic")

        _benchmark_results["anthropic"].append(result)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _accuracy_by_category(
    results: list[QueryResult],
) -> dict[str, dict[str, float | int]]:
    """Per-category accuracy breakdown."""
    categories: dict[str, list[QueryResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    out: dict[str, dict[str, float | int]] = {}
    for cat, rs in sorted(categories.items()):
        correct = sum(1 for r in rs if r.answer_correct)
        out[cat] = {
            "total": len(rs),
            "correct": correct,
            "accuracy": round(correct / len(rs), 4) if rs else 0.0,
        }
    return out


def _head_to_head(
    results_a: list[QueryResult],
    results_b: list[QueryResult],
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """Compute head-to-head wins/ties/losses between two modes."""
    a_map = {r.query_id: r for r in results_a}
    b_map = {r.query_id: r for r in results_b}
    common = set(a_map) & set(b_map)

    a_wins = 0
    b_wins = 0
    ties = 0

    for qid in sorted(common):
        ra, rb = a_map[qid], b_map[qid]
        sa = _score_correctness(ra.correctness_detail)
        sb = _score_correctness(rb.correctness_detail)
        if sa > sb:
            a_wins += 1
        elif sb > sa:
            b_wins += 1
        else:
            ties += 1

    score_a = _composite_score(results_a)
    score_b = _composite_score(results_b)
    uplift = round(score_b["composite"] - score_a["composite"], 4)

    return {
        "modes": f"{label_b} vs {label_a}",
        f"{label_a}_wins": a_wins,
        f"{label_b}_wins": b_wins,
        "ties": ties,
        "uplift": uplift,
        f"{label_a}_composite": score_a["composite"],
        f"{label_b}_composite": score_b["composite"],
    }


def _generate_markdown_report(
    timestamp: str,
    provider_avail: dict[str, bool],
    per_query: dict[str, list[dict]],
    summaries: dict[str, dict[str, float]],
    category_acc: dict[str, dict],
    h2h: list[dict],
    latency_stats: dict[str, dict],
    llm_activation: dict[str, dict],
    issue_list: list[str],
) -> str:
    """Build the Markdown report."""
    lines: list[str] = []
    lines.append(f"# LLM Mode Benchmark Report — {timestamp}")
    lines.append("")

    # Provider availability
    lines.append("## Provider Availability")
    lines.append("")
    lines.append("| Provider | Available |")
    lines.append("|----------|-----------|")
    for prov, avail in sorted(provider_avail.items()):
        lines.append(f"| {prov} | {'Yes' if avail else 'No'} |")
    lines.append("")

    # Overall intelligence scores
    lines.append("## Overall Intelligence Scores")
    lines.append("")
    lines.append(
        "| Mode | Correctness (50%) | Confidence (15%) | Latency (15%) "
        "| Narrative (20%) | **Composite** |"
    )
    lines.append("|------|-------------------|------------------|---------------|"
                 "-----------------|---------------|")
    for mode in MODES:
        s = summaries.get(mode)
        if s is None:
            continue
        lines.append(
            f"| {mode} | {s['correctness']:.2%} | {s['confidence']:.2%} "
            f"| {s['latency']:.2%} | {s['narrative']:.2%} "
            f"| **{s['composite']:.2%}** |"
        )
    lines.append("")

    # Accuracy by category
    lines.append("## Accuracy by Category")
    lines.append("")
    cats = set()
    for mode_acc in category_acc.values():
        cats.update(mode_acc.keys())
    cats_sorted = sorted(cats)

    header = "| Category |"
    sep = "|----------|"
    for mode in MODES:
        if mode in category_acc:
            header += f" {mode} |"
            sep += "------|"
    lines.append(header)
    lines.append(sep)
    for cat in cats_sorted:
        row = f"| {cat} |"
        for mode in MODES:
            if mode in category_acc:
                info = category_acc[mode].get(cat, {})
                acc = info.get("accuracy", 0.0)
                correct = info.get("correct", 0)
                total = info.get("total", 0)
                row += f" {acc:.0%} ({correct}/{total}) |"
        lines.append(row)
    lines.append("")

    # Latency comparison
    lines.append("## Latency Comparison")
    lines.append("")
    lines.append("| Mode | Median (ms) | Mean (ms) | Min (ms) | Max (ms) | P95 (ms) |")
    lines.append("|------|-------------|-----------|----------|----------|----------|")
    for mode in MODES:
        ls = latency_stats.get(mode)
        if ls is None:
            continue
        lines.append(
            f"| {mode} | {ls['median']:.0f} | {ls['mean']:.0f} "
            f"| {ls['min']:.0f} | {ls['max']:.0f} | {ls['p95']:.0f} |"
        )
    lines.append("")

    # Head-to-head
    if h2h:
        lines.append("## Head-to-Head Comparison")
        lines.append("")
        for item in h2h:
            lines.append(f"**{item['modes']}** — Uplift: {item['uplift']:+.2%}")
            lines.append("")
            for k, v in item.items():
                if k not in ("modes", "uplift"):
                    lines.append(f"- {k}: {v}")
            lines.append("")

    # LLM step activation
    lines.append("## LLM Step Activation")
    lines.append("")
    lines.append("| Mode | Queries | Intake Used | Narrative Used | Effective |")
    lines.append("|------|---------|-------------|----------------|-----------|")
    for mode in MODES:
        info = llm_activation.get(mode)
        if info is None:
            continue
        lines.append(
            f"| {mode} | {info['total']} | {info['intake_used']} "
            f"| {info['narrative_used']} | {info['effective']} |"
        )
    lines.append("")

    # Issues
    if issue_list:
        lines.append("## Warnings & Issues")
        lines.append("")
        for issue in issue_list:
            lines.append(f"- {issue}")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by test_benchmark_llm_modes.py*")
    return "\n".join(lines)


@pytest.mark.benchmark
class TestBenchmarkReport:
    """Collects results from other classes and produces the final report."""

    def test_generate_report(self, bench_client: TestClient) -> None:
        """Generate JSON and Markdown benchmark reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ------------------------------------------------------------------
        # Gather per-mode summaries
        # ------------------------------------------------------------------
        summaries: dict[str, dict[str, float]] = {}
        category_acc: dict[str, dict] = {}
        latency_stats: dict[str, dict] = {}
        llm_activation: dict[str, dict] = {}
        per_query_results: dict[str, list[dict]] = {}

        active_modes: list[str] = []

        for mode in MODES:
            results = _benchmark_results[mode]
            if not results:
                continue
            active_modes.append(mode)

            summaries[mode] = _composite_score(results)
            category_acc[mode] = _accuracy_by_category(results)

            latencies = [r.latency_ms for r in results]
            latencies_sorted = sorted(latencies)
            p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)
            latency_stats[mode] = {
                "median": statistics.median(latencies),
                "mean": statistics.mean(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "p95": latencies_sorted[p95_idx],
            }

            intake_cnt = sum(1 for r in results if r.llm_intake_used)
            narr_cnt = sum(1 for r in results if r.llm_narrative_used)
            eff_cnt = sum(1 for r in results if r.llm_effective)
            llm_activation[mode] = {
                "total": len(results),
                "intake_used": intake_cnt,
                "narrative_used": narr_cnt,
                "effective": eff_cnt,
            }

            per_query_results[mode] = [asdict(r) for r in results]

        # ------------------------------------------------------------------
        # Head-to-head comparisons
        # ------------------------------------------------------------------
        h2h: list[dict] = []
        det_results = _benchmark_results["deterministic"]
        if det_results:
            for mode in ("local", "openai", "anthropic"):
                if _benchmark_results[mode]:
                    h2h.append(
                        _head_to_head(det_results, _benchmark_results[mode],
                                      "deterministic", mode)
                    )
            # pairwise: local vs openai, local vs anthropic, openai vs anthropic
            llm_modes = [m for m in ("local", "openai", "anthropic") if _benchmark_results[m]]
            for i, ma in enumerate(llm_modes):
                for mb in llm_modes[i + 1:]:
                    h2h.append(
                        _head_to_head(_benchmark_results[ma],
                                      _benchmark_results[mb],
                                      ma, mb)
                    )

        # ------------------------------------------------------------------
        # Collect issues/warnings
        # ------------------------------------------------------------------
        issue_list: list[str] = []
        for mode, results in _benchmark_results.items():
            for r in results:
                if r.warnings:
                    for w in r.warnings:
                        issue_list.append(f"[{mode}] {r.query_id}: {w}")
                if r.correctness_detail == "error":
                    issue_list.append(
                        f"[{mode}] {r.query_id}: ERROR — {r.raw_answer[:120]}"
                    )

        # ------------------------------------------------------------------
        # Write JSON report
        # ------------------------------------------------------------------
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        json_path = REPORTS_DIR / f"benchmark_llm_comparison_{timestamp}.json"
        json_report = {
            "metadata": {
                "timestamp": timestamp,
                "query_count": len(QUERIES),
                "modes_tested": active_modes,
            },
            "provider_availability": _provider_available.copy(),
            "per_query_results": per_query_results,
            "summary_by_mode": summaries,
            "summary_by_category": category_acc,
            "latency_stats": latency_stats,
            "llm_activation": llm_activation,
            "head_to_head": h2h,
            "issues": issue_list,
        }
        json_path.write_text(json.dumps(json_report, indent=2, default=str))

        # ------------------------------------------------------------------
        # Write Markdown report
        # ------------------------------------------------------------------
        md_path = REPORTS_DIR / f"benchmark_llm_comparison_{timestamp}.md"
        md_content = _generate_markdown_report(
            timestamp=timestamp,
            provider_avail=_provider_available,
            per_query=per_query_results,
            summaries=summaries,
            category_acc=category_acc,
            h2h=h2h,
            latency_stats=latency_stats,
            llm_activation=llm_activation,
            issue_list=issue_list,
        )
        md_path.write_text(md_content)

        # ------------------------------------------------------------------
        # Print summary to stdout
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("  LLM MODE BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"\n  Provider availability:")
        for prov, avail in sorted(_provider_available.items()):
            print(f"    {prov:15s} {'AVAILABLE' if avail else 'unavailable'}")

        print(f"\n  Intelligence Scores:")
        print(f"    {'Mode':15s} {'Correct':>10s} {'Confid':>10s} "
              f"{'Latency':>10s} {'Narr':>10s} {'COMPOSITE':>10s}")
        print(f"    {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for mode in MODES:
            s = summaries.get(mode)
            if s is None:
                continue
            print(
                f"    {mode:15s} {s['correctness']:>9.1%} {s['confidence']:>9.1%} "
                f"{s['latency']:>9.1%} {s['narrative']:>9.1%} "
                f"{s['composite']:>9.1%}"
            )

        if h2h:
            print(f"\n  LLM Uplift vs Deterministic:")
            for item in h2h:
                if "deterministic" in item["modes"]:
                    print(f"    {item['modes']:30s}  uplift = {item['uplift']:+.2%}")

        print(f"\n  Reports written to:")
        print(f"    {json_path}")
        print(f"    {md_path}")
        print("=" * 70 + "\n")

        # At least the deterministic baseline should have run
        assert len(active_modes) >= 1, "No benchmark results collected"
        assert json_path.exists(), "JSON report was not written"
        assert md_path.exists(), "Markdown report was not written"
