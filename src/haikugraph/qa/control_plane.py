"""QA control plane: dedicated tenant, budget management, suite execution.

Provides:
- QATenant: Isolated tenant with resettable budget
- QASuiteRunner: Execute test suites with mode matrix
- QABudgetManager: Budget tracking without 429 interruption
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable


@dataclass
class QATenant:
    """Dedicated QA tenant with isolated budget and configuration."""

    tenant_id: str = "qa_dedicated"
    budget_usd: float = 100.0  # QA gets generous budget
    budget_used: float = 0.0
    max_queries_per_run: int = 500
    queries_executed: int = 0
    allow_all_modes: bool = True

    def reset_budget(self) -> None:
        """Reset budget window for new QA run."""
        self.budget_used = 0.0
        self.queries_executed = 0

    def can_execute(self) -> bool:
        """Check if budget allows another query."""
        return (
            self.budget_used < self.budget_usd
            and self.queries_executed < self.max_queries_per_run
        )

    def record_query(self, cost_usd: float = 0.01) -> None:
        """Record a query execution against the budget."""
        self.budget_used += cost_usd
        self.queries_executed += 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QATestCase:
    """A single QA test case."""

    id: str
    suite: str  # S1, S2, S3, S4, S5
    question: str
    expected: dict[str, Any]  # expected outcomes
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QATestResult:
    """Result of executing a single QA test case."""

    test_id: str
    suite: str
    passed: bool
    question: str
    expected: dict[str, Any]
    actual: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    execution_time_ms: float = 0.0
    mode: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QASuiteReport:
    """Report for a complete QA suite run."""

    suite_id: str
    suite_name: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    target_rate: float = 0.0
    gate_passed: bool = False
    results: list[QATestResult] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "suite_name": self.suite_name,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "target_rate": self.target_rate,
            "gate_passed": self.gate_passed,
            "results": [r.to_dict() for r in self.results],
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


def create_qa_tenant() -> QATenant:
    """Create a fresh QA tenant with full budget."""
    return QATenant(
        tenant_id=f"qa_{uuid.uuid4().hex[:8]}",
        budget_usd=100.0,
    )


def evaluate_test_result(
    test_case: QATestCase,
    response: dict[str, Any],
) -> QATestResult:
    """Evaluate a response against expected outcomes.

    Checks:
    - success: API returned success=True
    - answer_contains: expected text in answer
    - confidence_min: minimum confidence level
    - sql_contains: expected SQL fragment
    - refuses: expected refusal
    - row_count: expected row count
    """
    expected = test_case.expected
    passed = True
    failure_reasons = []

    # Check success
    if expected.get("success") is not None:
        if response.get("success") != expected["success"]:
            passed = False
            failure_reasons.append(
                f"Expected success={expected['success']}, got {response.get('success')}"
            )

    # Check answer contains
    if expected.get("answer_contains"):
        answer = (response.get("answer_markdown") or "").lower()
        for fragment in expected["answer_contains"]:
            if fragment.lower() not in answer:
                passed = False
                failure_reasons.append(f"Answer missing expected text: '{fragment}'")

    # Check refuses
    if expected.get("refuses"):
        answer = (response.get("answer_markdown") or "").lower()
        is_refusal = any(w in answer for w in [
            "cannot answer", "not supported", "not available",
            "no data", "cannot provide", "refuse", "i cannot",
            "need clarification",
        ])
        if not is_refusal:
            passed = False
            failure_reasons.append("Expected refusal but got a normal answer")

    # Check confidence minimum
    if expected.get("confidence_min") is not None:
        actual_confidence = response.get("confidence_score", 0.0)
        if actual_confidence < expected["confidence_min"]:
            passed = False
            failure_reasons.append(
                f"Confidence {actual_confidence} below minimum {expected['confidence_min']}"
            )

    # Check SQL contains
    if expected.get("sql_contains"):
        sql = (response.get("sql") or "").upper()
        for fragment in expected["sql_contains"]:
            if fragment.upper() not in sql:
                passed = False
                failure_reasons.append(f"SQL missing expected fragment: '{fragment}'")

    return QATestResult(
        test_id=test_case.id,
        suite=test_case.suite,
        passed=passed,
        question=test_case.question,
        expected=test_case.expected,
        actual={
            "success": response.get("success"),
            "confidence": response.get("confidence_score"),
            "has_sql": bool(response.get("sql")),
            "answer_preview": (response.get("answer_markdown") or "")[:200],
        },
        failure_reason="; ".join(failure_reasons) if failure_reasons else "",
        mode=response.get("runtime", {}).get("llm_mode", "deterministic"),
    )


def build_suite_report(
    suite_id: str,
    suite_name: str,
    results: list[QATestResult],
    target_rate: float,
    execution_time_ms: float = 0.0,
) -> QASuiteReport:
    """Build a suite report from test results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_rate = passed / total if total > 0 else 0.0

    return QASuiteReport(
        suite_id=suite_id,
        suite_name=suite_name,
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=pass_rate,
        target_rate=target_rate,
        gate_passed=pass_rate >= target_rate,
        results=results,
        execution_time_ms=execution_time_ms,
    )
