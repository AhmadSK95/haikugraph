"""Decision-grade explainability for dataDa.

Builds the 'Explain yourself' panel - a single flow view showing every
decision the system made, what alternatives it considered, and why it
chose the path it did.

Required sections (per Gate 7 - >=95% completeness):
1. query_intent: What the system understood
2. chosen_contract: The semantic contract binding
3. rejected_alternatives: What else was considered and why not
4. sql_annotation: The SQL with comments explaining each clause
5. audit_summary: What checks were run and their results
6. confidence_decomposition: Factor-by-factor confidence breakdown
7. narrative_summary: Final human-readable summary
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any


REQUIRED_SECTIONS = [
    "query_intent",
    "chosen_contract",
    "rejected_alternatives",
    "sql_annotation",
    "audit_summary",
    "confidence_decomposition",
    "narrative_summary",
]


@dataclass
class ExplainSection:
    """A single section in the Explain Yourself panel."""

    section_id: str
    title: str
    content: dict[str, Any]
    present: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExplainYourselfPanel:
    """Complete Explain Yourself decision flow."""

    sections: list[ExplainSection] = field(default_factory=list)
    completeness: float = 0.0  # fraction of required sections present
    missing_sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sections": [s.to_dict() for s in self.sections],
            "completeness": round(self.completeness, 4),
            "missing_sections": self.missing_sections,
        }

    def to_decision_flow(self) -> list[dict[str, Any]]:
        """Convert to the decision_flow format expected by AssistantQueryResponse."""
        return [
            {
                "step": s.section_id,
                "title": s.title,
                **s.content,
            }
            for s in self.sections
            if s.present
        ]


def build_explain_yourself(
    *,
    goal: str = "",
    intent: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
    contract_validation: dict[str, Any] | None = None,
    rejected_alternatives: list[dict[str, Any]] | None = None,
    sql: str = "",
    audit: dict[str, Any] | None = None,
    confidence: dict[str, Any] | None = None,
    narrative: str = "",
    answer_summary: str = "",
) -> ExplainYourselfPanel:
    """Build the complete Explain Yourself panel from pipeline artifacts.

    Each parameter maps to a pipeline stage output.  The function assembles
    all available information into the standardised 7-section format.

    Args:
        goal: Original user question.
        intent: Intent classification result.
        contract: AnalysisContract as dict.
        contract_validation: ContractValidationResult as dict.
        rejected_alternatives: List of alternative plans/tables considered.
        sql: Final SQL executed.
        audit: AuditResult as dict.
        confidence: CalibratedConfidence as dict.
        narrative: Final narrative text.
        answer_summary: One-line answer summary.

    Returns:
        ExplainYourselfPanel with all available sections.
    """
    sections: list[ExplainSection] = []

    # Section 1: Query Intent
    intent_content = _build_intent_section(goal, intent)
    sections.append(ExplainSection(
        section_id="query_intent",
        title="What I understood",
        content=intent_content,
        present=True,  # Always present (at minimum we have the goal)
    ))

    # Section 2: Chosen Contract
    contract_content = _build_contract_section(contract, contract_validation)
    sections.append(ExplainSection(
        section_id="chosen_contract",
        title="Analysis contract",
        content=contract_content,
        present=bool(contract),
    ))

    # Section 3: Rejected Alternatives
    alternatives_content = _build_alternatives_section(rejected_alternatives)
    sections.append(ExplainSection(
        section_id="rejected_alternatives",
        title="What I considered and rejected",
        content=alternatives_content,
        present=True,  # Always present, may be empty
    ))

    # Section 4: SQL Annotation
    sql_content = _build_sql_section(sql, contract)
    sections.append(ExplainSection(
        section_id="sql_annotation",
        title="SQL query (annotated)",
        content=sql_content,
        present=bool(sql),
    ))

    # Section 5: Audit Summary
    audit_content = _build_audit_section(audit)
    sections.append(ExplainSection(
        section_id="audit_summary",
        title="Quality checks",
        content=audit_content,
        present=bool(audit),
    ))

    # Section 6: Confidence Decomposition
    confidence_content = _build_confidence_section(confidence)
    sections.append(ExplainSection(
        section_id="confidence_decomposition",
        title="Confidence breakdown",
        content=confidence_content,
        present=bool(confidence),
    ))

    # Section 7: Narrative Summary
    narrative_content = _build_narrative_section(narrative, answer_summary)
    sections.append(ExplainSection(
        section_id="narrative_summary",
        title="Final answer",
        content=narrative_content,
        present=bool(narrative or answer_summary),
    ))

    # Compute completeness
    present_ids = {s.section_id for s in sections if s.present}
    missing = [sid for sid in REQUIRED_SECTIONS if sid not in present_ids]
    completeness = len(present_ids & set(REQUIRED_SECTIONS)) / len(REQUIRED_SECTIONS)

    return ExplainYourselfPanel(
        sections=sections,
        completeness=completeness,
        missing_sections=missing,
    )


# -------------------------------------------------------------------------
# Section builders
# -------------------------------------------------------------------------

def _build_intent_section(goal: str, intent: dict[str, Any] | None) -> dict[str, Any]:
    """Build the query intent section."""
    content: dict[str, Any] = {
        "original_question": goal,
    }

    if intent:
        content["detected_intent"] = intent.get("type", "unknown")
        content["intent_confidence"] = intent.get("confidence", 0.0)
        content["rationale"] = intent.get("rationale", "")
        content["requires_comparison"] = intent.get("requires_comparison", False)
    else:
        content["detected_intent"] = "not_classified"
        content["intent_confidence"] = 0.0
        content["rationale"] = "Intent classification not performed"

    return content


def _build_contract_section(
    contract: dict[str, Any] | None,
    validation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the analysis contract section."""
    if not contract:
        return {"status": "no_contract", "reason": "Contract not generated for this query"}

    content: dict[str, Any] = {
        "metric": contract.get("metric", "unknown"),
        "domain": contract.get("domain", "unknown"),
        "grain": contract.get("grain", []),
        "time_scope": contract.get("time_scope", {}),
        "filters": contract.get("filters", []),
        "exclusions": contract.get("exclusions", []),
    }

    if validation:
        content["validation"] = {
            "valid": validation.get("valid", False),
            "violations": validation.get("violations", []),
            "checks_count": len(validation.get("checks", [])),
        }

    return content


def _build_alternatives_section(
    alternatives: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build the rejected alternatives section."""
    if not alternatives:
        return {
            "alternatives_considered": 0,
            "items": [],
            "note": "No alternative approaches were evaluated for this query.",
        }

    items = []
    for alt in alternatives:
        items.append({
            "option": alt.get("description", "Unknown alternative"),
            "reason_rejected": alt.get("rejection_reason", "Lower confidence or fitness"),
            "score": alt.get("score", 0.0),
        })

    return {
        "alternatives_considered": len(alternatives),
        "items": items,
    }


def _build_sql_section(sql: str, contract: dict[str, Any] | None) -> dict[str, Any]:
    """Build the SQL annotation section."""
    if not sql:
        return {"sql": "", "annotations": [], "note": "No SQL generated"}

    annotations = _annotate_sql(sql, contract)

    return {
        "sql": sql,
        "annotations": annotations,
    }


def _annotate_sql(sql: str, contract: dict[str, Any] | None) -> list[dict[str, str]]:
    """Add human-readable annotations to SQL clauses."""
    annotations: list[dict[str, str]] = []
    sql_upper = sql.upper()

    # SELECT clause
    if re.search(r'\bSELECT\b(.+?)(?:\bFROM\b)', sql_upper, re.DOTALL):
        annotations.append({
            "clause": "SELECT",
            "explanation": "Columns and aggregations being computed",
        })

    # FROM clause
    from_match = re.search(r'\bFROM\b\s+(\S+)', sql, re.IGNORECASE)
    if from_match:
        table = from_match.group(1).strip('"').strip("'")
        annotations.append({
            "clause": "FROM",
            "explanation": f"Data source: {table}",
        })

    # WHERE clause
    if "WHERE" in sql_upper:
        annotations.append({
            "clause": "WHERE",
            "explanation": "Filters applied to narrow the data",
        })

    # GROUP BY clause
    if "GROUP BY" in sql_upper:
        annotations.append({
            "clause": "GROUP BY",
            "explanation": "Dimensions for aggregation breakdown",
        })

    # ORDER BY clause
    if "ORDER BY" in sql_upper:
        annotations.append({
            "clause": "ORDER BY",
            "explanation": "Result ordering",
        })

    # LIMIT clause
    if "LIMIT" in sql_upper:
        annotations.append({
            "clause": "LIMIT",
            "explanation": "Row count cap for safety",
        })

    # Contract alignment annotation
    if contract:
        metric = contract.get("metric", "")
        domain = contract.get("domain", "")
        if metric and metric != "unknown":
            annotations.append({
                "clause": "CONTRACT",
                "explanation": f"Bound to contract: {metric} on {domain}",
            })

    return annotations


def _build_audit_section(audit: dict[str, Any] | None) -> dict[str, Any]:
    """Build the audit checks summary section."""
    if not audit:
        return {"status": "no_audit", "checks": []}

    checks = audit.get("checks", [])
    summary_checks = []
    for check in checks:
        summary_checks.append({
            "name": check.get("check_name", "unknown"),
            "status": check.get("status", "skipped"),
            "message": check.get("message", ""),
        })

    return {
        "overall_pass": audit.get("overall_pass", False),
        "passed": audit.get("passed", 0),
        "warned": audit.get("warned", 0),
        "failed": audit.get("failed", 0),
        "checks": summary_checks,
    }


def _build_confidence_section(confidence: dict[str, Any] | None) -> dict[str, Any]:
    """Build the confidence decomposition section."""
    if not confidence:
        return {"status": "no_confidence_data"}

    factors = confidence.get("factors", [])
    factor_summary = []
    for f in factors:
        factor_summary.append({
            "name": f.get("name", "unknown"),
            "score": f.get("score", 0.0),
            "weight": f.get("weight", 0.0),
            "reason": f.get("reason", ""),
        })

    return {
        "overall": confidence.get("overall", 0.0),
        "level": confidence.get("level", "UNCERTAIN"),
        "factors": factor_summary,
        "penalties": confidence.get("penalties", []),
        "reasoning": confidence.get("reasoning", ""),
    }


def _build_narrative_section(narrative: str, summary: str) -> dict[str, Any]:
    """Build the narrative summary section."""
    return {
        "answer_summary": summary or (narrative[:200] + "..." if len(narrative) > 200 else narrative),
        "full_narrative": narrative,
    }


def compute_panel_completeness(panel: ExplainYourselfPanel) -> float:
    """Compute the completeness score for Gate 7 evaluation."""
    return panel.completeness
