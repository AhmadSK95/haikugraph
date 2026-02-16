"""Multi-agent data assistant module.

This module provides the agent implementations for the analyst loop:
- IntakeAgent: Goal clarification and extraction
- SchemaAgent: Schema introspection and catalog building
- QueryAgent: SQL plan generation and execution
- AuditAgent: Result validation and sanity checks
- NarratorAgent: Answer formatting with evidence
"""

from haikugraph.agents.contracts import (
    # Enums
    AgentStatus,
    AuditCheckStatus,
    ConfidenceLevel,
    # Intake
    ExtractedDimension,
    ExtractedMetric,
    ExtractedTimeWindow,
    IntakeResult,
    # Schema
    ColumnProfile,
    JoinEdge,
    SchemaResult,
    TableProfile,
    # Query
    QueryExecution,
    QueryPlanResult,
    QueryStep,
    # Audit
    AuditCheck,
    AuditResult,
    # Narrator
    EvidenceItem,
    NarrationResult,
    SanityCheck,
    # Trace
    AgentTrace,
    RunTrace,
    # API
    AssistantQueryRequest,
    AssistantQueryResponse,
)

__all__ = [
    # Enums
    "AgentStatus",
    "AuditCheckStatus",
    "ConfidenceLevel",
    # Intake
    "ExtractedDimension",
    "ExtractedMetric",
    "ExtractedTimeWindow",
    "IntakeResult",
    # Schema
    "ColumnProfile",
    "JoinEdge",
    "SchemaResult",
    "TableProfile",
    # Query
    "QueryExecution",
    "QueryPlanResult",
    "QueryStep",
    # Audit
    "AuditCheck",
    "AuditResult",
    # Narrator
    "EvidenceItem",
    "NarrationResult",
    "SanityCheck",
    # Trace
    "AgentTrace",
    "RunTrace",
    # API
    "AssistantQueryRequest",
    "AssistantQueryResponse",
]
