"""v2 runtime package for quality-first, stage-oriented orchestration."""

from haikugraph.v2.compat_adapter import apply_v2_compat_fields
from haikugraph.v2.event_bus import STAGE_ORDER, StageEventBusV2, StageTransitionError
from haikugraph.v2.orchestrator import V2Orchestrator, V2RunResult
from haikugraph.v2.semantic_cache import SemanticProfileCache
from haikugraph.v2.semantic_profiler import profile_dataset
from haikugraph.v2.types import (
    AssistantResponseV2,
    ConversationStateV2,
    ExecutionResultV2,
    IntentSpecV2,
    PlanCandidateV2,
    PlanSetV2,
    QualityReportV2,
    QueryPlanV2,
    SemanticCatalogV2,
    StageEventV2,
)

__all__ = [
    "AssistantResponseV2",
    "ConversationStateV2",
    "ExecutionResultV2",
    "IntentSpecV2",
    "PlanCandidateV2",
    "PlanSetV2",
    "QualityReportV2",
    "QueryPlanV2",
    "SemanticCatalogV2",
    "StageEventV2",
    "STAGE_ORDER",
    "StageEventBusV2",
    "StageTransitionError",
    "V2Orchestrator",
    "V2RunResult",
    "apply_v2_compat_fields",
    "SemanticProfileCache",
    "profile_dataset",
]
