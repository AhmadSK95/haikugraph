"""Orchestrator runtime for multi-agent data assistant.

This module provides the main orchestration logic that runs agents in order:
Intake → Schema → Query → Audit → (refinement if needed) → Narrator

Key features:
- Deterministic flow with explicit steps
- Max 2 refinement loops (termination guarantee)
- State tracking via RunTrace
- Graceful degradation when LLM unavailable
- Self-learning from successful queries and errors
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from haikugraph.agents.contracts import (
    AgentStatus,
    AgentTrace,
    AssistantQueryResponse,
    AuditResult,
    ConfidenceLevel,
    EvidenceItem,
    IntakeResult,
    NarrationResult,
    QueryPlanResult,
    RunTrace,
    SanityCheck,
    SchemaResult,
)
from haikugraph.agents.intake import IntakeAgent
from haikugraph.agents.schema_agent import SchemaAgent
from haikugraph.agents.query_agent import QueryAgent
from haikugraph.agents.audit_agent import AuditAgent
from haikugraph.agents.narrator_agent import NarratorAgent

# Try to import memory module (optional)
try:
    from haikugraph.memory import MemoryStore, QueryMemory
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    
    # Refinement settings
    max_refinement_loops: int = 2
    
    # Agent settings
    use_llm: bool = True  # Set to False to use simple/rule-based agents
    llm_provider: str | None = None  # Optional provider override (ollama/openai/anthropic)
    llm_model_overrides: dict[str, str] = field(default_factory=dict)
    
    # Timeouts (in seconds)
    total_timeout: int = 120
    agent_timeout: int = 30
    
    # Memory/learning settings
    enable_memory: bool = True  # Enable self-learning from queries
    memory_dir: str | None = None  # Directory for memory storage (default: data/.memory)


@dataclass
class OrchestratorState:
    """Internal state during orchestration."""
    
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=datetime.utcnow)
    
    # Agent results
    intake_result: IntakeResult | None = None
    schema_result: SchemaResult | None = None
    query_result: QueryPlanResult | None = None
    audit_result: AuditResult | None = None
    narration_result: NarrationResult | None = None
    
    # Tracking
    agent_traces: list[AgentTrace] = field(default_factory=list)
    refinement_count: int = 0
    current_step: str = "init"
    error: str | None = None


class AnalystOrchestrator:
    """Main orchestrator for the multi-agent analyst loop.
    
    This orchestrator runs agents in a deterministic order:
    1. IntakeAgent - Parse and clarify the user's goal
    2. SchemaAgent - Introspect database schema
    3. QueryAgent - Generate and execute SQL
    4. AuditAgent - Validate results
    5. (Refinement loop if audit fails, max 2 times)
    6. NarratorAgent - Format final answer
    
    Usage:
        orchestrator = AnalystOrchestrator(db_path)
        response = orchestrator.run("What is total revenue this month?")
    """
    
    def __init__(
        self,
        db_path: Path | str,
        config: OrchestratorConfig | None = None,
    ):
        """Initialize orchestrator.
        
        Args:
            db_path: Path to DuckDB database
            config: Optional configuration
        """
        self.db_path = Path(db_path)
        self.config = config or OrchestratorConfig()
        
        # Initialize agents
        self.intake_agent = IntakeAgent()
        self.schema_agent = SchemaAgent(db_path)
        self.query_agent = QueryAgent(db_path)
        self.audit_agent = AuditAgent()
        self.narrator_agent = NarratorAgent()
        
        if self.config.llm_provider:
            self.intake_agent.llm_provider = self.config.llm_provider
            self.query_agent.llm_provider = self.config.llm_provider
            self.narrator_agent.llm_provider = self.config.llm_provider
        if self.config.llm_model_overrides:
            self.intake_agent.llm_model_overrides = self.config.llm_model_overrides.copy()
            self.query_agent.llm_model_overrides = self.config.llm_model_overrides.copy()
            self.narrator_agent.llm_model_overrides = self.config.llm_model_overrides.copy()
        
        # Initialize memory (optional)
        self.memory_store = None
        self.query_memory = None
        if self.config.enable_memory and HAS_MEMORY:
            memory_dir = self.config.memory_dir or str(self.db_path.parent / ".memory")
            self.memory_store = MemoryStore(memory_dir)
            self.query_memory = QueryMemory(self.memory_store)
    
    def run(
        self,
        goal: str,
        constraints: dict[str, Any] | None = None,
    ) -> AssistantQueryResponse:
        """Run the full analyst loop.
        
        Args:
            goal: User's natural language goal/question
            constraints: Optional constraints (time bounds, filters, etc.)
        
        Returns:
            AssistantQueryResponse with answer, confidence, and evidence
        """
        state = OrchestratorState()
        constraints = constraints or {}
        
        try:
            # Step 1: Intake
            state.current_step = "intake"
            state.intake_result = self._run_intake(goal, state)
            
            if state.intake_result.status == AgentStatus.NEEDS_CLARIFICATION:
                return self._build_clarification_response(state)
            
            # Step 2: Schema
            state.current_step = "schema"
            state.schema_result = self._run_schema(state)
            
            # Steps 3-4: Query and Audit (with refinement loop)
            for iteration in range(self.config.max_refinement_loops + 1):
                state.refinement_count = iteration
                
                # Step 3: Query
                state.current_step = f"query_{iteration}"
                state.query_result = self._run_query(state)
                
                # Step 4: Audit
                state.current_step = f"audit_{iteration}"
                state.audit_result = self._run_audit(state)
                
                # Check if audit passed or max iterations reached
                if state.audit_result.overall_pass or iteration >= self.config.max_refinement_loops:
                    break
                
                # Audit failed - will refine in next iteration
                # Could inject refinement suggestions into next query attempt
            
            # Step 5: Narrator
            state.current_step = "narrator"
            state.narration_result = self._run_narrator(state)
            
            return self._build_success_response(state)
        
        except Exception as e:
            state.error = str(e)
            return self._build_error_response(state, str(e))
    
    def _run_intake(self, goal: str, state: OrchestratorState) -> IntakeResult:
        """Run intake agent."""
        start = datetime.utcnow()
        
        try:
            if self.config.use_llm:
                result = self.intake_agent.run(goal)
            else:
                result = self.intake_agent.run_simple(goal)
            
            state.agent_traces.append(AgentTrace(
                agent_name="intake",
                started_at=start,
                completed_at=datetime.utcnow(),
                duration_ms=result.processing_time_ms,
                status=result.status,
                input_summary={"goal": goal[:100]},
                output_summary={"intent": result.intent_type, "metrics": len(result.metrics)},
            ))
            
            return result
        except Exception as e:
            state.agent_traces.append(AgentTrace(
                agent_name="intake",
                started_at=start,
                completed_at=datetime.utcnow(),
                duration_ms=0,
                status=AgentStatus.FAILED,
                error=str(e),
            ))
            raise
    
    def _run_schema(self, state: OrchestratorState) -> SchemaResult:
        """Run schema agent."""
        start = datetime.utcnow()
        
        try:
            result = self.schema_agent.run(state.intake_result)
            
            state.agent_traces.append(AgentTrace(
                agent_name="schema",
                started_at=start,
                completed_at=datetime.utcnow(),
                duration_ms=result.processing_time_ms,
                status=result.status,
                input_summary={"intake_intent": state.intake_result.intent_type},
                output_summary={
                    "tables": len(result.tables),
                    "relevant_tables": len(result.relevant_tables),
                },
            ))
            
            return result
        except Exception as e:
            state.agent_traces.append(AgentTrace(
                agent_name="schema",
                started_at=start,
                completed_at=datetime.utcnow(),
                duration_ms=0,
                status=AgentStatus.FAILED,
                error=str(e),
            ))
            raise
    
    def _run_query(self, state: OrchestratorState) -> QueryPlanResult:
        """Run query agent."""
        start = datetime.utcnow()
        
        try:
            if self.config.use_llm:
                result = self.query_agent.run(
                    state.intake_result,
                    state.schema_result,
                    execute=True,
                )
            else:
                result = self.query_agent.run_simple(
                    state.intake_result,
                    state.schema_result,
                    execute=True,
                )
            
            state.agent_traces.append(AgentTrace(
                agent_name="query",
                started_at=start,
                completed_at=datetime.utcnow(),
                duration_ms=result.processing_time_ms,
                status=result.status,
                input_summary={
                    "goal": state.intake_result.clarified_goal[:50],
                    "tables": state.schema_result.relevant_tables[:3],
                },
                output_summary={
                    "row_count": result.final_result.row_count if result.final_result else 0,
                    "success": result.final_result.success if result.final_result else False,
                },
            ))
            
            return result
        except Exception as e:
            state.agent_traces.append(AgentTrace(
                agent_name="query",
                started_at=start,
                completed_at=datetime.utcnow(),
                duration_ms=0,
                status=AgentStatus.FAILED,
                error=str(e),
            ))
            raise
    
    def _run_audit(self, state: OrchestratorState) -> AuditResult:
        """Run audit agent."""
        start = datetime.utcnow()
        
        try:
            result = self.audit_agent.run(
                state.query_result,
                state.schema_result,
            )
            
            state.agent_traces.append(AgentTrace(
                agent_name="audit",
                started_at=start,
                completed_at=datetime.utcnow(),
                duration_ms=result.processing_time_ms,
                status=result.status,
                input_summary={
                    "row_count": state.query_result.final_result.row_count if state.query_result.final_result else 0,
                },
                output_summary={
                    "passed": result.passed,
                    "warned": result.warned,
                    "failed": result.failed,
                    "overall_pass": result.overall_pass,
                },
            ))
            
            return result
        except Exception as e:
            state.agent_traces.append(AgentTrace(
                agent_name="audit",
                started_at=start,
                completed_at=datetime.utcnow(),
                duration_ms=0,
                status=AgentStatus.FAILED,
                error=str(e),
            ))
            raise
    
    def _run_narrator(self, state: OrchestratorState) -> NarrationResult:
        """Run narrator agent."""
        start = datetime.utcnow()
        
        try:
            if self.config.use_llm:
                result = self.narrator_agent.run(
                    state.intake_result,
                    state.query_result,
                    state.audit_result,
                )
            else:
                result = self.narrator_agent.run_simple(
                    state.intake_result,
                    state.query_result,
                )
            
            state.agent_traces.append(AgentTrace(
                agent_name="narrator",
                started_at=start,
                completed_at=datetime.utcnow(),
                duration_ms=result.processing_time_ms,
                status=result.status,
                input_summary={
                    "row_count": state.query_result.final_result.row_count if state.query_result.final_result else 0,
                    "audit_pass": state.audit_result.overall_pass,
                },
                output_summary={
                    "confidence": result.confidence_level.value,
                    "evidence_count": len(result.evidence),
                },
            ))
            
            return result
        except Exception as e:
            state.agent_traces.append(AgentTrace(
                agent_name="narrator",
                started_at=start,
                completed_at=datetime.utcnow(),
                duration_ms=0,
                status=AgentStatus.FAILED,
                error=str(e),
            ))
            raise
    
    def _build_success_response(self, state: OrchestratorState) -> AssistantQueryResponse:
        """Build successful response from state."""
        narration = state.narration_result
        
        # Record successful query to memory for learning
        if self.query_memory and state.query_result and state.query_result.final_sql:
            try:
                self.query_memory.record_successful_query(
                    query=state.intake_result.original_goal if state.intake_result else "",
                    sql=state.query_result.final_sql,
                    result_summary={
                        "row_count": state.query_result.final_result.row_count if state.query_result.final_result else 0,
                        "confidence": narration.confidence_score,
                    },
                )
            except Exception:
                pass  # Don't fail the response if memory recording fails
        
        return AssistantQueryResponse(
            success=True,
            answer_markdown=narration.answer_markdown,
            confidence=narration.confidence_level,
            confidence_score=narration.confidence_score,
            definition_used=narration.definition_used,
            evidence=narration.evidence,
            sanity_checks=narration.sanity_checks,
            sql=state.query_result.final_sql if state.query_result else None,
            row_count=(
                state.query_result.final_result.row_count
                if state.query_result and state.query_result.final_result
                else None
            ),
            columns=(
                state.query_result.final_result.columns
                if state.query_result and state.query_result.final_result
                else []
            ),
            sample_rows=(
                state.query_result.final_result.sample_rows
                if state.query_result and state.query_result.final_result
                else []
            ),
            execution_time_ms=(
                state.query_result.final_result.execution_time_ms
                if state.query_result and state.query_result.final_result
                else None
            ),
            trace_id=state.trace_id,
            suggested_questions=narration.suggested_questions,
        )
    
    def _build_clarification_response(self, state: OrchestratorState) -> AssistantQueryResponse:
        """Build response requesting clarification."""
        intake = state.intake_result
        
        return AssistantQueryResponse(
            success=False,
            answer_markdown="**Clarification needed**\n\n" + "\n".join(
                f"- {q}" for q in intake.clarification_questions
            ),
            confidence=ConfidenceLevel.UNCERTAIN,
            confidence_score=0.0,
            definition_used=intake.clarified_goal,
            evidence=[],
            sanity_checks=[],
            trace_id=state.trace_id,
            error="Clarification needed",
            suggested_questions=intake.clarification_questions,
        )
    
    def _build_error_response(self, state: OrchestratorState, error: str) -> AssistantQueryResponse:
        """Build error response."""
        return AssistantQueryResponse(
            success=False,
            answer_markdown=f"**Error processing request**\n\n{error}",
            confidence=ConfidenceLevel.UNCERTAIN,
            confidence_score=0.0,
            definition_used=state.intake_result.clarified_goal if state.intake_result else "",
            evidence=[],
            sanity_checks=[SanityCheck(
                check_name="orchestration",
                passed=False,
                message=error,
            )],
            trace_id=state.trace_id,
            error=error,
            suggested_questions=["Try rephrasing your question"],
        )
    
    def build_trace(self, state: OrchestratorState) -> RunTrace:
        """Build a full run trace from state."""
        completed_at = datetime.utcnow()
        total_duration = (completed_at - state.started_at).total_seconds() * 1000
        
        return RunTrace(
            trace_id=state.trace_id,
            started_at=state.started_at,
            completed_at=completed_at,
            total_duration_ms=total_duration,
            original_goal=state.intake_result.original_goal if state.intake_result else "",
            db_connection_id=str(self.db_path),
            constraints={},
            agents=state.agent_traces,
            refinement_count=state.refinement_count,
            final_status=AgentStatus.SUCCESS if state.narration_result and state.narration_result.status == AgentStatus.SUCCESS else AgentStatus.FAILED,
            final_answer=state.narration_result.answer_markdown if state.narration_result else None,
            final_error=state.error,
        )
    
    def close(self) -> None:
        """Close all agents."""
        self.query_agent.close()
    
    def __enter__(self) -> "AnalystOrchestrator":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
