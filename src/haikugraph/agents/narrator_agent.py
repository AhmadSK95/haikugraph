"""NarratorAgent for formatting final answer with evidence.

This agent:
1. Formats query results into human-readable markdown
2. Includes evidence supporting the answer
3. Adds confidence level and factors
4. Suggests follow-up questions
"""

from typing import Any

from haikugraph.agents.base import LLMAgent, AgentError
from haikugraph.agents.contracts import (
    AgentStatus,
    AuditResult,
    ConfidenceLevel,
    EvidenceItem,
    IntakeResult,
    NarrationResult,
    QueryPlanResult,
    SanityCheck,
)


NARRATOR_SYSTEM_PROMPT = """You are an expert data analyst who presents query results clearly and professionally.

Your response must:
1. Start with a bold, direct answer to the question
2. Include supporting evidence from the data
3. Format numbers properly (commas, currency symbols)
4. Use markdown tables for multiple rows
5. Add brief interpretation and context
6. Be concise but complete

Output ONLY valid JSON. No prose outside JSON."""


NARRATOR_USER_PROMPT_TEMPLATE = """Present these query results to a business user.

Original Question: {goal}
Intent: {intent_type}

Query Results:
{results_summary}

SQL Used: {sql}

Audit Summary: {audit_summary}

Format your response as:
{{
  "answer_markdown": "**Bold answer line**\\n\\nSupporting details...",
  "answer_summary": "One-line summary of the answer",
  "definition_used": "How the metric/query was defined",
  "evidence": [
    {{"description": "What this shows", "value": "The value", "source": "query result"}}
  ],
  "sanity_checks": [
    {{"check_name": "data completeness", "passed": true, "message": "All expected data present"}}
  ],
  "confidence_level": "high|medium|low|uncertain",
  "confidence_score": 0.0-1.0,
  "confidence_factors": ["Factor 1", "Factor 2"],
  "suggested_questions": ["Follow-up question 1", "Follow-up question 2"],
  "caveats": ["Any limitations or caveats"]
}}

Generate the narration now. Output ONLY the JSON object."""


class NarratorAgent(LLMAgent[NarrationResult]):
    """Agent for formatting final answer with evidence.
    
    This agent takes query results and presents them in a clear,
    professional format with evidence, confidence, and suggestions.
    """
    
    name = "narrator_agent"
    llm_role = "narrator"
    system_prompt = NARRATOR_SYSTEM_PROMPT
    user_prompt_template = NARRATOR_USER_PROMPT_TEMPLATE
    
    @property
    def output_schema(self) -> type[NarrationResult]:
        return NarrationResult
    
    def run(
        self,
        intake_result: IntakeResult,
        query_result: QueryPlanResult,
        audit_result: AuditResult,
    ) -> NarrationResult:
        """Format results into final answer.
        
        Args:
            intake_result: Result from IntakeAgent
            query_result: Result from QueryAgent
            audit_result: Result from AuditAgent
        
        Returns:
            NarrationResult with formatted answer and evidence
        """
        self._start_timer()
        
        try:
            # Handle failed queries
            if query_result.status == AgentStatus.FAILED:
                return self._narrate_failure(intake_result, query_result)
            
            # Build results summary for prompt
            results_summary = self._build_results_summary(query_result)
            audit_summary = self._build_audit_summary(audit_result)
            
            # Build prompt
            user_prompt = self.user_prompt_template.format(
                goal=intake_result.clarified_goal,
                intent_type=intake_result.intent_type,
                results_summary=results_summary,
                sql=query_result.final_sql,
                audit_summary=audit_summary,
            )
            
            # Call LLM
            data = self.call_llm_with_retry(user_prompt)
            
            # Extract and validate fields
            answer_md = data.get("answer_markdown", "No answer generated")
            answer_summary = data.get("answer_summary", answer_md[:100])
            
            # Parse confidence level
            confidence_str = data.get("confidence_level", "medium").lower()
            confidence_level = ConfidenceLevel.MEDIUM
            if confidence_str == "high":
                confidence_level = ConfidenceLevel.HIGH
            elif confidence_str == "low":
                confidence_level = ConfidenceLevel.LOW
            elif confidence_str == "uncertain":
                confidence_level = ConfidenceLevel.UNCERTAIN
            
            # Build evidence items
            evidence = []
            for e in data.get("evidence", []):
                if isinstance(e, dict):
                    evidence.append(EvidenceItem(
                        description=e.get("description", ""),
                        value=str(e.get("value", "")),
                        source=e.get("source", "query result"),
                        sql_reference=query_result.final_sql,
                    ))
            
            # Build sanity checks
            sanity_checks = []
            for c in data.get("sanity_checks", []):
                if isinstance(c, dict):
                    sanity_checks.append(SanityCheck(
                        check_name=c.get("check_name", ""),
                        passed=c.get("passed", True),
                        message=c.get("message", ""),
                    ))
            
            # Add audit checks as sanity checks
            for audit_check in audit_result.checks:
                sanity_checks.append(SanityCheck(
                    check_name=f"audit_{audit_check.check_name}",
                    passed=audit_check.status.value in ["pass", "warn"],
                    message=audit_check.message,
                ))
            
            elapsed = self._stop_timer()
            
            return NarrationResult(
                status=AgentStatus.SUCCESS,
                answer_markdown=answer_md,
                answer_summary=answer_summary,
                definition_used=data.get("definition_used", intake_result.clarified_goal),
                evidence=evidence,
                sanity_checks=sanity_checks,
                confidence_level=confidence_level,
                confidence_score=data.get("confidence_score", 0.7),
                confidence_factors=data.get("confidence_factors", []),
                suggested_questions=data.get("suggested_questions", []),
                caveats=data.get("caveats", []),
                processing_time_ms=elapsed,
            )
        
        except AgentError:
            raise
        except Exception as e:
            elapsed = self._stop_timer()
            return NarrationResult(
                status=AgentStatus.FAILED,
                answer_markdown=f"**Error generating answer**\n\n{str(e)}",
                answer_summary="Error generating answer",
                definition_used=intake_result.clarified_goal,
                evidence=[],
                sanity_checks=[],
                confidence_level=ConfidenceLevel.UNCERTAIN,
                confidence_score=0.0,
                confidence_factors=["Error occurred during narration"],
                suggested_questions=[],
                caveats=[str(e)],
                processing_time_ms=elapsed,
            )
    
    def _narrate_failure(
        self,
        intake: IntakeResult,
        query: QueryPlanResult,
    ) -> NarrationResult:
        """Generate narration for failed query."""
        elapsed = self._stop_timer()
        
        error_msg = "Query execution failed"
        if query.final_result and query.final_result.error:
            error_msg = query.final_result.error
        
        return NarrationResult(
            status=AgentStatus.FAILED,
            answer_markdown=f"**Unable to answer: {intake.clarified_goal}**\n\n"
                           f"The query could not be executed successfully.\n\n"
                           f"Error: {error_msg}",
            answer_summary="Query execution failed",
            definition_used=intake.clarified_goal,
            evidence=[],
            sanity_checks=[SanityCheck(
                check_name="query_execution",
                passed=False,
                message=error_msg,
            )],
            confidence_level=ConfidenceLevel.UNCERTAIN,
            confidence_score=0.0,
            confidence_factors=["Query failed to execute"],
            suggested_questions=[
                "Can you rephrase your question?",
                "What specific data are you looking for?",
            ],
            caveats=[error_msg],
            processing_time_ms=elapsed,
        )
    
    def _build_results_summary(self, query: QueryPlanResult) -> str:
        """Build a text summary of query results for the prompt."""
        if not query.final_result:
            return "No results available"
        
        result = query.final_result
        lines = [
            f"Row count: {result.row_count}",
            f"Columns: {', '.join(result.columns)}",
        ]
        
        if result.sample_rows:
            lines.append("\nSample data:")
            for i, row in enumerate(result.sample_rows[:5], 1):
                row_str = ", ".join(f"{k}={v}" for k, v in row.items())
                lines.append(f"  {i}. {row_str}")
        
        # Add result summary if available
        summary = query.result_summary
        if summary:
            if "single_result" in summary:
                lines.append(f"\nSingle result: {summary['single_result']}")
        
        return "\n".join(lines)
    
    def _build_audit_summary(self, audit: AuditResult) -> str:
        """Build a text summary of audit results."""
        lines = [
            f"Passed: {audit.passed}, Warnings: {audit.warned}, Failed: {audit.failed}",
            f"Overall: {'PASS' if audit.overall_pass else 'NEEDS ATTENTION'}",
        ]
        
        if audit.refinement_suggestions:
            lines.append(f"Suggestions: {', '.join(audit.refinement_suggestions)}")
        
        return "\n".join(lines)
    
    def run_simple(
        self,
        intake_result: IntakeResult,
        query_result: QueryPlanResult,
    ) -> NarrationResult:
        """Simple narration without LLM.
        
        Use when LLM is unavailable.
        """
        self._start_timer()
        
        if query_result.status == AgentStatus.FAILED:
            return self._narrate_failure(intake_result, query_result)
        
        result = query_result.final_result
        if not result:
            elapsed = self._stop_timer()
            return NarrationResult(
                status=AgentStatus.FAILED,
                answer_markdown="No results available",
                answer_summary="No results",
                definition_used=intake_result.clarified_goal,
                evidence=[],
                sanity_checks=[],
                confidence_level=ConfidenceLevel.UNCERTAIN,
                confidence_score=0.0,
                confidence_factors=[],
                suggested_questions=[],
                caveats=[],
                processing_time_ms=elapsed,
            )
        
        # Build simple markdown response
        md_lines = [f"**{intake_result.clarified_goal}**\n"]
        
        if result.row_count == 1 and result.sample_rows:
            # Single result - format as key-value
            row = result.sample_rows[0]
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    md_lines.append(f"- **{key}**: {value:,.2f}" if isinstance(value, float) else f"- **{key}**: {value:,}")
                else:
                    md_lines.append(f"- **{key}**: {value}")
        elif result.row_count > 1 and result.sample_rows:
            # Multiple rows - format as table
            if result.columns:
                header = "| " + " | ".join(result.columns) + " |"
                separator = "| " + " | ".join(["---"] * len(result.columns)) + " |"
                md_lines.extend([header, separator])
                
                for row in result.sample_rows[:10]:
                    values = [str(row.get(c, "")) for c in result.columns]
                    md_lines.append("| " + " | ".join(values) + " |")
                
                if result.row_count > 10:
                    md_lines.append(f"\n*Showing 10 of {result.row_count} rows*")
        else:
            md_lines.append("No data found matching your query.")
        
        elapsed = self._stop_timer()
        
        return NarrationResult(
            status=AgentStatus.SUCCESS,
            answer_markdown="\n".join(md_lines),
            answer_summary=f"Found {result.row_count} results",
            definition_used=intake_result.clarified_goal,
            evidence=[EvidenceItem(
                description="Query result",
                value=f"{result.row_count} rows",
                source="database query",
                sql_reference=query_result.final_sql,
            )],
            sanity_checks=[],
            confidence_level=ConfidenceLevel.MEDIUM,
            confidence_score=0.6,
            confidence_factors=["Simple formatting without LLM analysis"],
            suggested_questions=[],
            caveats=[],
            processing_time_ms=elapsed,
        )
