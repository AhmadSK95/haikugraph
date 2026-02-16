"""IntakeAgent for goal clarification and extraction.

This agent is the first in the analyst loop. It:
1. Parses the user's natural language goal
2. Extracts metrics, dimensions, and time windows
3. Detects intent type (metric, grouped_metric, comparison, lookup, diagnostic)
4. Identifies if clarification is needed
"""

from haikugraph.agents.base import LLMAgent, AgentError
from haikugraph.agents.contracts import (
    AgentStatus,
    ExtractedDimension,
    ExtractedMetric,
    ExtractedTimeWindow,
    IntakeResult,
)


INTAKE_SYSTEM_PROMPT = """You are an expert data analyst intake specialist. Your job is to understand what the user wants to analyze and extract the structured components of their request.

You MUST output valid JSON following the exact schema. No prose, no markdown."""


INTAKE_USER_PROMPT_TEMPLATE = """Analyze this data analysis request and extract its components:

User's Goal: "{goal}"

Extract:
1. Intent type: One of:
   - "metric": Single aggregated value (e.g., "total revenue", "count of orders")
   - "grouped_metric": Aggregation by dimension (e.g., "revenue by customer", "orders per month")
   - "comparison": Comparing periods/cohorts (e.g., "this month vs last month")
   - "lookup": Raw data listing (e.g., "show me recent transactions")
   - "diagnostic": Health/anomaly analysis (e.g., "why did sales drop?")

2. Metrics: What numerical measures are being asked for?
   - name: Descriptive name (e.g., "total_revenue", "transaction_count")
   - aggregation: sum, count, avg, min, max, or count_distinct
   - column_hint: Likely column name if you can infer it

3. Dimensions: What groupings or breakdowns are requested?
   - name: Dimension name (e.g., "customer", "month")
   - is_time_dimension: true if it's a time-based grouping
   - time_grain: day, week, month, quarter, year (if time dimension)

4. Time window: Any time filters mentioned?
   - has_time_filter: true/false
   - period_type: "absolute" (specific dates), "relative" (this month, last week), or "comparison" (vs last month)
   - relative_period: For relative periods (today, yesterday, this_week, last_week, this_month, last_month, this_year, last_year)
   - comparison_period: For comparisons, what's being compared to

5. Clarification: Is the request unclear or ambiguous?
   - needs_clarification: true/false
   - clarification_questions: List of questions to ask if unclear

Output JSON schema:
{{
  "status": "success" | "needs_clarification",
  "original_goal": "<exact user goal>",
  "clarified_goal": "<normalized/clarified version>",
  "intent_type": "<intent>",
  "metrics": [
    {{"name": "...", "aggregation": "...", "column_hint": "..." | null, "definition": "..." | null}}
  ],
  "dimensions": [
    {{"name": "...", "column_hint": "..." | null, "is_time_dimension": true/false, "time_grain": "..." | null}}
  ],
  "time_window": {{
    "has_time_filter": true/false,
    "period_type": "..." | null,
    "start_date": "..." | null,
    "end_date": "..." | null,
    "relative_period": "..." | null,
    "comparison_period": "..." | null
  }} | null,
  "filters": [],
  "needs_clarification": true/false,
  "clarification_questions": [],
  "confidence": 0.0-1.0,
  "reasoning": "<brief explanation>"
}}

Output ONLY the JSON object."""


class IntakeAgent(LLMAgent[IntakeResult]):
    """Agent for intake and goal clarification.
    
    This agent extracts structured information from the user's natural
    language goal including metrics, dimensions, time windows, and filters.
    """
    
    name = "intake_agent"
    llm_role = "intent"  # Uses intent classification model
    system_prompt = INTAKE_SYSTEM_PROMPT
    user_prompt_template = INTAKE_USER_PROMPT_TEMPLATE
    
    @property
    def output_schema(self) -> type[IntakeResult]:
        return IntakeResult
    
    def run(self, goal: str) -> IntakeResult:
        """Extract structured components from user's goal.
        
        Args:
            goal: User's natural language goal/question
        
        Returns:
            IntakeResult with extracted metrics, dimensions, etc.
        """
        self._start_timer()
        
        try:
            # Call LLM
            user_prompt = self.user_prompt_template.format(goal=goal)
            data = self.call_llm_with_retry(user_prompt)
            
            # Ensure required fields
            data["original_goal"] = goal
            data.setdefault("clarified_goal", goal)
            data.setdefault("intent_type", "unknown")
            data.setdefault("confidence", 0.7)
            data.setdefault("reasoning", "Extracted from user goal")
            
            # Convert nested objects
            metrics = []
            for m in data.get("metrics", []):
                if isinstance(m, dict):
                    metrics.append(ExtractedMetric(
                        name=m.get("name", "unknown"),
                        aggregation=m.get("aggregation", "count"),
                        column_hint=m.get("column_hint"),
                        definition=m.get("definition"),
                    ))
            data["metrics"] = metrics
            
            dimensions = []
            for d in data.get("dimensions", []):
                if isinstance(d, dict):
                    dimensions.append(ExtractedDimension(
                        name=d.get("name", "unknown"),
                        column_hint=d.get("column_hint"),
                        is_time_dimension=d.get("is_time_dimension", False),
                        time_grain=d.get("time_grain"),
                    ))
            data["dimensions"] = dimensions
            
            # Handle time window
            tw = data.get("time_window")
            if tw and isinstance(tw, dict):
                data["time_window"] = ExtractedTimeWindow(
                    has_time_filter=tw.get("has_time_filter", False),
                    period_type=tw.get("period_type"),
                    start_date=tw.get("start_date"),
                    end_date=tw.get("end_date"),
                    relative_period=tw.get("relative_period"),
                    comparison_period=tw.get("comparison_period"),
                )
            else:
                data["time_window"] = None
            
            # Set status
            if data.get("needs_clarification", False):
                data["status"] = AgentStatus.NEEDS_CLARIFICATION
            else:
                data["status"] = AgentStatus.SUCCESS
            
            # Add processing time
            data["processing_time_ms"] = self._stop_timer()
            
            return IntakeResult(**data)
        
        except AgentError:
            raise
        except Exception as e:
            elapsed = self._stop_timer()
            # Return a minimal result indicating failure
            return IntakeResult(
                status=AgentStatus.FAILED,
                original_goal=goal,
                clarified_goal=goal,
                intent_type="unknown",
                confidence=0.0,
                reasoning=f"Failed to process goal: {str(e)}",
                processing_time_ms=elapsed,
            )
    
    def run_simple(self, goal: str) -> IntakeResult:
        """Simple rule-based extraction without LLM.
        
        Use this when LLM is unavailable or for simple queries.
        
        Args:
            goal: User's natural language goal
        
        Returns:
            IntakeResult with basic extraction
        """
        self._start_timer()
        
        goal_lower = goal.lower()
        
        # Detect intent type
        intent_type = "unknown"
        if any(kw in goal_lower for kw in ["vs", "versus", "compare", "compared to"]):
            intent_type = "comparison"
        elif any(kw in goal_lower for kw in ["by", "per", "breakdown", "each"]):
            intent_type = "grouped_metric"
        elif any(kw in goal_lower for kw in ["total", "sum", "count", "average", "how many", "how much"]):
            intent_type = "metric"
        elif any(kw in goal_lower for kw in ["show", "list", "display", "get"]):
            intent_type = "lookup"
        elif any(kw in goal_lower for kw in ["why", "problem", "issue", "wrong"]):
            intent_type = "diagnostic"
        
        # Detect time window
        has_time = False
        relative_period = None
        if "today" in goal_lower:
            has_time = True
            relative_period = "today"
        elif "yesterday" in goal_lower:
            has_time = True
            relative_period = "yesterday"
        elif "this month" in goal_lower:
            has_time = True
            relative_period = "this_month"
        elif "last month" in goal_lower:
            has_time = True
            relative_period = "last_month"
        elif "this year" in goal_lower:
            has_time = True
            relative_period = "this_year"
        elif "last year" in goal_lower:
            has_time = True
            relative_period = "last_year"
        
        time_window = None
        if has_time:
            time_window = ExtractedTimeWindow(
                has_time_filter=True,
                period_type="relative",
                relative_period=relative_period,
            )
        
        elapsed = self._stop_timer()
        
        return IntakeResult(
            status=AgentStatus.SUCCESS,
            original_goal=goal,
            clarified_goal=goal,
            intent_type=intent_type,
            metrics=[],
            dimensions=[],
            time_window=time_window,
            filters=[],
            needs_clarification=False,
            clarification_questions=[],
            confidence=0.5,  # Lower confidence for rule-based
            reasoning="Simple rule-based extraction",
            processing_time_ms=elapsed,
        )
