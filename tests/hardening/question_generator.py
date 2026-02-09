"""
Combinatorial question generator for HaikuGraph hardening.

Generates 200-5000 natural language questions by combining dimensions:
- Intents (revenue, count, unique customers, avg, breakdown)
- Time windows (today, yesterday, this month, last month, this year, etc)
- Comparisons (X vs Y, MoM, YoY)
- Breakdowns (by day, week, month, platform, etc)
- Filters (status, platform, date ranges)
"""

import itertools
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class IntentType(Enum):
    """Question intent types"""
    SCALAR_METRIC = "scalar_metric"  # Single number: "total revenue"
    SCALAR_COUNT = "scalar_count"  # Single count: "how many transactions"
    SCALAR_UNIQUE = "scalar_unique"  # Unique count: "unique customers"
    SCALAR_AVG = "scalar_avg"  # Average: "average ticket size"
    BREAKDOWN = "breakdown"  # Grouped: "revenue by platform"
    TREND = "trend"  # Time series: "revenue by month"
    COMPARISON_SCALAR = "comparison_scalar"  # "this month vs last month" → 2 numbers
    COMPARISON_MOM = "comparison_mom"  # "month over month"
    COMPARISON_YOY = "comparison_yoy"  # "year over year"
    TOP_K = "top_k"  # "top 10 platforms"


class TimeWindow(Enum):
    """Time window specifications"""
    # Specific time points
    TODAY = "today"
    YESTERDAY = "yesterday"
    
    # Calendar periods
    THIS_WEEK = "this week"
    LAST_WEEK = "last week"
    THIS_MONTH = "this month"
    LAST_MONTH = "last month"
    THIS_YEAR = "this year"
    LAST_YEAR = "last year"
    
    # Rolling windows
    LAST_7_DAYS = "last 7 days"
    LAST_30_DAYS = "last 30 days"
    LAST_90_DAYS = "last 90 days"
    
    # Multi-month rolling
    LAST_3_MONTHS = "last 3 months"
    LAST_6_MONTHS = "last 6 months"
    LAST_12_MONTHS = "last 12 months"
    
    # Specific months (for testing)
    DECEMBER = "December"
    JANUARY = "January"
    
    # No time filter
    ALL_TIME = None


class BreakdownType(Enum):
    """Breakdown dimensions"""
    NONE = None
    BY_DAY = "by day"
    BY_WEEK = "by week"
    BY_MONTH = "by month"
    BY_PLATFORM = "by platform"
    BY_STATUS = "by status"


@dataclass
class QuestionSpec:
    """Specification for a generated question"""
    intent: IntentType
    metric: str  # "revenue", "transactions", "customers", etc
    time_window: Optional[TimeWindow] = None
    breakdown: Optional[BreakdownType] = None
    filter_status: Optional[str] = None
    filter_platform: Optional[str] = None
    comparison_window: Optional[TimeWindow] = None  # For comparisons
    top_k: Optional[int] = None
    
    # Expected output characteristics (for oracle)
    expected_shape: str = field(default="scalar")  # scalar, series, grouped
    expected_group_by: bool = field(default=False)
    expected_distinct: bool = field(default=False)
    
    def to_natural_language(self) -> str:
        """Convert spec to natural language question"""
        # Base metric phrase
        if self.intent == IntentType.SCALAR_METRIC:
            if self.metric == "revenue":
                phrase = "What is the total revenue"
            elif self.metric == "volume":
                phrase = "What is the total volume"
            else:
                phrase = f"What is the total {self.metric}"
        
        elif self.intent == IntentType.SCALAR_COUNT:
            if self.metric == "transactions":
                phrase = "How many transactions"
            elif self.metric == "customers":
                phrase = "How many customers"
            else:
                phrase = f"How many {self.metric}"
        
        elif self.intent == IntentType.SCALAR_UNIQUE:
            if self.metric == "customers":
                phrase = "How many unique customers"
            elif self.metric == "transactions":
                phrase = "How many unique transactions"
            else:
                phrase = f"How many unique {self.metric}"
        
        elif self.intent == IntentType.SCALAR_AVG:
            if self.metric == "revenue":
                phrase = "What is the average revenue per transaction"
            elif self.metric == "ticket_size":
                phrase = "What is the average ticket size"
            else:
                phrase = f"What is the average {self.metric}"
        
        elif self.intent == IntentType.BREAKDOWN:
            if self.metric == "revenue":
                phrase = "What is the revenue"
            elif self.metric == "transactions":
                phrase = "How many transactions"
            else:
                phrase = f"What is the {self.metric}"
        
        elif self.intent == IntentType.TREND:
            phrase = f"Show me {self.metric}"
        
        elif self.intent in [IntentType.COMPARISON_SCALAR, IntentType.COMPARISON_MOM, IntentType.COMPARISON_YOY]:
            phrase = f"Compare {self.metric}"
        
        elif self.intent == IntentType.TOP_K:
            phrase = f"Show me top {self.top_k or 10} platforms by {self.metric}"
        
        # Add filters
        if self.filter_status:
            phrase += f" with status {self.filter_status}"
        
        if self.filter_platform:
            phrase += f" for {self.filter_platform} platform"
        
        # Add time window
        if self.time_window and self.time_window != TimeWindow.ALL_TIME:
            phrase += f" {self.time_window.value}"
        
        # Add breakdown
        if self.breakdown and self.breakdown != BreakdownType.NONE:
            phrase += f" {self.breakdown.value}"
        
        # Add comparison window
        if self.comparison_window:
            if self.intent == IntentType.COMPARISON_SCALAR:
                phrase += f" vs {self.comparison_window.value}"
            elif self.intent == IntentType.COMPARISON_MOM:
                phrase += " month over month"
            elif self.intent == IntentType.COMPARISON_YOY:
                phrase += " year over year"
        
        return phrase + "?"


def generate_question_matrix(
    max_questions: int = 500,
    include_comparisons: bool = True,
    include_breakdowns: bool = True,
    include_filters: bool = True,
) -> List[QuestionSpec]:
    """
    Generate combinatorial matrix of questions.
    
    Args:
        max_questions: Maximum number of questions to generate
        include_comparisons: Include comparison queries
        include_breakdowns: Include breakdown queries
        include_filters: Include filtered queries
    
    Returns:
        List of QuestionSpec objects
    """
    questions = []
    
    # ========================================================================
    # 1. SCALAR METRICS (no GROUP BY expected)
    # ========================================================================
    scalar_metrics = ["revenue", "volume"]
    scalar_times = [
        TimeWindow.ALL_TIME,
        TimeWindow.TODAY,
        TimeWindow.THIS_WEEK,
        TimeWindow.THIS_MONTH,
        TimeWindow.LAST_MONTH,
        TimeWindow.THIS_YEAR,
        TimeWindow.LAST_7_DAYS,
        TimeWindow.LAST_30_DAYS,
        TimeWindow.DECEMBER,
    ]
    
    for metric, time_window in itertools.product(scalar_metrics, scalar_times):
        questions.append(QuestionSpec(
            intent=IntentType.SCALAR_METRIC,
            metric=metric,
            time_window=time_window,
            breakdown=BreakdownType.NONE,
            expected_shape="scalar",
            expected_group_by=False,
        ))
    
    # ========================================================================
    # 2. SCALAR COUNTS (no GROUP BY expected)
    # ========================================================================
    count_entities = ["transactions", "customers"]
    for entity, time_window in itertools.product(count_entities, scalar_times[:5]):
        questions.append(QuestionSpec(
            intent=IntentType.SCALAR_COUNT,
            metric=entity,
            time_window=time_window,
            expected_shape="scalar",
            expected_group_by=False,
        ))
    
    # ========================================================================
    # 3. UNIQUE COUNTS (COUNT DISTINCT expected)
    # ========================================================================
    unique_entities = ["customers", "transactions"]
    for entity, time_window in itertools.product(unique_entities, scalar_times[:5]):
        questions.append(QuestionSpec(
            intent=IntentType.SCALAR_UNIQUE,
            metric=entity,
            time_window=time_window,
            expected_shape="scalar",
            expected_group_by=False,
            expected_distinct=True,
        ))
    
    # ========================================================================
    # 4. AVERAGES (no GROUP BY unless breakdown)
    # ========================================================================
    avg_metrics = ["revenue", "ticket_size"]
    for metric, time_window in itertools.product(avg_metrics, scalar_times[:5]):
        questions.append(QuestionSpec(
            intent=IntentType.SCALAR_AVG,
            metric=metric,
            time_window=time_window,
            expected_shape="scalar",
            expected_group_by=False,
        ))
    
    if not include_breakdowns:
        return questions[:max_questions]
    
    # ========================================================================
    # 5. BREAKDOWNS (GROUP BY expected)
    # ========================================================================
    breakdown_dims = [BreakdownType.BY_PLATFORM, BreakdownType.BY_STATUS]
    breakdown_metrics = ["revenue", "transactions"]
    breakdown_times = [TimeWindow.ALL_TIME, TimeWindow.THIS_MONTH, TimeWindow.LAST_MONTH]
    
    for metric, breakdown, time_window in itertools.product(
        breakdown_metrics, breakdown_dims, breakdown_times
    ):
        questions.append(QuestionSpec(
            intent=IntentType.BREAKDOWN,
            metric=metric,
            time_window=time_window,
            breakdown=breakdown,
            expected_shape="grouped",
            expected_group_by=True,
        ))
    
    # ========================================================================
    # 6. TRENDS (GROUP BY time bucket expected)
    # ========================================================================
    trend_breakdowns = [BreakdownType.BY_DAY, BreakdownType.BY_WEEK, BreakdownType.BY_MONTH]
    trend_metrics = ["revenue", "transactions"]
    trend_times = [TimeWindow.THIS_MONTH, TimeWindow.LAST_90_DAYS, TimeWindow.THIS_YEAR]
    
    for metric, breakdown, time_window in itertools.product(
        trend_metrics, trend_breakdowns, trend_times
    ):
        questions.append(QuestionSpec(
            intent=IntentType.TREND,
            metric=metric,
            time_window=time_window,
            breakdown=breakdown,
            expected_shape="series",
            expected_group_by=True,
        ))
    
    if not include_comparisons:
        return questions[:max_questions]
    
    # ========================================================================
    # 7. SCALAR COMPARISONS (NO GROUP BY - should return 2 scalars or delta)
    # ========================================================================
    comparison_pairs = [
        (TimeWindow.THIS_MONTH, TimeWindow.LAST_MONTH),
        (TimeWindow.THIS_WEEK, TimeWindow.LAST_WEEK),
        (TimeWindow.THIS_YEAR, TimeWindow.LAST_YEAR),
    ]
    comparison_metrics = ["revenue", "transactions"]
    
    for metric, (period1, period2) in itertools.product(comparison_metrics, comparison_pairs):
        questions.append(QuestionSpec(
            intent=IntentType.COMPARISON_SCALAR,
            metric=metric,
            time_window=period1,
            comparison_window=period2,
            expected_shape="comparison",  # Special: 2 scalars + delta
            expected_group_by=False,  # CRITICAL: No GROUP BY for scalar comparisons
        ))
    
    # ========================================================================
    # 8. TOP-K (GROUP BY + ORDER BY + LIMIT)
    # ========================================================================
    topk_values = [5, 10]
    topk_metrics = ["revenue", "transactions"]
    
    for metric, k in itertools.product(topk_metrics, topk_values):
        questions.append(QuestionSpec(
            intent=IntentType.TOP_K,
            metric=metric,
            top_k=k,
            expected_shape="grouped",
            expected_group_by=True,
        ))
    
    if not include_filters:
        return questions[:max_questions]
    
    # ========================================================================
    # 9. FILTERED QUERIES
    # ========================================================================
    filter_statuses = ["success", "failed"]
    filter_platforms = ["B2B", "B2C-APP"]
    
    for status in filter_statuses:
        questions.append(QuestionSpec(
            intent=IntentType.SCALAR_METRIC,
            metric="revenue",
            time_window=TimeWindow.THIS_MONTH,
            filter_status=status,
            expected_shape="scalar",
            expected_group_by=False,
        ))
    
    for platform in filter_platforms:
        questions.append(QuestionSpec(
            intent=IntentType.SCALAR_COUNT,
            metric="transactions",
            time_window=TimeWindow.LAST_MONTH,
            filter_platform=platform,
            expected_shape="scalar",
            expected_group_by=False,
        ))
    
    # Limit to max_questions
    return questions[:max_questions]


def generate_question_variations(base_questions: List[QuestionSpec]) -> List[str]:
    """
    Generate multiple natural language variations for each question spec.
    
    Returns list of (question_text, spec) tuples.
    """
    variations = []
    
    for spec in base_questions:
        # Generate primary question
        primary = spec.to_natural_language()
        variations.append((primary, spec))
        
        # Generate variations based on intent
        if spec.intent == IntentType.SCALAR_METRIC:
            if spec.metric == "revenue":
                variations.append((f"Total revenue {spec.time_window.value if spec.time_window else ''}?", spec))
                variations.append((f"Revenue total {spec.time_window.value if spec.time_window else ''}?", spec))
        
        elif spec.intent == IntentType.SCALAR_COUNT:
            if spec.metric == "transactions":
                variations.append((f"Count of transactions {spec.time_window.value if spec.time_window else ''}?", spec))
                variations.append((f"Number of transactions {spec.time_window.value if spec.time_window else ''}?", spec))
        
        elif spec.intent == IntentType.COMPARISON_SCALAR:
            # Critical variations for comparison testing
            variations.append((
                f"Compare {spec.metric} {spec.time_window.value if spec.time_window else ''} to {spec.comparison_window.value if spec.comparison_window else ''}?",
                spec
            ))
            variations.append((
                f"{spec.metric.capitalize()} {spec.time_window.value if spec.time_window else ''} vs {spec.comparison_window.value if spec.comparison_window else ''}?",
                spec
            ))
    
    return variations


if __name__ == "__main__":
    # Generate and print sample questions
    questions = generate_question_matrix(max_questions=100)
    
    print(f"Generated {len(questions)} question specifications\\n")
    print("=" * 80)
    print("Sample Questions:")
    print("=" * 80)
    
    for i, spec in enumerate(questions[:20], 1):
        nl_question = spec.to_natural_language()
        print(f"{i}. {nl_question}")
        print(f"   Intent: {spec.intent.value}, Shape: {spec.expected_shape}, GROUP BY: {spec.expected_group_by}")
        print()
