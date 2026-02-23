"""Policy gates for trust-critical autonomy in dataDa.

Implements:
- UnsupportedConceptDetector: Detect concepts outside known domain
- AntiFabricationGate: Prevent fabricated KPI output
- FutureTimeIntegrityCheck: Block future-time queries returning historical data

All gates return PolicyVerdict with action: "allow", "refuse", "clarify"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Any


# Known domains and their supported concepts
SUPPORTED_DOMAINS = {
    "transactions": {
        "metrics": [
            "payment_amount", "deal_details_amount", "amount_collected",
            "transaction_id", "count", "sum", "avg", "min", "max",
        ],
        "dimensions": [
            "platform_name", "state", "txn_flow", "payment_status",
            "customer_id", "payee_id", "mt103_created_at", "created_at",
            "refund_refund_id",
        ],
        "concepts": [
            "payment", "transaction", "transfer", "wire", "remittance",
            "mt103", "refund", "amount", "revenue",
        ],
    },
    "quotes": {
        "metrics": [
            "total_amount_to_be_paid", "total_additional_charges",
            "forex_markup", "exchange_rate",
        ],
        "dimensions": [
            "source_currency", "destination_currency", "customer_id",
            "created_at",
        ],
        "concepts": [
            "quote", "fx", "forex", "exchange", "rate", "currency",
            "markup", "charges",
        ],
    },
    "customers": {
        "metrics": ["customer_id", "payee_id"],
        "dimensions": [
            "is_university", "type", "status", "address_country",
            "created_at",
        ],
        "concepts": [
            "customer", "client", "user", "payee", "beneficiary",
            "university", "country",
        ],
    },
    "bookings": {
        "metrics": ["booked_amount", "rate", "deal_id"],
        "dimensions": [
            "deal_type", "customer_id", "payee_id", "quote_id",
            "created_at", "updated_at",
        ],
        "concepts": [
            "booking", "deal", "booked", "forward", "spot", "option",
        ],
    },
}

# Concepts that are absolutely NOT supported
UNSUPPORTED_CONCEPTS = [
    "stock", "equity", "bond", "derivative", "futures", "options trading",
    "cryptocurrency", "bitcoin", "ethereum", "nft",
    "insurance", "claim", "premium", "deductible",
    "mortgage", "loan", "interest rate", "credit score",
    "inventory", "warehouse", "supply chain", "logistics",
    "employee", "salary", "payroll", "hr",
    "marketing", "campaign", "click-through", "impression",
    "social media", "followers", "engagement",
    "weather", "temperature", "precipitation",
    "medical", "diagnosis", "prescription",
    "gdp", "inflation", "unemployment",
    "joy score", "satisfaction score", "nps", "churn rate", "lifetime value",
    "retention rate", "engagement score", "sentiment",
]

# Fabrication indicators - phrases that suggest the user wants synthetic data
FABRICATION_TRIGGERS = [
    r"make\s+up",
    r"fabricat",
    r"invent\s+(?:some|a|the)",
    r"create\s+fake",
    r"generate\s+(?:fake|synthetic|dummy|mock)",
    r"pretend",
    r"hypothetical\s+(?:data|numbers|values|scenario)",
    r"imagine\s+(?:that|if)",
    r"what\s+if\s+.*\s+were\s+different",
    r"simulate\s+(?:data|results|numbers)",
    r"ignore\s+(?:the\s+)?(?:data|dataset)",
    r"invent\s+(?:\w+\s+)*(?:kpi|metric|number|data|figure)",
    r"make\s+(?:your\s+)?best\s+(?:guess|estimate)",
    r"if\s+data\s+is\s+missing.*(?:estimate|guess|invent|make\s+up)",
    r"do\s+not\s+tell\s+me\b.*\b(?:missing|unavailable|don'?t\s+have)",
]

# Coercion patterns - attempts to override safety
COERCION_PATTERNS = [
    r"ignore\s+(?:safety|rules|guardrails|policy|guidelines)",
    r"bypass\s+(?:safety|rules|guardrails|checks)",
    r"override\s+(?:safety|the\s+rules|policy)",
    r"don'?t\s+(?:check|validate|verify|refuse)",
    r"skip\s+(?:validation|checks|safety)",
    r"i\s+(?:insist|demand|command)\s+you",
    r"just\s+(?:give\s+me|show\s+me|tell\s+me)\s+(?:any|some)\s+(?:number|data|answer)",
    r"(?:admin|root|superuser)\s+(?:mode|access|override)",
]


@dataclass
class PolicyVerdict:
    """Verdict from a policy gate check."""

    action: str  # "allow", "refuse", "clarify"
    gate: str  # which gate produced this verdict
    reason: str  # human-readable explanation
    details: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_unsupported_concept(question: str) -> PolicyVerdict:
    """Detect if the question references concepts outside the supported domain.

    Returns "refuse" if the question is clearly about unsupported topics,
    "clarify" if ambiguous, "allow" if supported.
    """
    lower = question.lower()

    # Check for explicitly unsupported concepts
    matched_unsupported = []
    for concept in UNSUPPORTED_CONCEPTS:
        if concept in lower:
            matched_unsupported.append(concept)

    if matched_unsupported:
        return PolicyVerdict(
            action="refuse",
            gate="unsupported_concept",
            reason=(
                f"This question references concepts not available in the data: "
                f"{', '.join(matched_unsupported)}. "
                f"The system supports: transactions, quotes, customers, and bookings data."
            ),
            details={"matched_unsupported": matched_unsupported},
            confidence=0.95,
        )

    # Check if question mentions any supported concept
    all_supported = set()
    for domain_concepts in SUPPORTED_DOMAINS.values():
        all_supported.update(domain_concepts["concepts"])

    has_supported = any(
        re.search(rf"\b{re.escape(concept)}\b", lower)
        for concept in all_supported
    )

    if not has_supported:
        # Check for very generic questions that might still be answerable
        generic_ok = any(w in lower for w in [
            "how many", "total", "count", "list", "show", "what",
            "average", "sum", "data", "table", "column", "schema",
        ])
        if generic_ok:
            return PolicyVerdict(
                action="allow",
                gate="unsupported_concept",
                reason="Generic analytical question, allowing with domain inference.",
                confidence=0.7,
            )

        # Short queries with time/filter references are likely follow-ups
        followup_signals = [
            "just", "only", "but", "and", "also", "instead", "that",
            "those", "these", "them", "it", "its", "their",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            "month", "year", "week", "day", "quarter",
            "last", "previous", "next", "current", "this",
            "platform", "status", "type", "country", "state",
            "b2c", "b2b", "completed", "pending", "failed",
            "break", "breakdown", "split", "group", "by",
        ]
        is_likely_followup = (
            len(lower.split()) <= 6
            or any(w in lower for w in followup_signals)
        )
        if is_likely_followup:
            return PolicyVerdict(
                action="allow",
                gate="unsupported_concept",
                reason="Likely follow-up query, allowing with context resolution.",
                confidence=0.6,
            )

        return PolicyVerdict(
            action="clarify",
            gate="unsupported_concept",
            reason=(
                "Could not identify a supported domain in your question. "
                "The system supports: transactions, quotes, customers, and bookings. "
                "Could you rephrase with specific domain context?"
            ),
            details={"matched_supported": False},
            confidence=0.5,
        )

    return PolicyVerdict(
        action="allow",
        gate="unsupported_concept",
        reason="Supported domain concept detected.",
        details={"has_supported_concept": True},
        confidence=0.9,
    )


def check_fabrication_request(question: str) -> PolicyVerdict:
    """Detect if the user is requesting fabricated/synthetic data.

    The system must NEVER generate synthetic KPI values.
    """
    lower = question.lower()

    for pattern in FABRICATION_TRIGGERS:
        if re.search(pattern, lower):
            return PolicyVerdict(
                action="refuse",
                gate="anti_fabrication",
                reason=(
                    "This request appears to ask for fabricated or synthetic data. "
                    "The system only reports actual data from the database. "
                    "It cannot generate, simulate, or invent data values."
                ),
                details={"matched_pattern": pattern},
                confidence=0.92,
            )

    return PolicyVerdict(
        action="allow",
        gate="anti_fabrication",
        reason="No fabrication request detected.",
        confidence=0.95,
    )


def check_coercion_attempt(question: str) -> PolicyVerdict:
    """Detect prompt injection or coercion attempts."""
    lower = question.lower()

    for pattern in COERCION_PATTERNS:
        if re.search(pattern, lower):
            return PolicyVerdict(
                action="refuse",
                gate="anti_coercion",
                reason=(
                    "This request contains patterns that attempt to override "
                    "safety policies. The system applies data quality and safety "
                    "checks on all queries to protect data integrity."
                ),
                details={"matched_pattern": pattern},
                confidence=0.88,
            )

    return PolicyVerdict(
        action="allow",
        gate="anti_coercion",
        reason="No coercion attempt detected.",
        confidence=0.95,
    )


def check_future_time_integrity(question: str, reference_date: date | None = None) -> PolicyVerdict:
    """Detect queries about future time periods that cannot have data.

    Prevents returning historical data when the user asks about the future.
    """
    if reference_date is None:
        reference_date = date.today()

    lower = question.lower()

    # Future month/year patterns
    future_patterns = [
        (r"(?:in|for|during)\s+(\w+)\s+(\d{4})", "month_year"),
        (r"(?:next|upcoming|future)\s+(?:month|quarter|year)", "relative_future"),
        (r"(\d{4})\s+(?:forecast|prediction|projection)", "forecast"),
        (r"(?:predict|forecast|project)\s+", "forecast_verb"),
        (r"\b(\d{4})\b", "bare_year"),
    ]

    for pattern, ptype in future_patterns:
        match = re.search(pattern, lower)
        if match:
            if ptype == "month_year":
                month_name = match.group(1)
                year = int(match.group(2))
                month_map = {
                    "january": 1, "february": 2, "march": 3, "april": 4,
                    "may": 5, "june": 6, "july": 7, "august": 8,
                    "september": 9, "october": 10, "november": 11, "december": 12,
                    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
                    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
                    "oct": 10, "nov": 11, "dec": 12,
                }
                month_num = month_map.get(month_name)
                if month_num and year >= reference_date.year:
                    query_date = date(year, month_num, 1)
                    if query_date > reference_date:
                        return PolicyVerdict(
                            action="refuse",
                            gate="future_time_integrity",
                            reason=(
                                f"The requested time period ({month_name.title()} {year}) "
                                f"is in the future. No data exists for this period. "
                                f"The system cannot provide forecasts or predictions."
                            ),
                            details={
                                "requested_period": f"{month_name} {year}",
                                "reference_date": reference_date.isoformat(),
                            },
                            confidence=0.95,
                        )

            elif ptype == "relative_future":
                return PolicyVerdict(
                    action="refuse",
                    gate="future_time_integrity",
                    reason=(
                        "The system only reports on historical data. "
                        "Future periods (next month/quarter/year) have no data. "
                        "The system cannot provide forecasts or predictions."
                    ),
                    details={"pattern_type": "relative_future"},
                    confidence=0.90,
                )

            elif ptype in ("forecast", "forecast_verb"):
                return PolicyVerdict(
                    action="refuse",
                    gate="future_time_integrity",
                    reason=(
                        "The system reports on actual data only. "
                        "Forecasting, prediction, and projection capabilities "
                        "are not supported."
                    ),
                    details={"pattern_type": ptype},
                    confidence=0.92,
                )

            elif ptype == "bare_year":
                year = int(match.group(1))
                if year > reference_date.year:
                    return PolicyVerdict(
                        action="refuse",
                        gate="future_time_integrity",
                        reason=(
                            f"The requested year ({year}) is in the future. "
                            f"No data exists for this period. "
                            f"The system only reports on historical data."
                        ),
                        details={"requested_year": year, "reference_date": reference_date.isoformat()},
                        confidence=0.95,
                    )

    return PolicyVerdict(
        action="allow",
        gate="future_time_integrity",
        reason="No future time integrity issues detected.",
        confidence=0.9,
    )


def run_all_policy_gates(
    question: str,
    reference_date: date | None = None,
) -> list[PolicyVerdict]:
    """Run all policy gates on a question.

    Returns list of all verdicts. If any verdict is "refuse" or "clarify",
    the query should not proceed to SQL generation.
    """
    verdicts = [
        check_unsupported_concept(question),
        check_fabrication_request(question),
        check_coercion_attempt(question),
        check_future_time_integrity(question, reference_date),
    ]
    return verdicts


def get_blocking_verdict(verdicts: list[PolicyVerdict]) -> PolicyVerdict | None:
    """Get the first blocking verdict from a list of verdicts.

    Returns the first "refuse" verdict, or the first "clarify" verdict,
    or None if all pass.
    """
    # Refuse takes priority
    for v in verdicts:
        if v.action == "refuse":
            return v
    # Then clarify
    for v in verdicts:
        if v.action == "clarify":
            return v
    return None


def format_refusal_response(verdict: PolicyVerdict) -> str:
    """Format a policy verdict into a user-facing refusal message."""
    if verdict.action == "refuse":
        return (
            f"**I cannot answer this question.**\n\n"
            f"{verdict.reason}\n\n"
            f"*Policy gate: {verdict.gate}*"
        )
    elif verdict.action == "clarify":
        return (
            f"**I need clarification before I can answer.**\n\n"
            f"{verdict.reason}\n\n"
            f"*Policy gate: {verdict.gate}*"
        )
    return ""
