# Query Patterns Guide

This guide shows how to write natural-language queries that work correctly with HaikuGraph's refined handling of timestamps and comparisons.

## VARCHAR Timestamp Queries

HaikuGraph now automatically handles VARCHAR timestamp columns. You don't need to do anything special - just ask your question naturally.

### ✅ Working Patterns

**Monthly aggregations with VARCHAR timestamps:**
```
"Show me monthly revenue"
"Monthly unique customers"
"Revenue by month for this year"
```

**Time constraints with VARCHAR timestamps:**
```
"Revenue this month"
"Customers from last week"
"Orders in the last 30 days"
```

**Time bucketing with VARCHAR timestamps:**
```
"Group revenue by quarter"
"Weekly sales totals"
"Daily active users for last month"
```

### Technical Details

- If your timestamp column is VARCHAR, HaikuGraph automatically wraps it with `TRY_CAST(col AS TIMESTAMP)`
- If your timestamp column is already TIMESTAMP/DATE, HaikuGraph uses it directly (zero overhead)
- No configuration needed - works automatically

## Comparison Queries

HaikuGraph distinguishes between **scalar comparisons** (two totals) and **time-series comparisons** (aligned trends).

### ✅ Scalar Comparisons (Default)

Use these patterns for side-by-side comparison of totals:

**Month-over-month:**
```
"Revenue this month vs last month"
"This month vs last month sales"
"Compare customers this month to last month"
```

**Year-over-year:**
```
"Revenue this year vs last year"
"This year vs last year profit"
"Compare customers this year to last year"
```

**Other periods:**
```
"This week vs last week orders"
"Q1 vs Q2 revenue"
"Today vs yesterday users"
```

**Result format:** Two scalar values
```json
{
  "SQ1_current": {"sum_revenue": 50000},
  "SQ2_comparison": {"sum_revenue": 45000}
}
```

### ✅ Time-Series Comparisons (Explicit)

Use these patterns for aligned month-by-month or year-by-year trends:

**Monthly trends:**
```
"Monthly revenue this year vs last year"
"Show monthly customers for this year and last year"
"Compare monthly sales trends this year vs last year"
```

**Other time-series:**
```
"Weekly revenue this quarter vs last quarter"
"Daily users this month vs last month"
"Quarterly revenue for past 2 years"
```

**Result format:** Aligned time series
```json
{
  "month": "2024-01",
  "this_year_revenue": 5000,
  "last_year_revenue": 4500
}
```

### Pattern Keywords

**Scalar comparison indicators:**
- "vs" / "versus" / "compared to"
- "this [period] vs last [period]"
- NO time granularity words

**Time-series comparison indicators:**
- "monthly" / "weekly" / "daily"
- "by month" / "by week" / "by day"
- "trend" / "over time"
- "breakdown by [time unit]"

## Unique Customer Queries

### ✅ Total Unique Customers (Scalar)
```
"How many unique customers?"
"Total unique customers"
"Count distinct customers"
"Unique customer count for this year"
```

**Result:** Single scalar value
```json
{"count_distinct_customer_id": 1250}
```

### ✅ Monthly Unique Customers (Time-Series)
```
"Monthly unique customers"
"Unique customers by month"
"Show unique customer count per month"
"Monthly breakdown of unique customers"
```

**Result:** Monthly time series
```json
[
  {"month": "2024-01", "count_distinct_customer_id": 120},
  {"month": "2024-02", "count_distinct_customer_id": 135}
]
```

## Common Pitfalls

### ❌ Ambiguous Queries

**Problem:** "Customers this month vs last month"
- Could mean: total unique customers (scalar)
- Could mean: daily unique customers (time-series)

**Solution:** Be explicit:
- "Total unique customers this month vs last month" → scalar
- "Daily unique customers this month vs last month" → time-series

### ❌ Mixed Patterns

**Problem:** "Monthly revenue vs last year"
- "Monthly" suggests time-series
- "vs last year" suggests scalar comparison

**Solution:** Clarify intent:
- "Monthly revenue this year vs last year" → time-series comparison
- "Total revenue this year vs last year" → scalar comparison

## Advanced Examples

### Multi-Metric Comparisons
```
"Revenue and profit this quarter vs last quarter"
```
→ Produces two scalar values for each metric

### Grouped Comparisons
```
"Revenue by product category this year vs last year"
```
→ Produces scalar comparison for each category

### Time-Constrained Aggregations
```
"Monthly unique customers for orders over $100"
```
→ Produces monthly time-series with filter applied

### Complex Time Windows
```
"Revenue last 30 days vs previous 30 days"
```
→ Produces two scalar totals with custom time windows

## Testing Your Queries

To verify query behavior, check the generated plan:

1. **Check for group_by field:**
   - Present → time-series result
   - Absent → scalar result

2. **Check constraints:**
   - Comparison queries MUST have `applies_to` on all time constraints
   - Each subquestion gets its own time constraint

3. **Check aggregations:**
   - `distinct: true` → COUNT(DISTINCT col)
   - No `distinct` field → regular aggregation

## Quick Reference Table

| Query Type | Example | Has group_by? | Result Type |
|------------|---------|---------------|-------------|
| Scalar comparison | "Revenue this month vs last" | No | 2 totals |
| Time-series comparison | "Monthly revenue this year vs last" | Yes | Aligned series |
| Total unique | "Total unique customers" | No | 1 scalar |
| Monthly unique | "Monthly unique customers" | Yes | Time series |
| Simple metric | "Total revenue" | No | 1 scalar |
| Grouped metric | "Revenue by category" | Yes | Per-group series |
| Time-series | "Monthly revenue" | Yes | Monthly series |

## Need Help?

If you're unsure about query behavior:

1. Start with simple queries first
2. Add complexity incrementally
3. Check the plan JSON to see what HaikuGraph generated
4. Look at the SQL to understand the query execution
5. Review test examples in `tests/test_varchar_timestamps_and_scalar_comparisons.py`

## Related Documentation

- Technical details: `REFINEMENTS_SUMMARY.md`
- Previous fixes: `FIXES_SUMMARY.md`
- Architecture: `ARCHITECTURE.md`
