# Quick Reference - DISTINCT & Time Bucketing

## For Planner (LLM)

### DISTINCT Counts
```json
// ✅ CORRECT
{"agg": "count", "col": "customer_id", "distinct": true}

// ✅ ALSO CORRECT
{"agg": "count_distinct", "col": "customer_id"}

// ❌ WRONG (will be rejected)
{"agg": "count", "col": "DISTINCT customer_id"}
{"agg": "count", "col": "customer id"}  // spaces not allowed
```

### Time Bucketing
```json
// ✅ Monthly aggregation
{
  "group_by": [
    {"type": "time_bucket", "grain": "month", "col": "created_at"}
  ],
  "aggregations": [{"agg": "sum", "col": "amount"}]
}

// ✅ Yearly aggregation
{
  "group_by": [
    {"type": "time_bucket", "grain": "year", "col": "created_at"}
  ],
  "aggregations": [{"agg": "count", "col": "customer_id", "distinct": true}]
}

// ✅ Mixed: time bucket + regular column
{
  "group_by": [
    {"type": "time_bucket", "grain": "month", "col": "date"},
    "status"
  ],
  "aggregations": [{"agg": "sum", "col": "revenue"}]
}
```

### Supported Grains
- `"month"` - Monthly buckets
- `"year"` - Yearly buckets
- `"day"` - Daily buckets
- `"week"` - Weekly buckets
- `"quarter"` - Quarterly buckets

### Comparisons
```json
// ✅ CORRECT - Each subquestion has scoped constraint
{
  "subquestions": [
    {"id": "SQ1_current", ...},
    {"id": "SQ2_comparison", ...}
  ],
  "constraints": [
    {
      "type": "time",
      "expression": "test_1_1.created_at in this_month",
      "applies_to": "SQ1_current"
    },
    {
      "type": "time",
      "expression": "test_1_1.created_at in previous_month",
      "applies_to": "SQ2_comparison"
    }
  ]
}

// ❌ WRONG - Missing applies_to
{
  "subquestions": [
    {"id": "SQ1_current", ...},
    {"id": "SQ2_comparison", ...}
  ],
  "constraints": [
    {
      "type": "time",
      "expression": "test_1_1.created_at in this_month"
      // Missing applies_to!
    }
  ]
}
```

## Generated SQL Examples

### DISTINCT Count
**Input**:
```json
{"agg": "count", "col": "customer_id", "distinct": true}
```

**Output SQL**:
```sql
COUNT(DISTINCT "test_1_1"."customer_id") AS "count_distinct_customer_id"
```

### Monthly Time Bucket
**Input**:
```json
{
  "group_by": [{"type": "time_bucket", "grain": "month", "col": "created_at"}],
  "aggregations": [{"agg": "sum", "col": "amount"}]
}
```

**Output SQL**:
```sql
SELECT 
  date_trunc('month', "test_1_1"."created_at") AS "month",
  SUM(TRY_CAST("test_1_1"."amount" AS DOUBLE)) AS "sum_amount"
FROM "test_1_1"
GROUP BY date_trunc('month', "test_1_1"."created_at")
ORDER BY "month"
```

### Monthly Unique Customers (Combined)
**Input**:
```json
{
  "group_by": [{"type": "time_bucket", "grain": "month", "col": "created_at"}],
  "aggregations": [{"agg": "count", "col": "customer_id", "distinct": true}]
}
```

**Output SQL**:
```sql
SELECT 
  date_trunc('month', "test_1_1"."created_at") AS "month",
  COUNT(DISTINCT "test_1_1"."customer_id") AS "count_distinct_customer_id"
FROM "test_1_1"
GROUP BY date_trunc('month', "test_1_1"."created_at")
ORDER BY "month"
```

## Query Examples

| Natural Language | Intent | Plan Structure |
|-----------------|--------|----------------|
| "How many unique customers?" | metric | DISTINCT count, no group_by |
| "Monthly revenue" | grouped_metric | time_bucket + sum |
| "Unique customers by month" | grouped_metric | time_bucket + DISTINCT count |
| "Revenue this month vs last month" | comparison | 2 subquestions, scoped time constraints |
| "Yearly total sales" | grouped_metric | time_bucket year + sum |
| "Monthly unique customers last year" | grouped_metric | time_bucket + DISTINCT + time constraint |

## Validation Rules

### Column Names
- ✅ Simple identifiers: `customer_id`, `amount`, `created_at`
- ❌ SQL keywords: `DISTINCT`, `SELECT`, `FROM`, `WHERE`, `JOIN`, `UNION`
- ❌ Spaces: `customer id`, `amount total`
- ❌ Expressions: `amount * 2`, `CAST(amount AS INT)`

### Aggregations
- ✅ Required fields: `agg`, `col`
- ✅ Optional field: `distinct` (for count)
- ✅ Valid agg values: `sum`, `avg`, `count`, `min`, `max`, `count_distinct`

### Group By
- ✅ List of strings: `["status", "region"]`
- ✅ List of dicts: `[{"type": "time_bucket", "grain": "month", "col": "date"}]`
- ✅ Mixed: `["status", {"type": "time_bucket", "grain": "month", "col": "date"}]`

### Time Constraints (Comparisons)
- ✅ Must have `applies_to` for comparison queries
- ✅ Each comparison subquestion needs its own constraint
- ✅ Subquestion IDs must match `applies_to` values

## Testing

```bash
# Run DISTINCT and time bucket tests
pytest tests/test_distinct_and_time_bucket.py -v

# Run all tests
pytest tests/ -v

# Quick test summary
pytest tests/ -q
```

Expected: **179 passed** ✅

## Common Patterns

### Pattern 1: Simple DISTINCT Count
```json
{
  "original_question": "How many unique customers?",
  "subquestions": [{
    "id": "SQ1",
    "tables": ["orders"],
    "aggregations": [{"agg": "count", "col": "customer_id", "distinct": true}]
  }]
}
```

### Pattern 2: Monthly Aggregation
```json
{
  "original_question": "Monthly revenue",
  "subquestions": [{
    "id": "SQ1",
    "tables": ["orders"],
    "group_by": [{"type": "time_bucket", "grain": "month", "col": "order_date"}],
    "aggregations": [{"agg": "sum", "col": "revenue"}]
  }]
}
```

### Pattern 3: Monthly + DISTINCT
```json
{
  "original_question": "Unique customers per month",
  "subquestions": [{
    "id": "SQ1",
    "tables": ["orders"],
    "group_by": [{"type": "time_bucket", "grain": "month", "col": "order_date"}],
    "aggregations": [{"agg": "count", "col": "customer_id", "distinct": true}]
  }]
}
```

### Pattern 4: Month-to-Month Comparison
```json
{
  "original_question": "Revenue this month vs last month",
  "subquestions": [
    {
      "id": "SQ1_current",
      "tables": ["orders"],
      "aggregations": [{"agg": "sum", "col": "revenue"}]
    },
    {
      "id": "SQ2_comparison",
      "tables": ["orders"],
      "aggregations": [{"agg": "sum", "col": "revenue"}]
    }
  ],
  "constraints": [
    {"type": "time", "expression": "orders.order_date in this_month", "applies_to": "SQ1_current"},
    {"type": "time", "expression": "orders.order_date in previous_month", "applies_to": "SQ2_comparison"}
  ]
}
```

### Pattern 5: Year-over-Year Comparison
```json
{
  "original_question": "Revenue this year vs last year",
  "subquestions": [
    {
      "id": "SQ1_current",
      "tables": ["orders"],
      "aggregations": [{"agg": "sum", "col": "revenue"}]
    },
    {
      "id": "SQ2_comparison",
      "tables": ["orders"],
      "aggregations": [{"agg": "sum", "col": "revenue"}]
    }
  ],
  "constraints": [
    {"type": "time", "expression": "orders.order_date in this_year", "applies_to": "SQ1_current"},
    {"type": "time", "expression": "orders.order_date in previous_year", "applies_to": "SQ2_comparison"}
  ]
}
```

## UI Visualization Hints

The API automatically adds visualization hints:

| Data Shape | display_hint | Renders As |
|------------|-------------|------------|
| 1 row, 1 col | `number` | Large number card |
| Multi-row, 2 cols | `bar_chart` | Bar chart |
| Time bucket + value | `line_chart` | Line chart (time series) |
| 3+ cols or 20+ rows | `table` | Data table |

Time buckets are detected by:
- Column names: "month", "year", "day", "week", "quarter", "date", "time"
- Values matching pattern: `YYYY-MM`, `YYYY-MM-DD`

## Troubleshooting

### Error: "Column name contains forbidden SQL keyword 'distinct'"
**Cause**: Column name has "DISTINCT" in it  
**Fix**: Use `"distinct": true` field instead

### Error: "Column name contains spaces"
**Cause**: Column name like "customer id"  
**Fix**: Use simple identifier: "customer_id"

### Error: "unscoped time constraint"
**Cause**: Comparison query missing `applies_to`  
**Fix**: Add `applies_to` to ALL time constraints in comparison queries

### SQL Error: "Binder Error: column \"DISTINCT customer_id\" does not exist"
**Cause**: Old-style DISTINCT hack in plan  
**Fix**: Regenerate plan with new schema (will be rejected during validation)

## Migration Guide

### Existing Plans
- ✅ All valid plans continue to work
- ✅ New fields are optional
- ✅ Backward compatible

### If You Have Plans with SQL Keywords in Columns
```python
# OLD (now rejected)
{"agg": "count", "col": "DISTINCT customer_id"}

# NEW
{"agg": "count", "col": "customer_id", "distinct": True}
```

### If You Need Monthly Aggregations
```python
# OLD (treated as single metric)
{
  "group_by": ["month"]  # Assumes column named "month"
}

# NEW (time bucketing)
{
  "group_by": [
    {"type": "time_bucket", "grain": "month", "col": "created_at"}
  ]
}
```
