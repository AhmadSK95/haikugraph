# Industry Onboarding Playbook

Purpose: onboard a new industry dataset into dataDa with repeatable profiling, glossary setup, and QA checks.

## 1) Intake Checklist

1. Confirm data owner, SLA, and update cadence.
2. Confirm source-of-truth tables for transactions, quotes, customers, bookings (or nearest equivalents).
3. Confirm business metric definitions and exclusion rules.
4. Confirm compliance constraints (PII, retention, access controls).

## 2) Schema Profiling

Generate a profile artifact before any model tuning:

```bash
python scripts/generate_onboarding_profile.py \
  --db-path data/new_industry.duckdb \
  --out-json reports/onboarding_profile_new_industry.json \
  --out-md reports/onboarding_profile_new_industry.md
```

Use template: `docs/templates/schema_profile_template.yaml`

## 3) Semantic Mapping

1. Map each source table to a target business domain.
2. Map metric columns to canonical metric IDs.
3. Confirm time columns for each metric.
4. Create or update glossary terms and synonyms.

## 4) Contract Readiness Gate

Before enabling non-deterministic modes:

1. Contract validation pass for top 20 business questions.
2. Explainability payload check:
   - `business_view` is readable for non-technical users.
   - `technical_view` contains SQL, contract, trace, and data quality diagnostics.
3. Policy/safety gate pass for refusal and anti-coercion prompts.

## 5) Round QA Protocol (Fresh + Fast)

1. Generate fresh prompt variants for the round.
2. Run quick smoke (`deterministic` + `auto`) for latency screening.
3. Run full multimode black-box only after smoke passes.
4. Publish artifacts in `reports/` and update both trackers.

## 6) Exit Criteria

1. No critical factual regressions in seeded capability suites.
2. No policy/safety regressions.
3. Latency and queue controls are within configured SLO targets.
4. Tracker tasks for onboarding slice include evidence links.
