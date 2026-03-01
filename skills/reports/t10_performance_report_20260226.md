# T10 Performance Report - Skills-Integrated Runtime

Generated: 2026-02-26T19:04:08Z
- Previous benchmark: `reports/provider_model_benchmark_20260226_134612.json`
- Current benchmark: `reports/provider_model_benchmark_20260226_140408.json`
- Current HTML: `reports/provider_model_benchmark_20260226_140408.html`

## Mode Performance (Current)

| Profile | Factual % | Success % | Avg latency (ms) | P95 (ms) | Avg LLM calls |
|---|---:|---:|---:|---:|---:|
| deterministic | 83.33 | 100.0 | 1323.67 | 1962.48 | 0 |
| auto | 66.67 | 100.0 | 11091.72 | 18387.15 | 4 |
| local:qwen2.5:7b-instruct | 83.33 | 100.0 | 18718.62 | 31349.59 | 4 |
| openai:gpt-5.3 | 66.67 | 100.0 | 12726.65 | 21164.84 | 8 |
| anthropic:claude-opus-4-6 | 83.33 | 100.0 | 3099.36 | 4385.28 | 15 |

## Delta vs Previous Run (same suite)

| Profile | Factual delta | Latency delta (ms) |
|---|---:|---:|
| deterministic | +0.00 | +13.74 |
| auto | -16.66 | -470.41 |
| local:qwen2.5:7b-instruct | +0.00 | +48.97 |
| openai:gpt-5.3 | +0.00 | -1159.97 |
| anthropic:claude-opus-4-6 | +0.00 | +161.51 |

## Failure Analysis (Current)

- Common failure across all modes: `tx_total_amount_dec_2025`
  - expected SQL uses `event_ts`
  - produced SQL uses `created_ts`
- Additional drift in `auto` and `openai:gpt-5.3`: `mt103_month_platform`
  - expected SQL uses `mt103_created_ts` with conditional distinct count
  - produced SQL uses `created_ts` + plain distinct count over `has_mt103=true`

## T10 Closure Check

- T0-T10 tracker status: all marked completed in `skills/T0_T10_TRACKER.md`.
- Skills bootstrap reproducibility: `./scripts/setup_skills_base.sh` completed successfully.
- Security hygiene: `./scripts/security_hygiene_check.sh` -> PASS.
- Targeted test suite: 22/22 passed.
- Live benchmark report generated with deterministic, auto, local, OpenAI, and Anthropic profiles.

## Skills Integration Verification

- Trace fields present per agent step:
  - `selected_skills`
  - `skill_policy_reason`
  - `skill_contract_file`
  - `skill_layer_file`
- Explain view shows skills and contract references per step.
- Runtime payload includes `skills_runtime` summary.

## Bottom Line

- Reliability guardrails are stable: 100% success across profiles.
- Factual exact-match remains bounded by two canonical grounding drifts (`event_ts`/`created_ts`, mt103 time-column contract in LLM paths).
- Performance profile remains unchanged in ordering:
  - fastest LLM path: Anthropic
  - lowest-latency overall: deterministic
  - highest-latency path: local 7B
