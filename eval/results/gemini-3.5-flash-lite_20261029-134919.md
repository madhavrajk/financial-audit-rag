# Eval: gemini-3.5-flash-lite
prompt v9 · index 66d1995fb6b9 · judge gemini-3.1-flash-lite · 2026-10-29 13:47:20

**Accuracy over 1 run(s): mean 100% (min 100%, max 100%)**

| metric (run 1) | value |
|---|---|
| accuracy | 1.0 |
| mean_score | 1.0 |
| routing_accuracy | 1.0 |
| answers_with_unsupported_numbers | 0 |
| errors | 0 |
| judge_errors_excluded | 0 |
| judge_cost_usd | 0.0 |
| latency_p50_s | 118.9 |
| latency_p95_s | 118.9 |
| mean_cost_usd | 0.0 |
| total_cost_usd | 0.0 |
| mean_prompt_tokens | 15243 |
| mean_cached_tokens | 0 |
| mean_completion_tokens | 317 |
| mean_tool_calls | 3 |
| by tier | 1: 1/1 |
| by evaluation | fuzzy_numeric: 1/1 |

| id | tier | scorer | pass (per run) | routing | tools | unsupported | latency s | $ | detail (run 1) |
|---|---|---|---|---|---|---|---|---|---|
| h_013 | 1 | fuzzy_numeric | ✅ | ok pdf,sql | query_database,calculate,search_filings | 0 | 118.9 | 0.0000 | gold=0.89 |
