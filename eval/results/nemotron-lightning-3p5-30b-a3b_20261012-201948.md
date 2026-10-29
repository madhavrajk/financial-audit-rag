# Eval: nemotron-lightning-3p5-30b-a3b
prompt v6 · index 1669a312787e · judge kimi-k3 · 2026-10-12 20:18:26

Rows for questions not in eval/questions.json were removed on 2026-10-29; the summaries are recomputed from the rows below.

**Accuracy over 1 run(s): mean 88% (min 88%, max 88%)**

| metric (run 1) | value |
|---|---|
| accuracy | 0.875 |
| mean_score | 0.875 |
| routing_accuracy | 0.933 |
| answers_with_unsupported_numbers | 1 |
| errors | 0 |
| judge_errors_excluded | 0 |
| judge_cost_usd | 0 |
| latency_p50_s | 4.0 |
| latency_p95_s | 8.0 |
| mean_cost_usd | 0.00079 |
| total_cost_usd | 0.0127 |
| mean_prompt_tokens | 17490 |
| mean_cached_tokens | 10976 |
| mean_completion_tokens | 1778 |
| mean_tool_calls | 2.6 |
| by tier | 1: 5/6, 2: 7/8, 3: 2/2 |
| by evaluation | exact_match_entity: 3/3, fuzzy_numeric: 7/7, llm_judge: 3/4, refusal: 1/2 |

| id | tier | scorer | pass (per run) | routing | tools | unsupported | latency s | $ | detail (run 1) |
|---|---|---|---|---|---|---|---|---|---|
| h_001 | 1 | fuzzy_numeric | ✅ | ok sql | query_database | 0 | 3.0 | 0.0003 | gold=88136000000 |
| h_002 | 1 | fuzzy_numeric | ✅ | ok sql | query_database | 0 | 2.8 | 0.0003 | gold=10.81 |
| h_003 | 2 | fuzzy_numeric | ✅ | ok sql | query_database,calculate | 0 | 2.7 | 0.0004 | gold=59.7 |
| h_004 | 2 | exact_match_entity | ✅ | ok sql | query_database,calculate | 0 | 2.2 | 0.0004 | entity Alphabet vs Alphabet (ok); number ok |
| h_005 | 2 | exact_match_entity | ✅ | ok sql | query_database,calculate | 0 | 8.0 | 0.0010 | entity Alphabet vs Alphabet (ok); number ok |
| h_006 | 1 | fuzzy_numeric | ✅ | ok pdf,sql | search_filings,query_database | 0 | 2.6 | 0.0007 | gold=151790000000 |
| h_007 | 1 | refusal | ❌ | ok  |  | 0 | 2.0 | 0.0003 | did not decline |
| h_008 | 1 | refusal | ✅ | MISS  |  | 0 | 6.4 | 0.0006 | declined correctly |
| h_009 | 2 | llm_judge | ✅ | ok pdf,sql | query_database,calculate,search_filings | 0 | 4.0 | 0.0006 | The candidate matches all key facts: 21.5% growth, $87.5B to $106.3B, and Azure and other cloud services growing 34% as  |
| h_010 | 3 | llm_judge | ✅ | ok sql | query_database,calculate | 4 | 6.1 | 0.0010 | The candidate matches the reference on all key facts: Google Cloud's share rose from 12.4% in FY2024 to 14.6% in FY2025, |
| h_011 | 2 | fuzzy_numeric | ✅ | ok pdf | search_filings,calculate | 0 | 4.5 | 0.0008 | gold=31510000000 |
| h_012 | 3 | llm_judge | ✅ | ok pdf,sql | search_filings,query_database,calculate | 0 | 13.9 | 0.0028 | The candidate correctly states the FY2024 10-K Intelligent Cloud revenue of $105.4 billion ($105,362M), the restated pos |
| h_013 | 1 | fuzzy_numeric | ✅ | ok sql | query_database,calculate | 0 | 3.7 | 0.0004 | gold=0.89 |
| h_014 | 2 | llm_judge | ❌ | ok pdf,sql | search_filings,query_database | 0 | 6.9 | 0.0012 | The candidate stated it could not produce a reliable answer, while the reference provides specific figures (Services 75. |
| h_015 | 2 | exact_match_entity | ✅ | ok sql | query_database,calculate | 0 | 2.9 | 0.0005 | entity Alphabet vs Alphabet (ok); number ok |
| h_016 | 2 | fuzzy_numeric | ✅ | ok pdf,sql | search_filings,query_database | 0 | 5.8 | 0.0011 | gold=394328000000 |
