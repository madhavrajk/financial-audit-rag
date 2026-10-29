# Eval: gemini-3.5-flash-lite
prompt v8 · index 66d1995fb6b9 · judge gemini-3.1-flash-lite · 2026-10-28 21:59:01

Rows for questions not in eval/questions.json were removed on 2026-10-29; the summaries are recomputed from the rows below.

**Accuracy over 1 run(s): mean 100% (min 100%, max 100%)**

| metric (run 1) | value |
|---|---|
| accuracy | 1.0 |
| mean_score | 1.0 |
| routing_accuracy | 0.933 |
| answers_with_unsupported_numbers | 0 |
| errors | 0 |
| judge_errors_excluded | 0 |
| judge_cost_usd | 0.0 |
| latency_p50_s | 14.1 |
| latency_p95_s | 83.2 |
| mean_cost_usd | 0.0 |
| total_cost_usd | 0.0 |
| mean_prompt_tokens | 10504 |
| mean_cached_tokens | 1260 |
| mean_completion_tokens | 320 |
| mean_tool_calls | 1.9 |
| by tier | 1: 6/6, 2: 8/8, 3: 2/2 |
| by evaluation | exact_match_entity: 3/3, fuzzy_numeric: 7/7, llm_judge: 4/4, refusal: 2/2 |

| id | tier | scorer | pass (per run) | routing | tools | unsupported | latency s | $ | detail (run 1) |
|---|---|---|---|---|---|---|---|---|---|
| h_001 | 1 | fuzzy_numeric | ✅ | ok sql | query_database | 0 | 6.0 | 0.0000 | gold=88136000000 |
| h_002 | 1 | fuzzy_numeric | ✅ | ok sql | query_database | 0 | 66.6 | 0.0000 | gold=10.81 |
| h_003 | 2 | fuzzy_numeric | ✅ | ok sql | query_database,calculate | 0 | 19.9 | 0.0000 | gold=59.7 |
| h_004 | 2 | exact_match_entity | ✅ | ok sql | query_database,calculate | 0 | 14.1 | 0.0000 | entity Alphabet vs Alphabet (ok); number ok |
| h_005 | 2 | exact_match_entity | ✅ | ok pdf,sql | query_database,calculate,search_filings | 0 | 83.2 | 0.0000 | entity Alphabet vs Alphabet (ok); number ok |
| h_006 | 1 | fuzzy_numeric | ✅ | ok pdf | search_filings,calculate | 0 | 12.0 | 0.0000 | gold=151790000000 |
| h_007 | 1 | refusal | ✅ | ok  |  | 0 | 5.2 | 0.0000 | declined correctly |
| h_008 | 1 | refusal | ✅ | MISS  |  | 0 | 1.7 | 0.0000 | declined correctly |
| h_009 | 2 | llm_judge | ✅ | ok pdf,sql | query_database,search_filings,calculate | 0 | 21.3 | 0.0000 | The candidate answer correctly identifies the revenue growth percentage, the dollar amounts, and the specific drivers me |
| h_010 | 3 | llm_judge | ✅ | ok sql | query_database,calculate | 0 | 79.8 | 0.0000 | The candidate answer correctly identifies the percentages and the change in percentage points, and provides the underlyi |
| h_011 | 2 | fuzzy_numeric | ✅ | ok pdf | search_filings | 0 | 3.1 | 0.0000 | gold=31510000000 |
| h_012 | 3 | llm_judge | ✅ | ok pdf,sql | query_database,search_filings,calculate | 0 | 9.3 | 0.0000 | The candidate answer provides all the information requested in the reference answer, including the original figure, the  |
| h_013 | 1 | fuzzy_numeric | ✅ | ok sql | query_database,calculate | 0 | 10.3 | 0.0000 | gold=0.89 |
| h_014 | 2 | llm_judge | ✅ | ok pdf | search_filings,calculate | 0 | 14.6 | 0.0000 | The candidate answer correctly identifies the Services gross margin (75.4%) and the Products gross margin (36.8%) for FY |
| h_015 | 2 | exact_match_entity | ✅ | ok sql | query_database,calculate | 0 | 129.2 | 0.0000 | entity Alphabet vs Alphabet (ok); number ok |
| h_016 | 2 | fuzzy_numeric | ✅ | ok pdf | search_filings | 0 | 6.2 | 0.0000 | gold=394328000000 |
