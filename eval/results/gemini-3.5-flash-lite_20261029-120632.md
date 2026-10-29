# Eval: gemini-3.5-flash-lite
prompt v9 · index 66d1995fb6b9 · judge gemini-3.1-flash-lite · 2026-10-29 11:56:06

**Accuracy over 1 run(s): mean 94% (min 94%, max 94%)**

| metric (run 1) | value |
|---|---|
| accuracy | 0.938 |
| mean_score | 0.938 |
| routing_accuracy | 0.929 |
| answers_with_unsupported_numbers | 0 |
| errors | 1 |
| judge_errors_excluded | 0 |
| judge_cost_usd | 0.0 |
| latency_p50_s | 113.4 |
| latency_p95_s | 163.6 |
| mean_cost_usd | 0.0 |
| total_cost_usd | 0.0 |
| mean_prompt_tokens | 14037 |
| mean_cached_tokens | 2024 |
| mean_completion_tokens | 364 |
| mean_tool_calls | 2.5 |
| by tier | 1: 5/6, 2: 8/8, 3: 2/2 |
| by evaluation | exact_match_entity: 3/3, fuzzy_numeric: 6/7, llm_judge: 4/4, refusal: 2/2 |

| id | tier | scorer | pass (per run) | routing | tools | unsupported | latency s | $ | detail (run 1) |
|---|---|---|---|---|---|---|---|---|---|
| h_001 | 1 | fuzzy_numeric | ✅ | ok sql | query_database,calculate | 0 | 58.3 | 0.0000 | gold=88136000000 |
| h_002 | 1 | fuzzy_numeric | ✅ | ok pdf,sql | query_database,search_filings,read_filing_pages | 0 | 113.4 | 0.0000 | gold=10.81 |
| h_003 | 2 | fuzzy_numeric | ✅ | ok pdf,sql | query_database,calculate,search_filings | 0 | 126.0 | 0.0000 | gold=59.7 |
| h_004 | 2 | exact_match_entity | ✅ | ok sql | query_database,calculate | 0 | 73.8 | 0.0000 | entity Alphabet vs Alphabet (ok); number ok |
| h_005 | 2 | exact_match_entity | ✅ | ok sql | query_database,calculate | 0 | 72.6 | 0.0000 | entity Alphabet vs Alphabet (ok); number ok |
| h_006 | 1 | fuzzy_numeric | ✅ | ok pdf,sql | query_database,calculate,search_filings | 0 | 104.0 | 0.0000 | gold=151790000000 |
| h_007 | 1 | refusal | ✅ | ok  |  | 0 | 22.3 | 0.0000 | declined correctly |
| h_008 | 1 | refusal | ✅ | MISS  |  | 0 | 25.8 | 0.0000 | declined correctly |
| h_009 | 2 | llm_judge | ✅ | ok pdf,sql | query_database,search_filings,calculate | 0 | 145.2 | 0.0000 | The candidate answer correctly identifies the growth rate (21.5%) and the revenue figures ($87.5B to $106.3B). It also c |
| h_010 | 3 | llm_judge | ✅ | ok pdf,sql | query_database,calculate,search_filings | 0 | 113.8 | 0.0000 | The candidate answer provides all the key facts and figures present in the reference answer, including the percentages a |
| h_011 | 2 | fuzzy_numeric | ✅ | ok pdf | search_filings | 0 | 85.4 | 0.0000 | gold=31510000000 |
| h_012 | 3 | llm_judge | ✅ | ok pdf,sql | search_filings,query_database,calculate | 0 | 148.3 | 0.0000 | The candidate answer accurately reports both the original FY2024 Intelligent Cloud revenue and the restated figure follo |
| h_013 | 1 | fuzzy_numeric | ❌ | MISS  |  | 0 | 180.0 | 0.0000 | Request exceeded 180s. |
| h_014 | 2 | llm_judge | ✅ | ok pdf,sql | search_filings,query_database,calculate | 0 | 163.6 | 0.0000 | The candidate answer correctly identifies the Services gross margin (75.4%) and the Products gross margin (36.8%) for FY |
| h_015 | 2 | exact_match_entity | ✅ | ok sql | query_database,calculate | 0 | 103.2 | 0.0000 | entity Alphabet vs Alphabet (ok); number ok |
| h_016 | 2 | fuzzy_numeric | ✅ | ok pdf | search_filings,read_filing_pages | 0 | 130.2 | 0.0000 | gold=394328000000 |
