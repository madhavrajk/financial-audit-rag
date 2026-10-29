# Data

- `financials.db`: SQLite database of annual financials for Apple, Microsoft and Alphabet: income statements (FY2023–25), balance sheets (FY2024–25), and revenue by segment and by region (FY2023–25). It is built from the inline XBRL in the companies' FY2025 10-Ks, as filed on EDGAR (`aapl-20250927.htm`, `goog-20251231.htm`, `msft-20250630.htm`), with `uv run python -m app.ingest database FILE ...`.
- `pdfs/`: the FY2024 and FY2025 10-K filings for the same companies (public SEC filings).
- `filings/`: companies added with `uv run python -m app.ingest add` (10-Ks as filed on EDGAR, named `TICKER_FY2025_10-K.htm`). Their annual financials are written to `financials.db` from the filings' inline XBRL.
- `index/`: the prebuilt page-level search index (page text, embeddings, manifest). Rebuild it with `uv run python -m app.ingest` (add `--no-embed` to skip embedding API calls; pages keep the vectors they have, and new pages are searched by keyword).
- `financebench/` (not in git): FinanceBench's questions, 84 PDFs and their index, downloaded by `uv run python -m eval.financebench download`.
