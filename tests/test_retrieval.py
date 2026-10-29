"""Extraction quality and golden retrieval tests (BM25-only, offline)."""

from __future__ import annotations

import pytest

from app.ingest.extract import extract_pages
from tests.conftest import PDFS


def test_tables_keep_rows_intact():
    pages = extract_pages(PDFS / "GOOGL_FY2025_10-K.pdf")
    assert any("YouTube ads | 36,147 | 40,367" in p.text for p in pages)


def test_section_tags(index):
    risk = index.section_pages("AAPL FY2025 10-K", "1A")
    assert risk and all("1A" in p.items for p in risk)
    # Apple FY2024's "Item 1. Business" heading sits on the table-of-contents page.
    p4 = index.get_pages("aapl fy2024 10-k", 4, 4)[0]  # document names are case-insensitive
    assert p4.items == ["1"] and p4.citation == "AAPL FY2024 10-K p.4"


@pytest.mark.parametrize(
    "query,ticker,fy,section,expected_pages",
    [
        ("Greater China net sales decreased iPhone", "AAPL", 2025, None, {24}),
        ("YouTube ads revenues increased", "GOOGL", 2025, None, {36}),
        (
            "supply chain single source components outsourcing partners",
            "AAPL",
            2025,
            "1A",
            set(range(8, 19)),
        ),
    ],
)
async def test_golden_retrieval_top3(index, query, ticker, fy, section, expected_pages):
    res = await index.search(query, ticker=ticker, fiscal_year=fy, section=section, k=3)
    got = {h.page.page for h in res.hits}
    assert got & expected_pages, f"expected one of {expected_pages}, got {got}"
    assert res.degraded  # no embedder in tests → BM25-only, and it says so


async def test_filters_with_no_match(index):
    res = await index.search("revenue", ticker="AAPL", fiscal_year=2023)
    assert not res.hits and "AAPL FY2024 10-K, AAPL FY2025 10-K" in res.note
