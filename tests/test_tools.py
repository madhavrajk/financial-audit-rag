"""SQL guard, calculator, tool-argument validation, and ticker aliasing."""

from __future__ import annotations

import json

import pytest

from app.core.errors import ToolError
from app.tools import ToolBox
from app.tools.calc import calculate_many, evaluate
from tests.conftest import tool


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE companies",
        "DELETE FROM income_statements",
        "INSERT INTO companies VALUES ('X','X','1','1','T','Dec')",
        "UPDATE income_statements SET revenue = 0",
        "ATTACH DATABASE '/tmp/x.db' AS x",
        "PRAGMA query_only = OFF",
        "CREATE TEMP TABLE t(x)",
        "SELECT 1; DROP TABLE companies",
    ],
)
def test_sql_writes_and_escapes_are_blocked(store, sql):
    with pytest.raises(ToolError):
        store.query(sql)


def test_sql_runaway_query_is_aborted(store):
    with pytest.raises(ToolError, match="aborted"):
        store.query(
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c"
        )


def test_sql_select_and_row_cap(store):
    r = store.query(
        "SELECT revenue FROM income_statements WHERE company_ticker='AAPL' AND fiscal_year=2025"
    )
    assert r.rows == [[416161000000]]
    big = store.query(
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c "
        "WHERE x < 1000) SELECT x FROM c"
    )
    assert big.truncated and len(big.rows) == 200


def test_schema_lists_enumerated_values(store):
    schema = store.schema_description()
    assert "Greater China" in schema and "U.S." in schema
    assert "AAPL: FY2023-FY2025 (except balance_sheets FY2024-FY2025)" in schema


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("(416161 - 391035) / 391035 * 100", 6.4255),
        ("144,546 / 281,724 * 100", 51.3077),
        ("-127 - 211", -338),
        ("round(2.0053, 2)", 2.01),
        ("max(1, 2) + abs(-3)", 5),
        ("max(1,2)", 2),
    ],
)
def test_calc(expr, expected):
    assert evaluate(expr) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "open('x')",
        "x + 1",
        "1/0",
        "2 ** 1000",
        "",
        "a.b",
    ],
)
def test_calc_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ToolError):
        evaluate(expr)


def test_calculate_many_reports_errors_per_label():
    out = calculate_many({"ok": "1+1", "bad": "1/0"})
    assert out["ok"]["value"] == 2 and "division by zero" in out["bad"]["error"]


async def test_toolbox_validation_and_aliases(store, index):
    tb = ToolBox(store, index)
    unknown = await tb.execute(tool("delete_everything"))
    assert unknown.error and "Unknown tool" in unknown.content
    bad_company = await tb.execute(tool("search_filings", query="x", ticker="Amazon"))
    assert bad_company.error and "AAPL, GOOGL, MSFT" in bad_company.content
    for name in ("Alphabet", "Alphabet Inc.", "google"):
        alias = await tb.execute(
            tool("search_filings", query="YouTube ads revenues", ticker=name, fiscal_year=2025)
        )
        assert not alias.error and all(s.label.startswith("GOOGL FY2025") for s in alias.sources)
    bad_doc = await tb.execute(
        tool("read_filing_pages", document="AAPL FY2022 10-K", start_page=1, end_page=2)
    )
    assert bad_doc.error and "AAPL FY2025 10-K" in bad_doc.content  # lists what exists


async def test_read_pages_is_capped_and_accepts_a_citation(store, index):
    tb = ToolBox(store, index)
    out = await tb.execute(
        tool("read_filing_pages", document="AAPL FY2025 10-K p.10", start_page=10, end_page=40)
    )
    assert [s.label for s in out.sources] == [f"AAPL FY2025 10-K p.{n}" for n in (10, 11, 12)]


async def test_sql_source_labels_name_tables(store, index):
    tb = ToolBox(store, index)
    out = await tb.execute(
        tool(
            "query_database",
            sql=(
                "SELECT i.revenue, b.total_assets FROM income_statements i JOIN balance_sheets b "
                "ON i.company_ticker=b.company_ticker AND i.fiscal_year=b.fiscal_year LIMIT 1"
            ),
        )
    )
    assert [s.label for s in out.sources] == ["SQL: income_statements", "SQL: balance_sheets"]


def test_tool_specs_are_valid_openai_schema_with_the_data_as_enums(store, index):
    specs = ToolBox(store, index).specs()
    assert [s["function"]["name"] for s in specs] == [
        "query_database",
        "search_filings",
        "read_filing_pages",
        "calculate",
    ]
    for s in specs:
        json.dumps(s)
        assert s["function"]["parameters"]["type"] == "object"
    search = specs[1]["function"]["parameters"]["properties"]
    assert search["ticker"]["anyOf"][0]["enum"] == ["AAPL", "GOOGL", "MSFT"]
    assert search["fiscal_year"]["anyOf"][0]["enum"] == [2024, 2025]
    assert "form" not in search  # only 10-Ks: nothing to filter on
    read = specs[2]["function"]["parameters"]["properties"]
    assert "AAPL FY2025 10-K" in read["document"]["enum"]


def test_without_a_database_only_filing_tools_are_offered(index):
    names = [s["function"]["name"] for s in ToolBox(None, index).specs()]
    assert names == ["search_filings", "read_filing_pages", "calculate"]
