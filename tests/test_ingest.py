"""Adding a company from its filing: HTML pages, inline-XBRL financials, the index, EDGAR."""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pytest

from app.adapters.local_index import LocalPageIndex
from app.adapters.sqlite_store import SQLiteFinancialStore
from app.core.agent import Agent
from app.core.config import Settings
from app.ingest import edgar
from app.ingest.add import add
from app.ingest.build_index import Document, build_index
from app.ingest.extract import extract_pages
from app.ingest.financials import build_database, financials_from_filing, upsert
from app.ingest.html_filing import ParsedFiling, parse_filing
from app.tools import ToolBox
from tests.conftest import DB, FakeLLM


def ctx(cid: str, period: str, dims: str = "") -> str:
    segment = f"<xbrli:segment>{dims}</xbrli:segment>" if dims else ""
    return (
        f'<xbrli:context id="{cid}"><xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier>'
        f"{segment}</xbrli:entity><xbrli:period>{period}</xbrli:period></xbrli:context>"
    )


def fy(year: int) -> str:
    start, end = f"{year}-01-01", f"{year}-12-31"
    return f"<xbrli:startDate>{start}</xbrli:startDate><xbrli:endDate>{end}</xbrli:endDate>"


def seg(member: str) -> str:
    axis = "us-gaap:StatementBusinessSegmentsAxis"
    return f'<xbrldi:explicitMember dimension="{axis}">exmp:{member}</xbrldi:explicitMember>'


def num(concept: str, context: str, value: str, sign: str = "") -> str:
    s = f' sign="{sign}"' if sign else ""
    return (
        f'<ix:nonFraction name="us-gaap:{concept}" contextRef="{context}" unitRef="usd" '
        f'scale="6" decimals="-6"{s}>{value}</ix:nonFraction>'
    )


def dei(name: str, value: str) -> str:
    return f'<ix:nonNumeric contextRef="c1" name="dei:{name}">{value}</ix:nonNumeric>'


# A 10-K in the shape EDGAR serves: hidden XBRL header, page breaks, tagged table cells.
FILING = f"""<html><body>
<div style="display:none"><ix:header><ix:hidden>
{dei("DocumentType", "10-K")}{dei("DocumentFiscalYearFocus", "2025")}
{dei("EntityRegistrantName", "Example Corp")}{dei("TradingSymbol", "EXMP")}
{dei("EntityCentralIndexKey", "0000000001")}
</ix:hidden><ix:resources>
{ctx("c1", fy(2025))}{ctx("c2", fy(2024))}
{ctx("c3", "<xbrli:instant>2025-12-31</xbrli:instant>")}
{ctx("c4", fy(2025), seg("CloudMember"))}{ctx("c5", fy(2025), seg("DevicesMember"))}
{ctx("c6", "<xbrli:instant>2026-01-20</xbrli:instant>")}
</ix:resources></ix:header></div>
<div>Shares outstanding at January 20, 2026: <ix:nonFraction
name="dei:EntityCommonStockSharesOutstanding" contextRef="c6" unitRef="shares"
decimals="0">1,000</ix:nonFraction></div>
<div><span>Item 1.</span><span style="padding-left:9pt">Business</span></div>
<div>We make example products.</div>
<hr style="page-break-after:always"/>
<div><span>Item 7.</span><span style="padding-left:9pt">Management's Discussion</span></div>
<table>
<tr><td>Revenue</td><td>$</td><td>{num("Revenues", "c1", "1,000")}</td>
<td>$</td><td>{num("Revenues", "c2", "900")}</td></tr>
<tr><td>Net income (loss)</td><td>{num("NetIncomeLoss", "c1", "100")}</td>
<td>(</td><td>{num("NetIncomeLoss", "c2", "80", sign="-")}</td><td>)</td></tr>
<tr><td>Cloud (1)</td><td>{num("Revenues", "c4", "600")}</td></tr>
<tr><td>Devices</td><td>{num("Revenues", "c5", "400")}</td></tr>
</table>
<hr style="page-break-after:always"/>
<div><span>Item 8.</span><span style="padding-left:9pt">Financial Statements</span></div>
<table><tr><td>Total assets</td><td>{num("Assets", "c3", "5,000")}</td></tr></table>
</body></html>"""


@pytest.fixture
def filing(tmp_path: Path) -> Path:
    path = tmp_path / "downloaded-10k.htm"
    path.write_text(FILING)
    return path


def test_html_pages_keep_rows_and_hide_the_xbrl_header(filing):
    parsed = parse_filing(filing)
    assert len(parsed.pages) == 3
    assert "Revenue | $1,000 | $900" in parsed.pages[1]
    assert "Net income (loss) | 100 | (80)" in parsed.pages[1]
    assert "Item 7. Management's Discussion" in parsed.pages[1]  # spaced like the layout
    assert all("EXMP" not in page for page in parsed.pages)  # hidden header stays hidden
    assert parsed.dei["DocumentFiscalYearFocus"] == ["2025"]


def test_html_pages_get_10k_item_tags(filing):
    pages = extract_pages(filing, ticker="EXMP", fiscal_year=2025, form="10-K")
    # As with PDFs, a page also belongs to the section continuing from the page before.
    assert [p.items for p in pages] == [["1"], ["1", "7"], ["7", "8"]]
    assert pages[1].citation == "EXMP FY2025 10-K p.2" and pages[0].company == "Example Corp"


def test_financials_come_from_the_tagged_numbers(filing):
    f = financials_from_filing(parse_filing(filing))
    assert (f.ticker, f.name, f.fiscal_year, f.fiscal_year_end) == (
        "EXMP",
        "Example Corp",
        2025,
        "December",
    )
    assert f.income[2025]["revenue"] == 1e9 and f.income[2024]["net_income"] == -80e6
    assert f.balance[2025]["total_assets"] == 5e9
    # The balance sheet's date, not the cover page's share count (20 days later).
    assert f.balance_dates[2025] == date(2025, 12, 31)
    assert f.segments[2025] == {"Cloud": 6e8, "Devices": 4e8}  # row labels, footnote dropped


@pytest.mark.parametrize(
    "filed,shown",
    [
        ("MICROSOFT CORPORATION", "Microsoft Corporation"),
        ("AMAZON.COM, INC.", "Amazon.com, Inc."),
        ("AT&T INC.", "AT&T Inc."),
        ("Apple Inc.", "Apple Inc."),
    ],
)
def test_company_names_filed_in_capitals_get_title_case(filed, shown):
    filing = ParsedFiling([], [], {}, {"EntityRegistrantName": [filed]}, "")
    assert filing.company == shown


def test_database_is_built_from_filings_and_replaced_whole(filing, tmp_path):
    db = tmp_path / "financials.db"
    db.write_text("not a database")  # replaced, not appended to
    (line,) = build_database([filing], db)
    assert line == "Example Corp (EXMP) FY2025 10-K: 2 income, 1 balance, 2 segments, 0 regions"
    store = SQLiteFinancialStore(db)
    assert store.companies() == {"EXMP": ("Example Corp", "December")}
    assert store.query("SELECT period_end FROM balance_sheets").rows == [["2025-12-31"]]


def test_newest_filing_wins_and_older_ones_only_fill_gaps(filing, tmp_path):
    db = tmp_path / "f.db"  # upsert creates the tables
    f = financials_from_filing(parse_filing(filing))
    assert upsert(db, f)["income"] == 2
    older = financials_from_filing(parse_filing(filing))
    older.period_end = older.period_end.replace(year=2024)
    older.income[2025]["revenue"] = 1.0  # a stale value must not overwrite the newer filing
    upsert(db, older)
    with sqlite3.connect(db) as conn:
        revenue = conn.execute(
            "SELECT revenue FROM income_statements WHERE company_ticker='EXMP' AND fiscal_year=2025"
        ).fetchone()[0]
    assert revenue == 1e9


def test_a_different_kind_of_breakdown_is_never_mixed_in(filing, tmp_path):
    db = tmp_path / "f.db"
    upsert(db, financials_from_filing(parse_filing(filing)))  # segments Cloud, Devices: 2025
    older = financials_from_filing(parse_filing(filing))
    older.period_end = older.period_end.replace(year=2024)
    older.income_periods[2023] = older.income_periods[2024]
    older.segments = {2023: {"North": 1.0, "South": 2.0}}  # regions, not product lines
    upsert(db, older)
    with sqlite3.connect(db) as conn:
        names = {n for (n,) in conn.execute("SELECT segment_name FROM segment_revenue")}
    assert names == {"Cloud", "Devices"}


def test_add_puts_a_new_company_in_the_database_index_tools_and_prompt(filing, tmp_path):
    db = tmp_path / "financials.db"
    shutil.copyfile(DB, db)
    settings = Settings(
        _env_file=None,
        db_path=db,
        pdf_dir=tmp_path / "no-pdfs",
        filings_dir=tmp_path / "filings",
        index_dir=tmp_path / "index",
    )
    lines = add([str(filing)], settings, years=2, ticker=None, no_embed=True)
    assert lines[0].startswith("Example Corp (EXMP) FY2025 10-K: 3 pages")
    assert (tmp_path / "filings" / "EXMP_FY2025_10-K.htm").exists()

    store = SQLiteFinancialStore(db)
    index = LocalPageIndex(settings.index_dir, embedder=None)
    assert "EXMP" in store.companies() and "Cloud" in store.schema_description()
    tools = ToolBox(store, index)
    assert "EXMP" in tools.tickers and tools.resolve_ticker("Example Corp") == "EXMP"
    prompt = Agent(FakeLLM([]), tools, store.schema_description()).system_prompt
    assert "Example Corp (EXMP)" in prompt and "EXMP (Example Corp): FY2025 10-K" in prompt


def test_add_by_ticker_explains_sec_needs_a_contact(tmp_path):
    settings = Settings(_env_file=None, filings_dir=tmp_path, sec_user_agent="")
    with pytest.raises(SystemExit, match="SEC requires a contact email"):
        add(["NVDA"], settings, years=1, ticker=None, no_embed=True)


class FakeEmbedder:
    model_name = "fake-embed"

    async def embed(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        return [[1.0, float(len(t) % 7), 0.5] for t in texts]


def test_rebuilding_without_an_embedder_keeps_existing_vectors(filing, tmp_path):
    index_dir = tmp_path / "index"
    first = Document(filing, ticker="EXMP", fiscal_year=2025, form="10-K")
    asyncio.run(build_index([first], index_dir, FakeEmbedder(), dims=3))
    other = tmp_path / "other.htm"
    other.write_text(FILING.replace("We make example products.", "Different words."))
    second = Document(other, ticker="OTHR", fiscal_year=2025, form="10-K")
    stats = asyncio.run(build_index([first, second], index_dir, None, dims=3))
    assert stats["embed_model"] == "fake-embed" and stats["pages_without_vectors"] == 1
    vectors = np.load(index_dir / "vectors.npy")
    assert np.all(np.any(vectors[:3], axis=1))  # EXMP's pages kept theirs (one page differs)
    index = LocalPageIndex(index_dir, embedder=FakeEmbedder())
    assert index.dense_enabled


def test_edgar_download_finds_the_latest_10ks(tmp_path):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["user-agent"])
        url = str(request.url)
        if url == edgar.TICKERS_URL:
            return httpx.Response(200, json={"0": {"cik_str": 1045810, "ticker": "NVDA"}})
        if "submissions" in url:
            recent = {
                "form": ["10-Q", "10-K", "8-K", "10-K"],
                "accessionNumber": ["a-1", "0001045810-25-000023", "a-3", "0001045810-24-000029"],
                "primaryDocument": ["q.htm", "nvda-20250126.htm", "e.htm", "nvda-20240128.htm"],
            }
            return httpx.Response(200, json={"filings": {"recent": recent}})
        return httpx.Response(200, text=f"<html>{url}</html>")

    files = edgar.download_10ks(
        "nvda", 1, tmp_path, "Jane Doe jane@example.com", transport=httpx.MockTransport(handler)
    )
    assert len(files) == 1 and "000104581025000023/nvda-20250126.htm" in files[0].read_text()
    assert set(seen) == {"Jane Doe jane@example.com"}


def test_without_a_database_the_prompt_is_about_the_documents(index):
    prompt = Agent(FakeLLM([]), ToolBox(None, index), "").system_prompt
    assert "## Database" not in prompt and "query_database" not in prompt
    assert "AAPL (Apple Inc.)" not in prompt  # names come from the database, absent here
    assert "AAPL: FY2024 10-K, FY2025 10-K" in prompt


def test_bundled_prompt_keeps_the_data_boundaries(store, index):
    prompt = Agent(FakeLLM([]), ToolBox(store, index), store.schema_description()).system_prompt
    assert "Nothing before FY2022 exists anywhere" in prompt
    assert "AAPL September, GOOGL December, MSFT June" in prompt
    assert all(note in prompt for note in ("hedging gains", "GEOGRAPHIC", "restated"))
    assert json.dumps(prompt)  # serialisable, static text
