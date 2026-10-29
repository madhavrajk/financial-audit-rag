"""Annual financials from a filing's inline XBRL → rows for financials.db.

A 10-K tags every number in its statements with a concept and a context (period, plus any
dimensions), so a company's database rows come from the same document the agent reads, with
no separate data feed. Statement lines map to standard US-GAAP concepts, with fallbacks for
companies that tag the same line differently; the first concept a filing reports for its own
fiscal year is used for every year in it. Segment and region revenue are revenue facts broken
down by one segment or geography dimension, named by the table row they appear in.

The bundled database is built the same way, from the FY2025 10-Ks:
uv run python -m app.ingest database aapl-20250927.htm goog-20251231.htm msft-20250630.htm
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from app.ingest.html_filing import Context, ParsedFiling, parse_filing

INCOME: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
    ],
    "gross_profit": ["GrossProfit"],
    "research_and_development": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "total_operating_expenses": ["OperatingExpenses", "CostsAndExpenses"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}
BALANCE: dict[str, list[str]] = {
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash_and_equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "total_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "short_term_debt": ["CommercialPaper", "ShortTermBorrowings", "DebtCurrent"],
    "accounts_receivable": ["AccountsReceivableNetCurrent"],
    "total_current_assets": ["AssetsCurrent"],
    "total_current_liabilities": ["LiabilitiesCurrent"],
}
SEGMENT_AXIS = "us-gaap:StatementBusinessSegmentsAxis"
REGION_AXIS = "srt:StatementGeographicalAxis"
_CLASS_AXIS = "us-gaap:StatementClassOfStockAxis"
_MONTHS = "January February March April May June July August September October November December"


def _columns(mapping: dict[str, list[str]]) -> str:
    return "".join(f"    {c} {'REAL' if c.startswith('eps_') else 'BIGINT'},\n" for c in mapping)


# Money in raw US dollars, EPS in dollars per share; period_type is 'FY' (annual).
SCHEMA = f"""\
CREATE TABLE IF NOT EXISTS companies (
    ticker TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    cik TEXT NOT NULL,
    fiscal_year_end TEXT
);
CREATE TABLE IF NOT EXISTS income_statements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_ticker TEXT NOT NULL REFERENCES companies(ticker),
    fiscal_year INTEGER NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    period_type TEXT NOT NULL,
{_columns(INCOME)}    UNIQUE(company_ticker, period_end, period_type)
);
CREATE TABLE IF NOT EXISTS balance_sheets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_ticker TEXT NOT NULL REFERENCES companies(ticker),
    fiscal_year INTEGER NOT NULL,
    period_end TEXT NOT NULL,
    period_type TEXT NOT NULL,
{_columns(BALANCE)}    UNIQUE(company_ticker, period_end, period_type)
);
CREATE TABLE IF NOT EXISTS segment_revenue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_ticker TEXT NOT NULL REFERENCES companies(ticker),
    fiscal_year INTEGER NOT NULL,
    period_end TEXT NOT NULL,
    period_type TEXT NOT NULL,
    segment_name TEXT NOT NULL,
    revenue BIGINT NOT NULL,
    UNIQUE(company_ticker, period_end, period_type, segment_name)
);
CREATE TABLE IF NOT EXISTS geographic_revenue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_ticker TEXT NOT NULL REFERENCES companies(ticker),
    fiscal_year INTEGER NOT NULL,
    period_end TEXT NOT NULL,
    period_type TEXT NOT NULL,
    region TEXT NOT NULL,
    revenue BIGINT NOT NULL,
    UNIQUE(company_ticker, period_end, period_type, region)
);
"""


def _neutral(axis: str, member: str) -> bool:
    """Newer filings also tag segment facts "operating segments" (us-gaap: or srt: prefix),
    which adds no breakdown of its own."""
    return axis.endswith(":ConsolidationItemsAxis") and member.endswith(":OperatingSegmentsMember")


@dataclass
class CompanyFinancials:
    ticker: str
    name: str
    cik: str
    fiscal_year: int  # the filing's own fiscal year
    period_end: date
    income: dict[int, dict[str, float]] = field(default_factory=dict)
    income_periods: dict[int, tuple[date, date]] = field(default_factory=dict)
    balance: dict[int, dict[str, float]] = field(default_factory=dict)
    balance_dates: dict[int, date] = field(default_factory=dict)
    segments: dict[int, dict[str, float]] = field(default_factory=dict)
    regions: dict[int, dict[str, float]] = field(default_factory=dict)

    @property
    def fiscal_year_end(self) -> str:
        return _MONTHS.split()[self.period_end.month - 1]


_FOOTNOTE = re.compile(r"\s*(\(\w{1,2}\)|\[\w{1,2}\]|\*+)$")  # "(1)", "(a)", "[2]", "*"
_TOTAL = re.compile(r"^total\s+|\s+total$", re.I)
_SMALL_WORDS = {"And", "Of", "The", "For", "In", "On", "To"}
# Row labels that name the line, not the segment ("Americas:" heading, then "Net sales").
_GENERIC = {"revenue", "revenues", "net revenue", "net revenues", "net sales", "sales"}


def _clean_label(label: str) -> str:
    while (stripped := _FOOTNOTE.sub("", label)) != label:
        label = stripped
    return _TOTAL.sub("", label).strip()


def _humanize(member: str) -> str:
    """'aapl:RestOfAsiaPacificSegmentMember' -> 'Rest of Asia Pacific'."""
    name = member.split(":")[-1].removesuffix("Member").removesuffix("Segment")
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", name)
    return " ".join(
        w.lower() if i and w in _SMALL_WORDS else w for i, w in enumerate(words)
    ).strip()


def _names(labels: dict[str, Counter[str]], members: set[str]) -> dict[str, str]:
    """Each member's row label, unless the labels don't tell members apart."""
    best = {m: labels[m].most_common(1)[0][0] for m in members if m in labels}
    distinct = len(set(best.values())) == len(best) == len(members) and not any(
        b.lower() in _GENERIC for b in best.values()
    )
    return {m: best[m] if distinct else _humanize(m) for m in members}


def financials_from_filing(parsed: ParsedFiling, ticker: str | None = None) -> CompanyFinancials:
    """Annual statement lines and revenue breakdowns for each fiscal year the filing reports."""
    dei = {k: v[0] for k, v in parsed.dei.items() if v}
    main = parsed.contexts.get(parsed.dei_context)
    if main is None or main.end is None or "DocumentFiscalYearFocus" not in dei:
        raise ValueError("no inline XBRL cover data (fiscal year and period) in this filing")
    fy, end = int(dei["DocumentFiscalYearFocus"]), main.end
    out = CompanyFinancials(
        ticker=(ticker or dei.get("TradingSymbol", "")).upper(),
        name=parsed.company,
        cik=dei.get("EntityCentralIndexKey", ""),
        fiscal_year=fy,
        period_end=end,
    )
    if not out.ticker:
        raise ValueError("the filing does not state a trading symbol; pass --ticker")

    def year_of(d: date | None) -> int | None:
        if d is None:
            return None
        k = round((end - d).days / 365.25)
        return fy - k if 0 <= k <= 4 and abs((end - d).days - k * 365.25) <= 20 else None

    def full_year(c: Context) -> bool:
        return c.start is not None and c.end is not None and 340 <= (c.end - c.start).days <= 380

    # concept -> fiscal year -> value (and its context), for facts with no dimensions
    flat: dict[str, dict[int, float]] = {}
    when: dict[tuple[str, int], Context] = {}
    by_class: dict[str, dict[int, set[float]]] = {}
    for f in parsed.facts:
        ctx = parsed.contexts.get(f.context)
        if ctx is None:
            continue
        year = year_of(ctx.end)
        if year is None or not (ctx.instant or full_year(ctx)):
            continue
        concept = f.concept.split(":")[-1]
        if not ctx.dims:
            flat.setdefault(concept, {}).setdefault(year, f.value)
            when.setdefault((concept, year), ctx)
        elif set(ctx.dims) == {_CLASS_AXIS}:  # per-share data reported per share class
            by_class.setdefault(concept, {}).setdefault(year, set()).add(f.value)
    for concept, years in by_class.items():  # same EPS for every class -> use it
        for year, per_class in years.items():
            if len(per_class) == 1:
                flat.setdefault(concept, {}).setdefault(year, next(iter(per_class)))

    for target, mapping in ((out.income, INCOME), (out.balance, BALANCE)):
        for column, candidates in mapping.items():
            concept = next((c for c in candidates if fy in flat.get(c, {})), None)
            for year, value in flat.get(concept or "", {}).items():
                target.setdefault(year, {})[column] = value
                # A statement's dates are those of its own lines (revenue, total assets), not
                # of the cover page's share count, which can fall days after the year end.
                ctx = when.get((concept or "", year))
                if ctx is not None and ctx.end is not None and ctx.instant:
                    out.balance_dates.setdefault(year, ctx.end)
                elif ctx is not None and ctx.start is not None and ctx.end is not None:
                    out.income_periods.setdefault(year, (ctx.start, ctx.end))
    # Keep whole statements only: an equity roll-forward also reports older year-ends.
    out.income = {
        y: r
        for y, r in out.income.items()
        if y in out.income_periods and ("revenue" in r or "net_income" in r)
    }
    out.balance = {
        y: r for y, r in out.balance.items() if y in out.balance_dates and "total_assets" in r
    }

    revenue_concepts = {f"us-gaap:{c}" for c in INCOME["revenue"]}
    for axis, target in ((SEGMENT_AXIS, out.segments), (REGION_AXIS, out.regions)):
        labels: dict[str, Counter[str]] = {}
        values: dict[tuple[int, str], float] = {}
        for f in parsed.facts:
            ctx = parsed.contexts.get(f.context)
            if f.concept not in revenue_concepts or ctx is None or not full_year(ctx):
                continue
            dims = {a: m for a, m in ctx.dims.items() if not _neutral(a, m)}
            year = year_of(ctx.end)
            if year is None or set(dims) != {axis}:
                continue
            member = dims[axis]
            values.setdefault((year, member), f.value)
            if label := _clean_label(f.row_label):
                labels.setdefault(member, Counter())[label] += 1
        names = _names(labels, {member for _, member in values})
        for (year, member), value in values.items():
            target.setdefault(year, {})[names[member]] = value
    return out


def read_10k(path: Path, ticker: str | None = None) -> tuple[ParsedFiling, CompanyFinancials]:
    """An HTML 10-K and its annual financials; other forms are rejected."""
    filing = parse_filing(path)
    form = (filing.dei.get("DocumentType") or [""])[0]
    if form != "10-K":
        raise ValueError(f"{path.name}: only 10-K filings can be added (this is {form or '?'})")
    return filing, financials_from_filing(filing, ticker)


def build_database(paths: Iterable[Path], db_path: Path) -> list[str]:
    """A new database from HTML 10-Ks, replacing db_path atomically. Returns a line per filing."""
    tmp = db_path.with_name(db_path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    lines = []
    for path in paths:
        _, fin = read_10k(path)
        counts = upsert(tmp, fin)
        lines.append(
            f"{fin.name} ({fin.ticker}) FY{fin.fiscal_year} 10-K: "
            + ", ".join(f"{n} {kind}" for kind, n in counts.items())
        )
    os.replace(tmp, db_path)
    return lines


def upsert(db_path: Path, f: CompanyFinancials) -> dict[str, int]:
    """Write a filing's rows (creating the tables if needed). The newest filing for a company
    wins (it carries restatements); an older one only fills years that are missing. Segment
    and region rows are only added for years with none, and only if the names match the
    company's existing breakdown (if any), so one company's rows never mix two kinds of
    breakdown (say, product lines for some years and regions for others)."""
    counts = {"income": 0, "balance": 0, "segments": 0, "regions": 0}
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        with conn:
            latest = conn.execute(
                "SELECT max(period_end) FROM income_statements WHERE company_ticker = ?",
                (f.ticker,),
            ).fetchone()[0]
            newest = not latest or f.period_end.isoformat() >= latest
            verb = "INSERT OR REPLACE" if newest else "INSERT OR IGNORE"
            conn.execute(
                "INSERT INTO companies (ticker, name, cik, fiscal_year_end) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(ticker) DO UPDATE SET fiscal_year_end = excluded.fiscal_year_end",
                (f.ticker, f.name, f.cik, f.fiscal_year_end),
            )
            for year, row in f.income.items():
                start, end = f.income_periods[year]
                cols = list(INCOME)
                conn.execute(
                    f"{verb} INTO income_statements (company_ticker, fiscal_year, period_start, "
                    f"period_end, period_type, {', '.join(cols)}) "
                    f"VALUES (?, ?, ?, ?, 'FY', {', '.join('?' * len(cols))})",
                    [f.ticker, year, start.isoformat(), end.isoformat()]
                    + [row.get(c) for c in cols],
                )
                counts["income"] += 1
            for year, row in f.balance.items():
                cols = list(BALANCE)
                conn.execute(
                    f"{verb} INTO balance_sheets (company_ticker, fiscal_year, period_end, "
                    f"period_type, {', '.join(cols)}) "
                    f"VALUES (?, ?, ?, 'FY', {', '.join('?' * len(cols))})",
                    [f.ticker, year, f.balance_dates[year].isoformat()]
                    + [row.get(c) for c in cols],
                )
                counts["balance"] += 1
            for table, column, data, key in (
                ("segment_revenue", "segment_name", f.segments, "segments"),
                ("geographic_revenue", "region", f.regions, "regions"),
            ):
                existing = {
                    name
                    for (name,) in conn.execute(
                        f"SELECT DISTINCT {column} FROM {table} WHERE company_ticker = ?",
                        (f.ticker,),
                    )
                }
                incoming = {name for rows in data.values() for name in rows}
                if existing and not existing & incoming:
                    continue  # a different kind of breakdown than the one stored
                for year, rows in data.items():
                    have = conn.execute(
                        f"SELECT 1 FROM {table} WHERE company_ticker = ? AND fiscal_year = ?",
                        (f.ticker, year),
                    ).fetchone()
                    if have or year not in f.income_periods:
                        continue
                    for name, value in rows.items():
                        conn.execute(
                            f"INSERT OR IGNORE INTO {table} (company_ticker, fiscal_year, "
                            f"period_end, period_type, {column}, revenue) "
                            "VALUES (?, ?, ?, 'FY', ?, ?)",
                            (f.ticker, year, f.income_periods[year][1].isoformat(), name, value),
                        )
                        counts[key] += 1
    finally:
        conn.close()
    return counts
