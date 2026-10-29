"""Filing (PDF, or HTML as filed on EDGAR) → pages with layout-preserving text and 10-K
section tags.

Why layout-preserving: these PDFs were rendered from SEC HTML, and PyMuPDF's default
reading order emits a table's label column first and its numbers many lines later
("YouTube ads ... 36,147"). `sort=True` reconstructs visual lines, so table rows stay intact
("YouTube ads | 36,147 | 40,367"). That matters because several answers are PDF-only numbers.
HTML filings keep rows intact by construction (see html_filing.py).

Why a page is the retrieval unit: filings are only ~60-110 pages, tables never get split
across chunks, and a citation is simply "AAPL FY2025 10-K p.23", which a user can open.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf

from app.core.ports import Page
from app.ingest.html_filing import parse_filing

EXTRACTOR_VERSION = "pages-v2"

ITEM_TITLES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity and Stockholder Matters",
    "6": "[Reserved]",
    "7": "MD&A (Management's Discussion and Analysis)",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

_FILENAME = re.compile(
    r"(?P<ticker>[A-Z0-9][A-Z0-9.-]*)_FY(?P<fy>\d{4})_(?P<form>10-K|10-Q)\.(?:pdf|html?)$", re.I
)
# A real heading: "Item 7. Management's ..." / "ITEM 1A. RISK FACTORS" (title starts upper/[).
_HEADING = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*\.\s+([A-Z\[’'].{2,})$", re.I)
# In HTML the title can follow with no space ("ITEM 7.MANAGEMENT'S") or in the next cell.
_HEADING_HTML = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*\.\s*(?:\|\s*)?([A-Z\[’'].{2,})$", re.I)
# Running page header used by Microsoft: a line that is just "Item 1A".
_RUNNING_HEADER = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*$", re.I)
# Table-of-contents rows end with a page number.
_TOC_ROW = re.compile(r"\s\d{1,3}\s*$")
_MULTISPACE = re.compile(r"[ \t]{3,}")
_DOLLAR_CELL = re.compile(r"\$\s*\|\s*")


def _normalize(text: str) -> str:
    text = text.replace("\xa0", " ").replace("​", "")
    lines = []
    for line in text.splitlines():
        line = _MULTISPACE.sub(" | ", line.strip())
        line = _DOLLAR_CELL.sub("$", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _item_label(code: str) -> str:
    return f"Item {code}. {ITEM_TITLES.get(code, '')}".strip()


def parse_filename(path: Path) -> tuple[str, int, str]:
    m = _FILENAME.search(path.name)
    if not m:
        raise ValueError(f"Unexpected filing name (want TICKER_FY2025_10-K.pdf): {path.name}")
    return m["ticker"].upper(), int(m["fy"]), m["form"].upper()


def extract_pages(
    path: Path,
    *,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    form: str | None = None,
    period: str = "",
    company: str = "",
) -> list[Page]:
    """Pages of one filing. Ticker, year and form default to what the file name says
    (TICKER_FY2025_10-K.pdf); an HTML filing also names its company."""
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as doc:
            raws = [str(doc[i].get_text(sort=True)) for i in range(doc.page_count)]
        heading = _HEADING
    else:
        filing = parse_filing(path)
        raws, heading = filing.pages, _HEADING_HTML
        company = company or filing.company
    if ticker is None or fiscal_year is None or form is None:
        name_ticker, name_fy, name_form = parse_filename(path)
        ticker, fiscal_year, form = ticker or name_ticker, fiscal_year or name_fy, form or name_form
    whole_year_10k = form == "10-K" and not period
    doc_id = f"{ticker}_FY{fiscal_year}" if whole_year_10k else f"{ticker}_{period}_{form}"
    pages: list[Page] = []
    current: str | None = None  # item code in effect at the top of the next page
    for idx, raw in enumerate(raws):
        text = _normalize(raw)
        lines = raw.splitlines()
        headings = [
            m.group(1).upper()
            for line in lines
            if (m := heading.match(line.strip())) and not _TOC_ROW.search(line)
        ]
        toc_rows = sum(1 for line in lines if heading.match(line.strip()) and _TOC_ROW.search(line))
        if toc_rows >= 5:
            # A TOC page can still contain the first real heading (Apple FY2024 p.3).
            items: list[str] = list(dict.fromkeys(headings))
            if headings:
                current = headings[-1]
            section = "; ".join(["Table of Contents"] + [_item_label(c) for c in items])
        else:
            running = [m.group(1).upper() for line in lines if (m := _RUNNING_HEADER.match(line))]
            if not headings and running:
                current = running[0]
            items = ([current] if current else []) + [h for h in headings if h != current]
            items = list(dict.fromkeys(items))
            if headings:
                current = headings[-1]
            section = "; ".join(_item_label(c) for c in items) if items else "Cover"
        pages.append(
            Page(
                doc_id=doc_id,
                ticker=ticker,
                fiscal_year=fiscal_year,
                company=company,
                form=form,
                period=period,
                page=idx + 1,
                section=section,
                items=items,
                text=text,
                sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
    return pages


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
