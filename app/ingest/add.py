"""Add any company: its 10-K pages go into the search index and, from an HTML filing, its
annual financials go into the database (from the filing's inline XBRL).

A target is a 10-K saved from sec.gov (the .htm document, as filed), a PDF named
TICKER_FY2025_10-K.pdf, or a ticker to download from EDGAR (needs SEC_USER_AGENT).
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from app.core.config import Settings
from app.ingest import edgar
from app.ingest.build_index import build_index, discover, make_embedder
from app.ingest.extract import parse_filename
from app.ingest.financials import read_10k, upsert

NO_USER_AGENT = (
    "{target!r} is not a file. To download filings by ticker, SEC requires a contact email: "
    'set SEC_USER_AGENT="Your Name you@example.com" in .env. Or open the 10-K on sec.gov, '
    "save it as HTML, and pass the file instead."
)


def import_html(path: Path, settings: Settings, ticker: str | None = None) -> str:
    """Copy one HTML 10-K into the filings folder and load its financials. Returns a summary."""
    filing, fin = read_10k(path, ticker)
    target = settings.filings_dir / f"{fin.ticker}_FY{fin.fiscal_year}_10-K.htm"
    if path.resolve() != target.resolve():
        shutil.copyfile(path, target)
    counts = upsert(settings.db_path, fin)
    years = sorted(fin.income)
    span = f"FY{years[0]}-FY{years[-1]}" if years else "no statement years"
    return (
        f"{fin.name} ({fin.ticker}) FY{fin.fiscal_year} 10-K: {len(filing.pages)} pages; "
        f"financials {span} ({counts['income']} income, {counts['balance']} balance-sheet, "
        f"{counts['segments']} segment and {counts['regions']} region rows)"
    )


def import_pdf(path: Path, settings: Settings) -> str:
    ticker, fy, form = parse_filename(path)  # PDFs carry no XBRL: metadata comes from the name
    shutil.copyfile(path, settings.filings_dir / path.name)
    return f"{ticker} FY{fy} {form} (PDF): pages only; the database is unchanged"


def add(
    targets: list[str], settings: Settings, *, years: int, ticker: str | None, no_embed: bool
) -> list[str]:
    settings.filings_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for target in targets:
        path = Path(target).expanduser()
        if path.is_file():
            if path.suffix.lower() == ".pdf":
                summaries.append(import_pdf(path, settings))
            else:
                summaries.append(import_html(path, settings, ticker))
            continue
        if not settings.sec_user_agent:
            raise SystemExit(NO_USER_AGENT.format(target=target))
        downloads = settings.filings_dir / ".downloads"
        try:
            for file in edgar.download_10ks(target, years, downloads, settings.sec_user_agent):
                summaries.append(import_html(file, settings, target))
        finally:
            shutil.rmtree(downloads, ignore_errors=True)
    embedder = None if no_embed else make_embedder(required=False)
    docs = discover(settings.pdf_dir, settings.filings_dir)
    stats = asyncio.run(build_index(docs, settings.index_dir, embedder, settings.embed_dims))
    without = stats["pages_without_vectors"]
    summaries.append(
        f"index {stats['index_version']}: {stats['pages']} pages"
        + (
            f", {without} searched by keyword only (set EMBED_MODEL to embed them)"
            if without
            else ""
        )
    )
    return summaries
