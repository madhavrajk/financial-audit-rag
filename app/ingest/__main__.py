"""Build the page index, or add companies.

uv run python -m app.ingest                rebuild the index (embeds new pages; needs EMBED_MODEL)
uv run python -m app.ingest --no-embed     rebuild without calling an embedding API
uv run python -m app.ingest add FILE ...   add 10-Ks saved from sec.gov (HTML) or PDFs
uv run python -m app.ingest add NVDA ...   download the latest 10-Ks (needs SEC_USER_AGENT)
uv run python -m app.ingest database FILE ...
                                           rebuild financials.db from HTML 10-Ks only
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.core.config import get_settings
from app.ingest.add import add
from app.ingest.build_index import rebuild
from app.ingest.financials import build_database


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-embed", action="store_true", help="don't call an embedding API")
    sub = parser.add_subparsers(dest="command")
    p_add = sub.add_parser("add", help="add filings for any company")
    p_add.add_argument("targets", nargs="+", help="10-K files (.htm or .pdf) or tickers")
    p_add.add_argument("--years", type=int, default=2, help="10-Ks per ticker to download")
    p_add.add_argument("--ticker", help="ticker to file a filing under, if it names none")
    p_add.add_argument("--no-embed", action="store_true", help="don't call an embedding API")
    p_db = sub.add_parser("database", help="rebuild the database from HTML 10-Ks")
    p_db.add_argument("files", nargs="+", type=Path, help="10-Ks saved from sec.gov (.htm)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if args.command == "database":
        for line in build_database(args.files, get_settings().db_path):
            print(line)
        return
    if args.command == "add":
        for line in add(
            args.targets,
            get_settings(),
            years=args.years,
            ticker=args.ticker,
            no_embed=args.no_embed,
        ):
            print(line)
        return
    stats = rebuild(args.no_embed)
    print(json.dumps({k: v for k, v in stats.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
