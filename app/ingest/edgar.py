"""Optional: download a company's latest 10-K filings from SEC EDGAR by ticker.

SEC requires automated requests to name a contact (requests without one get 403 Forbidden),
so this needs SEC_USER_AGENT, e.g. "Jane Doe jane@example.com". Without it, save the 10-K
from sec.gov in a browser and pass the file to `ingest add` instead. Requests are sequential,
far below SEC's limit of 10 per second.
"""

from __future__ import annotations

from pathlib import Path

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


def latest_10k_urls(client: httpx.Client, ticker: str, count: int) -> list[str]:
    """URLs of the primary documents of the company's `count` most recent 10-Ks."""
    resp = client.get(TICKERS_URL)
    resp.raise_for_status()
    entry = next((e for e in resp.json().values() if e["ticker"].upper() == ticker.upper()), None)
    if entry is None:
        raise ValueError(f"SEC lists no company with ticker {ticker}")
    cik = int(entry["cik_str"])
    resp = client.get(SUBMISSIONS_URL.format(cik=cik))
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]
    urls = [
        DOCUMENT_URL.format(cik=cik, accession=acc.replace("-", ""), document=doc)
        for form, acc, doc in zip(
            recent["form"], recent["accessionNumber"], recent["primaryDocument"], strict=True
        )
        if form == "10-K"
    ]
    return urls[:count]


def download_10ks(
    ticker: str,
    count: int,
    dest: Path,
    user_agent: str,
    transport: httpx.BaseTransport | None = None,
) -> list[Path]:
    """Save the latest 10-Ks as HTML in `dest`; returns the files written."""
    dest.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": user_agent}
    paths = []
    with httpx.Client(headers=headers, timeout=60, transport=transport) as client:
        for i, url in enumerate(latest_10k_urls(client, ticker, count)):
            resp = client.get(url)
            resp.raise_for_status()
            path = dest / f"{ticker.upper()}-download-{i}.htm"
            path.write_bytes(resp.content)
            paths.append(path)
    return paths
