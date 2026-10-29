"""Build the page index: extract → (re)embed changed pages → atomic write.

* Incremental: pages are keyed by the sha256 of their text, and vectors for unchanged pages
  are reused when the embedding model and dims match.
* Atomic: everything is written to a temp directory and swapped in with os.replace, so a
  crash mid-build never leaves a half-written index.
* Without an embedder, pages keep the vectors they already had and new pages get none, so
  adding a filing never throws away embeddings; pages without a vector are keyword-searched.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.ports import Embedder, Page, model_key
from app.ingest.extract import EXTRACTOR_VERSION, extract_pages, file_sha256, parse_filename

log = logging.getLogger(__name__)
MAX_EMBED_CHARS = 24_000  # ~6k tokens; the longest pages are well under this


@dataclass(frozen=True)
class Document:
    path: Path
    ticker: str | None = None  # None: read from the file name (TICKER_FY2025_10-K.pdf)
    fiscal_year: int | None = None
    form: str | None = None
    period: str = ""
    company: str = ""


def discover(*dirs: Path) -> list[Document]:
    """Filings named TICKER_FY2025_10-K.pdf (or .htm, as filed on EDGAR) in these folders."""
    docs = []
    for d in dirs:
        for path in sorted(d.glob("*")) if d.is_dir() else []:
            if path.suffix.lower() in {".pdf", ".htm", ".html"}:
                parse_filename(path)  # fail early on a misnamed file
                docs.append(Document(path))
    return docs


def _embed_text(p: Page) -> str:
    # Contextual header (cf. Anthropic "Contextual Retrieval", 2024): the vector knows which
    # company, year and section the page belongs to, not just its words.
    return f"{p.header}\n{p.text}"[:MAX_EMBED_CHARS]


def _load_previous(index_dir: Path) -> tuple[str | None, int | None, dict[str, np.ndarray]]:
    """(embed model, dims, sha256 -> vector) of the current index, if it has vectors."""
    try:
        manifest = json.loads((index_dir / "manifest.json").read_text())
        vectors = np.load(index_dir / "vectors.npy")
        with (index_dir / "pages.jsonl").open() as fh:
            shas = [json.loads(line)["sha256"] for line in fh if line.strip()]
        previous = {s: v for s, v in zip(shas, vectors, strict=True) if np.any(v)}
        return manifest.get("embed_model"), manifest.get("embed_dims"), previous
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return None, None, {}


async def build_index(
    documents: list[Document], index_dir: Path, embedder: Embedder | None, dims: int
) -> dict:
    t0 = time.monotonic()
    if not documents:
        raise FileNotFoundError("No filings to index. See data/README.md.")
    pages: list[Page] = []
    for doc in documents:
        doc_pages = extract_pages(
            doc.path,
            ticker=doc.ticker,
            fiscal_year=doc.fiscal_year,
            form=doc.form,
            period=doc.period,
            company=doc.company,
        )
        log.info("extracted %s: %d pages", doc.path.name, len(doc_pages))
        pages.extend(doc_pages)

    prev_model, prev_dims, previous = _load_previous(index_dir)
    vectors: np.ndarray | None = None
    embed_model, embed_dims = prev_model, prev_dims
    reused = embedded = 0
    if embedder is not None:
        embed_model, embed_dims = model_key(embedder.model_name), dims
        if model_key(prev_model or "") != embed_model or prev_dims != dims:
            previous = {}
        todo = [i for i, p in enumerate(pages) if p.sha256 not in previous]
        if todo:
            log.info("embedding %d pages with %s", len(todo), embedder.model_name)
            out = await embedder.embed([_embed_text(pages[i]) for i in todo], is_query=False)
            previous.update(
                {
                    pages[i].sha256: np.asarray(v, dtype=np.float32)
                    for i, v in zip(todo, out, strict=True)
                }
            )
            embedded = len(todo)
    if previous and embed_dims:
        zero = np.zeros(embed_dims, dtype=np.float32)
        vectors = np.stack([previous.get(p.sha256, zero) for p in pages]).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        reused = sum(p.sha256 in previous for p in pages) - embedded
    else:
        embed_model = embed_dims = None

    source_hashes = {doc.path.name: file_sha256(doc.path) for doc in documents}
    version_src = json.dumps([source_hashes, EXTRACTOR_VERSION, embed_model, dims], sort_keys=True)
    manifest = {
        "index_version": hashlib.sha256(version_src.encode()).hexdigest()[:12],
        "extractor_version": EXTRACTOR_VERSION,
        "embed_model": embed_model,
        "embed_dims": embed_dims,
        "pages": len(pages),
        "pages_without_vectors": (
            int((~np.any(vectors, axis=1)).sum()) if vectors is not None else len(pages)
        ),
        "sources": source_hashes,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    tmp = index_dir.with_name(index_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with (tmp / "pages.jsonl").open("w") as fh:
        for p in pages:
            fh.write(p.model_dump_json() + "\n")
    if vectors is not None:
        np.save(tmp / "vectors.npy", vectors)
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
    old = index_dir.with_name(index_dir.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if index_dir.exists():
        index_dir.replace(old)
    tmp.replace(index_dir)
    shutil.rmtree(old, ignore_errors=True)

    stats = {
        **manifest,
        "embedded": embedded,
        "reused": reused,
        "seconds": round(time.monotonic() - t0, 1),
    }
    log.info("index built: %s", {k: v for k, v in stats.items() if k != "sources"})
    return stats


def make_embedder(required: bool) -> Embedder | None:
    """The configured embedder; None (keyword search for new pages) if not configured."""
    from app.adapters.openai_compat import OpenAICompatEmbedder, make_client
    from app.core.config import get_settings

    s = get_settings()
    if not s.embed_model or (not s.embed_key and not s.is_local(s.embed_url)):
        if required:
            raise SystemExit(
                "EMBED_MODEL (and a key for it) is not set. Set it, or pass --no-embed."
            )
        return None
    client = make_client(s.embed_key, s.embed_url, s.llm_timeout_s)
    return OpenAICompatEmbedder(client, model=s.embed_model, dims=s.embed_dims)


def rebuild(no_embed: bool) -> dict:
    from app.core.config import get_settings

    s = get_settings()
    embedder = None if no_embed else make_embedder(required=True)
    docs = discover(s.pdf_dir, s.filings_dir)
    return asyncio.run(build_index(docs, s.index_dir, embedder, s.embed_dims))
