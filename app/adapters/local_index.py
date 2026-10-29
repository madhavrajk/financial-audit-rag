"""In-memory hybrid page retriever (the Retriever port): BM25 + dense cosine, fused with RRF.

In-memory because 516 pages x 1024 dims x 4 bytes ≈ 2 MB: exact search takes under 5 ms
with perfect recall, so a vector database would add a service and no quality. The index is
immutable after load, so concurrent requests share it without locks.

Hybrid because BM25 matches exact financial terms ("YouTube ads", "Greater China") and dense
vectors catch paraphrases ("why did China sales fall"). Reciprocal Rank Fusion (Cormack et
al., 2009) merges the two rankings without calibrating their scores.

If vectors are missing or the embedding call fails, search falls back to BM25 alone and
marks the result `degraded=True`. Pages added without an embedder have no vector and are
ranked by BM25 only.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from app.core.errors import NotReadyError
from app.core.ports import DocumentInfo, Embedder, Page, PageHit, SearchResult, model_key

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")
_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d{3})")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "which",
        "with",
        "how",
        "did",
        "does",
        "do",
        "their",
        "our",
        "we",
    ]
)
RRF_K = 60


def tokenize(text: str) -> list[str]:
    text = _DIGIT_COMMA.sub("", text.lower())  # "36,147" -> "36147"
    return [t for t in _TOKEN.findall(text) if t not in _STOP]


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75) over a fixed corpus."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.len.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str], candidates: np.ndarray) -> np.ndarray:
        out = np.zeros(len(candidates), dtype=np.float32)
        for j, i in enumerate(candidates):
            tf, norm = self.tf[i], self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
            s = 0.0
            for t in query:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out[j] = s
        return out


class LocalPageIndex:
    def __init__(self, index_dir: Path, embedder: Embedder | None) -> None:
        manifest_path = index_dir / "manifest.json"
        pages_path = index_dir / "pages.jsonl"
        if not manifest_path.exists() or not pages_path.exists():
            raise NotReadyError(
                f"Search index not found in {index_dir}. Build it with: uv run python -m app.ingest"
            )
        self.manifest = json.loads(manifest_path.read_text())
        self.index_version: str = self.manifest["index_version"]
        with pages_path.open() as fh:
            self.pages = [Page.model_validate_json(line) for line in fh if line.strip()]
        self._by_doc: dict[str, list[Page]] = {}
        for p in self.pages:
            self._by_doc.setdefault(_doc_key(p.document), []).append(p)
        self._bm25 = BM25([tokenize(f"{p.header}\n{p.text}") for p in self.pages])
        self._ticker = np.array([p.ticker for p in self.pages])
        self._fy = np.array([p.fiscal_year for p in self.pages])
        self._form = np.array([p.form.upper() for p in self.pages])

        self._vectors: np.ndarray | None = None
        self._has_vector = np.zeros(len(self.pages), dtype=bool)
        vec_path = index_dir / "vectors.npy"
        if vec_path.exists():
            vectors = np.load(vec_path)
            if vectors.shape[0] != len(self.pages):
                raise NotReadyError("vectors.npy does not match pages.jsonl; rebuild the index")
            self._vectors = vectors.astype(np.float32)
            self._has_vector = np.any(self._vectors, axis=1)
        self._embedder = embedder
        self._embed_model = self.manifest.get("embed_model")
        if (
            embedder is not None
            and self._embed_model
            and model_key(embedder.model_name) != model_key(self._embed_model)
        ):
            log.warning(
                "index embedded with %s but EMBED_MODEL=%s; dense search disabled "
                "(rebuild with: uv run python -m app.ingest)",
                self._embed_model,
                embedder.model_name,
            )
            self._vectors = None
        self._qcache: OrderedDict[str, np.ndarray] = OrderedDict()  # query-embedding LRU

    @property
    def dense_enabled(self) -> bool:
        return self._vectors is not None and self._embedder is not None

    def _candidates(
        self,
        ticker: str | None,
        fiscal_year: int | None,
        section: str | None,
        form: str | None = None,
    ) -> np.ndarray:
        mask = np.ones(len(self.pages), dtype=bool)
        if ticker:
            mask &= self._ticker == ticker.upper()
        if fiscal_year:
            mask &= self._fy == fiscal_year
        if form:
            mask &= self._form == form.upper()
        if section:
            code = section.upper().removeprefix("ITEM").strip().rstrip(".")
            mask &= np.array([code in p.items for p in self.pages])
        return np.flatnonzero(mask)

    async def _query_vector(self, query: str) -> np.ndarray | None:
        if not self.dense_enabled:
            return None
        if query in self._qcache:
            self._qcache.move_to_end(query)
            return self._qcache[query]
        assert self._embedder is not None
        try:
            vec = np.asarray(
                (await self._embedder.embed([query], is_query=True))[0], dtype=np.float32
            )
        except Exception as exc:  # degrade, never fail the request because of dense search
            log.warning("query embedding failed, falling back to BM25 only: %s", exc)
            return None
        vec /= np.linalg.norm(vec) or 1.0
        self._qcache[query] = vec
        if len(self._qcache) > 512:
            self._qcache.popitem(last=False)
        return vec

    async def search(
        self,
        query: str,
        *,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        form: str | None = None,
        section: str | None = None,
        k: int = 4,
        mode: str = "hybrid",  # "hybrid" | "bm25" | "dense" (ablations)
    ) -> SearchResult:
        cand = self._candidates(ticker, fiscal_year, section, form)
        if len(cand) == 0:
            return SearchResult(
                hits=[], note=f"No filing pages match those filters. {self.coverage(ticker)}"
            )
        bm25 = self._bm25.scores(tokenize(query), cand)
        order = np.argsort(-bm25, kind="stable")
        # Pages with zero lexical overlap get no BM25 rank (they can still come from dense).
        bm25_rank = {int(cand[j]): r for r, j in enumerate(order) if bm25[j] > 0}
        if mode == "dense":
            bm25_rank = {}

        qvec = await self._query_vector(query) if mode != "bm25" else None
        degraded = qvec is None  # dense unavailable (no vectors, no key, or API failure)
        dense_rank: dict[int, int] = {}
        if qvec is not None and self._vectors is not None:
            embedded = cand[self._has_vector[cand]]
            sims = self._vectors[embedded] @ qvec
            dense_rank = {
                int(i): r for r, i in enumerate(embedded[np.argsort(-sims, kind="stable")])
            }

        fused: dict[int, float] = {}
        for ranks in (bm25_rank, dense_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        hits = [PageHit(page=self.pages[i], score=round(s, 5)) for i, s in top]
        return SearchResult(
            hits=hits, degraded=degraded, note="" if hits else "No relevant pages found."
        )

    def get_pages(self, document: str, start: int, end: int) -> list[Page]:
        doc = self._by_doc.get(_doc_key(document), [])
        return [p for p in doc if start <= p.page <= end]

    def section_pages(self, document: str, section: str) -> list[Page]:
        return [p for p in self.get_pages(document, 1, 10_000) if section.upper() in p.items]

    def catalog(self) -> list[DocumentInfo]:
        docs = [
            DocumentInfo(
                name=ps[0].document,
                ticker=ps[0].ticker,
                company=ps[0].company,
                fiscal_year=ps[0].fiscal_year,
                form=ps[0].form,
                pages=len(ps),
            )
            for ps in self._by_doc.values()
        ]
        return sorted(docs, key=lambda d: (d.ticker, d.fiscal_year, d.name))

    def coverage(self, ticker: str | None = None) -> str:
        """Which documents exist, for a "nothing matched" message."""
        docs = self.catalog()
        mine = [d.name for d in docs if ticker and d.ticker == ticker.upper()]
        if mine:
            return f"Documents for {ticker.upper() if ticker else ''}: {', '.join(mine)}."
        return f"Companies with documents: {', '.join(sorted({d.ticker for d in docs}))}."

    def available_docs(self) -> list[str]:
        return [f"{d.name} ({d.pages} pages)" for d in self.catalog()]


def _doc_key(document: str) -> str:
    return " ".join(document.upper().split())
