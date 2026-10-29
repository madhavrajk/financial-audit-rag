"""Retrieval ablation: BM25 vs dense vs hybrid (RRF), recall@k and MRR on gold pages.

  uv run python -m eval.retrieval

Queries carry the ticker/fiscal-year filters the agent would set, so this measures ranking
within one filing, which is the decision that matters.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from app.adapters.local_index import LocalPageIndex
from app.adapters.openai_compat import OpenAICompatEmbedder, make_client
from app.core.config import PROJECT_ROOT, get_settings

GOLD = PROJECT_ROOT / "eval" / "retrieval_gold.json"
KS = (1, 3, 5)


async def evaluate(index: LocalPageIndex, modes: list[str]) -> dict[str, Any]:
    gold = json.loads(GOLD.read_text())
    report: dict[str, Any] = {}
    for mode in modes:
        hits_at = {k: 0 for k in KS}
        rr = 0.0
        per_q = {}
        for g in gold:
            res = await index.search(
                g["query"], ticker=g["ticker"], fiscal_year=g["fiscal_year"], k=max(KS), mode=mode
            )
            ranked = [h.page.page for h in res.hits]
            first = next((i for i, p in enumerate(ranked) if p in g["gold_pages"]), None)
            for k in KS:
                hits_at[k] += first is not None and first < k
            rr += 1 / (first + 1) if first is not None else 0
            per_q[g["id"]] = ranked
        n = len(gold)
        report[mode] = {
            **{f"recall@{k}": round(hits_at[k] / n, 2) for k in KS},
            "mrr": round(rr / n, 3),
            "ranked": per_q,
        }
    return report


def main() -> None:
    s = get_settings()
    embedder = None
    if s.embed_model and (s.embed_key or s.is_local(s.embed_url)):
        embedder = OpenAICompatEmbedder(
            make_client(s.embed_key, s.embed_url, 60), model=s.embed_model, dims=s.embed_dims
        )
    index = LocalPageIndex(s.index_dir, embedder)
    modes = ["bm25"] + (["dense", "hybrid"] if index.dense_enabled else [])
    report = asyncio.run(evaluate(index, modes))
    print("| mode | " + " | ".join(f"recall@{k}" for k in KS) + " | MRR |")
    print("|---|" + "---|" * (len(KS) + 1))
    for mode, r in report.items():
        print(f"| {mode} | " + " | ".join(str(r[f"recall@{k}"]) for k in KS) + f" | {r['mrr']} |")
    out = PROJECT_ROOT / "eval" / "results" / "retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
