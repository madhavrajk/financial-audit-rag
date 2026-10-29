"""Load test one server process: throughput, latency and errors as concurrency rises.

  # terminal 1 (cache and per-client rate limit off = worst case)
  ANSWER_CACHE=none RATE_LIMIT_PER_MINUTE=100000 PORT=8020 uv run python -m app.api.server
  # terminal 2
  uv run python -m eval.load_test --url http://127.0.0.1:8020 --levels 1,5,10,25,50

At each level, N concurrent clients each send `--per-client` questions (the eval questions,
rotated) back to back. Reports requests/min, p50/p95 latency, error codes and $/question.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from typing import Any

import httpx

from app.core.config import PROJECT_ROOT

QUESTIONS = [
    q["question"] for q in json.loads((PROJECT_ROOT / "eval" / "questions.json").read_text())
]


async def client(
    http: httpx.AsyncClient, url: str, start: int, n: int, out: list[dict[str, Any]]
) -> None:
    for i in range(n):
        q = QUESTIONS[(start + i) % len(QUESTIONS)]
        t0 = time.monotonic()
        try:
            r = await http.post(f"{url}/api/chat", json={"question": q}, timeout=300)
            body = r.json()
            out.append(
                {
                    "status": r.status_code,
                    "latency": time.monotonic() - t0,
                    "cost": body.get("cost_usd", 0.0),
                    "unsupported": len((body.get("grounding") or {}).get("unsupported", [])),
                }
            )
        except httpx.HTTPError as exc:
            out.append(
                {
                    "status": type(exc).__name__,
                    "latency": time.monotonic() - t0,
                    "cost": 0.0,
                    "unsupported": 0,
                }
            )


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))] if xs else 0.0


async def main_async(args: argparse.Namespace) -> None:
    rows = []
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=200)) as http:
        for level in [int(x) for x in args.levels.split(",")]:
            out: list[dict[str, Any]] = []
            t0 = time.monotonic()
            await asyncio.gather(
                *(client(http, args.url, c, args.per_client, out) for c in range(level))
            )
            wall = time.monotonic() - t0
            ok = [r for r in out if r["status"] == 200]
            lat = [r["latency"] for r in ok]
            row = {
                "concurrency": level,
                "requests": len(out),
                "ok": len(ok),
                "errors": dict(Counter(str(r["status"]) for r in out if r["status"] != 200)),
                "throughput_per_min": round(len(ok) / wall * 60, 1),
                "p50_s": round(pct(lat, 50), 1),
                "p95_s": round(pct(lat, 95), 1),
                "mean_cost_usd": round(statistics.mean([r["cost"] for r in ok]), 5) if ok else 0,
                "answers_with_unsupported_numbers": sum(r["unsupported"] > 0 for r in ok),
            }
            rows.append(row)
            print(json.dumps(row))
    out_path = (
        PROJECT_ROOT / "eval" / "results" / f"load_test_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    out_path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print("| concurrent users | requests | ok | errors | answers/min | p50 | p95 | $/question |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['concurrency']} | {r['requests']} | {r['ok']} | {r['errors'] or '-'} | "
            f"{r['throughput_per_min']} | {r['p50_s']} s | {r['p95_s']} s | "
            f"${r['mean_cost_usd']:.4f} |"
        )
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8020")
    p.add_argument("--levels", default="1,5,10,25,50")
    p.add_argument("--per-client", type=int, default=2)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
