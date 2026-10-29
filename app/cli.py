"""Terminal interface.

uv run python -m app.cli                      # interactive REPL (keeps conversation context)
uv run python -m app.cli "What was Apple's FY2025 revenue?"   # one-shot
Commands in the REPL: /new (reset context), /trace (toggle trace), /quit
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from app.container import build_container
from app.core.config import get_settings
from app.core.errors import AppError
from app.service import ChatResult


def _print_result(r: ChatResult, show_trace: bool) -> None:
    print(f"\n{r.answer}\n")
    g = r.grounding or {}
    flags = []
    if r.cached:
        flags.append("cache hit")
    if r.degraded:
        flags.append("keyword-only search")
    if g.get("unsupported"):
        flags.append(f"UNVERIFIED numbers: {', '.join(g['unsupported'])}")
    else:
        flags.append(f"{g.get('checked', 0)} numbers grounded")
    print(
        f"  [{' | '.join(flags)}] tools={','.join(r.tools_used) or '-'} "
        f"{r.latency_ms / 1000:.1f}s ${r.cost_usd:.4f}"
    )
    if show_trace:
        for s in r.trace:
            print(f"   {s['step']:>2}. {s['kind']:<8} {s.get('name') or ''} {s.get('args') or ''}")


async def _event(kind: str, data: dict[str, Any]) -> None:
    if kind == "tool_call":
        print(f"  → {data['name']}", flush=True)
    elif kind == "rewrite":
        print(f"  (interpreting as: {data['question']})", flush=True)
    elif kind in ("repair", "escalate"):
        print(f"  (re-checking numbers: {kind})", flush=True)


async def main_async(argv: list[str]) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    c = build_container(get_settings())
    if not c.ready or c.service is None:
        print("Not ready:\n  " + "\n  ".join(c.problems), file=sys.stderr)
        return 2
    service = c.service
    if argv:
        try:
            r = await service.ask(" ".join(argv), on_event=_event)
        except AppError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_result(r, show_trace=False)
        return 0

    docs = c.index.catalog() if c.index else []
    years = sorted({d.fiscal_year for d in docs})
    scope = "/".join(sorted({d.ticker for d in docs})) + (
        f", FY{years[0]}-FY{years[-1]}" if years else ""
    )
    print(f"10-K Research Assistant ({scope}). /new, /trace, /quit")
    history: list[dict[str, str]] = []
    show_trace = False
    while True:
        try:
            q = (await asyncio.to_thread(input, "\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q in ("/quit", "/exit", "/q"):
            return 0
        if q == "/new":
            history = []
            print("  (new conversation)")
            continue
        if q == "/trace":
            show_trace = not show_trace
            print(f"  (trace {'on' if show_trace else 'off'})")
            continue
        try:
            r = await service.ask(q, history=history, on_event=_event)
        except AppError as exc:
            print(f"  error: {exc}")
            continue
        _print_result(r, show_trace)
        history = (
            [*history, {"role": "user", "content": q}, {"role": "assistant", "content": r.answer}]
        )[-10:]


def main() -> None:
    sys.exit(asyncio.run(main_async(sys.argv[1:])))


if __name__ == "__main__":
    main()
