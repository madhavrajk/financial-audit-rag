"""End-to-end evaluation harness, on the questions in eval/questions.json.

  uv run python -m eval.run_eval --runs 3
  uv run python -m eval.run_eval --model glm-5p3 --judge-model kimi-k3

Runs every question through the same ChatService the API uses (answer cache OFF, so each
run measures the system, not the cache), scores it with the scorer named in the answer key,
and reports accuracy by tier and scorer, routing accuracy, unsupported-number rate, latency
p50/p95, tokens and $/question. Results are tagged with prompt/index/model versions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from app.adapters.openai_compat import OpenAICompatLLM, make_client
from app.container import build_container
from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.errors import AppError
from app.service import ChatService
from eval import scorers

QUESTIONS = PROJECT_ROOT / "eval" / "questions.json"
RESULTS = PROJECT_ROOT / "eval" / "results"


def load_questions() -> list[dict[str, Any]]:
    return json.loads(QUESTIONS.read_text())


async def score_one(
    q: dict[str, Any], answer: str, judge: OpenAICompatLLM | None, judge_model: str
) -> scorers.Score:
    kind = q["evaluation"]
    if kind == "fuzzy_numeric":
        return scorers.fuzzy_numeric(answer, q["gold_answer_numeric"])
    if kind == "exact_match_entity":
        return scorers.exact_match_entity(answer, q["gold_answer"], q.get("gold_answer_numeric"))
    if kind == "refusal":
        return scorers.refusal(answer)
    if kind == "llm_judge":
        if judge is None:
            return scorers.Score(passed=False, score=0.0, detail="judge disabled")
        return await scorers.llm_judge(judge, judge_model, q["question"], q["gold_answer"], answer)
    raise ValueError(f"unknown evaluation type {kind}")


async def run_question(
    service: ChatService,
    q: dict[str, Any],
    judge: OpenAICompatLLM | None,
    judge_model: str,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    async with sem:
        t0 = time.monotonic()
        try:
            r = await service.ask(q["question"], use_cache=False)
        except AppError as exc:
            return {
                "id": q["id"],
                "tier": q["tier"],
                "error": str(exc),
                "passed": False,
                "score": 0.0,
                "evaluation": q["evaluation"],
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }
        s = await score_one(q, r.answer, judge, judge_model)
        route = scorers.routing(r.tools_used, q.get("required_modalities", []))
        return {
            "id": q["id"],
            "tier": q["tier"],
            "evaluation": q["evaluation"],
            "question": q["question"],
            "gold": q["gold_answer"],
            "answer": r.answer,
            "passed": s.passed,
            "score": s.score,
            "score_detail": s.detail,
            "judge_error": s.judge_error,
            "judge_cost_usd": s.judge_cost_usd,
            "routing": route,
            "tools_used": r.tools_used,
            "unsupported_numbers": (r.grounding or {}).get("unsupported", []),
            "latency_ms": r.latency_ms,
            "cost_usd": r.cost_usd,
            "usage": r.usage,
            "llm_steps": sum(1 for t in r.trace if t["kind"] == "tool"),
            "degraded": r.degraded,
            "models": r.models,
        }


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round(p / 100 * (len(xs) - 1))))
    return xs[k]


def summarize(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    judge_errors = sum(bool(r.get("judge_error")) for r in all_rows)
    rows = [r for r in all_rows if not r.get("judge_error")]  # judge failures aren't scored
    ok = [r for r in rows if "error" not in r]
    lat = [r["latency_ms"] / 1000 for r in rows]
    by: dict[str, dict[str, Any]] = {}
    for key in ("tier", "evaluation"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(str(r[key]), []).append(r)
        by[key] = {k: f"{sum(x['passed'] for x in v)}/{len(v)}" for k, v in sorted(groups.items())}
    routed = [r for r in ok if r["routing"]["required"]]
    return {
        "n": len(rows),
        "accuracy": round(sum(r["passed"] for r in rows) / max(len(rows), 1), 3),
        "mean_score": round(sum(r["score"] for r in rows) / max(len(rows), 1), 3),
        "by": by,
        "routing_accuracy": round(sum(r["routing"]["ok"] for r in routed) / max(len(routed), 1), 3),
        "answers_with_unsupported_numbers": sum(bool(r["unsupported_numbers"]) for r in ok),
        "errors": len(rows) - len(ok),
        "judge_errors_excluded": judge_errors,
        "latency_p50_s": round(pct(lat, 50), 1),
        "latency_p95_s": round(pct(lat, 95), 1),
        "mean_cost_usd": round(statistics.mean([r.get("cost_usd", 0) for r in rows]), 5)
        if rows
        else 0.0,
        "total_cost_usd": round(sum(r.get("cost_usd", 0) for r in rows), 4),
        "judge_cost_usd": round(sum(r.get("judge_cost_usd", 0) for r in all_rows), 4),
        "mean_prompt_tokens": int(statistics.mean([r["usage"]["prompt_tokens"] for r in ok]))
        if ok
        else 0,
        "mean_cached_tokens": int(statistics.mean([r["usage"]["cached_tokens"] for r in ok]))
        if ok
        else 0,
        "mean_completion_tokens": int(
            statistics.mean([r["usage"]["completion_tokens"] for r in ok])
        )
        if ok
        else 0,
        "mean_tool_calls": round(statistics.mean([r["llm_steps"] for r in ok]), 1) if ok else 0,
    }


def to_markdown(meta: dict[str, Any], runs: list[dict[str, Any]]) -> str:
    lines = [
        f"# Eval: {meta['model']}",
        f"prompt {meta['versions']['prompt']} · index {meta['versions']['index']} · "
        f"judge {meta['judge_model']} · {meta['started']}",
        "",
    ]
    if meta.get("note"):
        lines += [meta["note"], ""]
    accs = [r["summary"]["accuracy"] for r in runs]
    lines.append(
        f"**Accuracy over {len(runs)} run(s): mean {statistics.mean(accs):.0%} "
        f"(min {min(accs):.0%}, max {max(accs):.0%})**\n"
    )
    s = runs[0]["summary"]
    lines += ["| metric (run 1) | value |", "|---|---|"]
    for k in (
        "accuracy",
        "mean_score",
        "routing_accuracy",
        "answers_with_unsupported_numbers",
        "errors",
        "judge_errors_excluded",
        "judge_cost_usd",
        "latency_p50_s",
        "latency_p95_s",
        "mean_cost_usd",
        "total_cost_usd",
        "mean_prompt_tokens",
        "mean_cached_tokens",
        "mean_completion_tokens",
        "mean_tool_calls",
    ):
        lines.append(f"| {k} | {s[k]} |")
    for k, v in s["by"].items():
        lines.append(f"| by {k} | {', '.join(f'{a}: {b}' for a, b in v.items())} |")
    lines += [
        "",
        "| id | tier | scorer | pass (per run) | routing | tools | unsupported | "
        "latency s | $ | detail (run 1) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(runs[0]["rows"]):
        passes = "".join("✅" if run["rows"][i]["passed"] else "❌" for run in runs)
        route = r.get("routing", {})
        lines.append(
            f"| {r['id']} | {r['tier']} | {r['evaluation']} | {passes} | "
            f"{'ok' if route.get('ok') else 'MISS'} {','.join(route.get('used', []))} | "
            f"{','.join(r.get('tools_used', []))} | {len(r.get('unsupported_numbers', []))} | "
            f"{r['latency_ms'] / 1000:.1f} | {r.get('cost_usd', 0):.4f} | "
            f"{(r.get('score_detail') or r.get('error', ''))[:120].replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


async def main_async(args: argparse.Namespace) -> None:
    base = get_settings()
    # Free tiers make requests wait out rate limits; an eval measures answers, not that wait.
    overrides: dict[str, Any] = {"answer_cache": "none", "request_timeout_s": 900.0}
    if args.model:
        overrides["llm_model"] = args.model
    if args.no_escalation:
        overrides["escalation_model"] = ""
    settings = Settings(**{**base.model_dump(), **overrides})
    c = build_container(settings)
    if not c.ready or c.service is None:
        raise SystemExit("Not ready: " + "; ".join(c.problems))
    judge = None
    judge_model = args.judge_model or settings.judge_model or settings.llm_model
    if not args.no_judge:
        judge = OpenAICompatLLM(
            make_client(settings.api_key, settings.llm_base_url, 120),
            default_model=judge_model,
            temperature=0.0,
            max_tokens=2048,
            reasoning_effort=None,
            max_concurrency=4,
        )

    questions = load_questions()
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    sem = asyncio.Semaphore(args.concurrency)
    meta = {
        "model": settings.llm_model,
        "judge_model": judge_model,
        "versions": c.service.versions.model_dump(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "escalation": settings.escalation_model,
    }
    runs = []
    for i in range(args.runs):
        rows = await asyncio.gather(
            *(run_question(c.service, q, judge, judge_model, sem) for q in questions)
        )
        summary = summarize(list(rows))
        runs.append({"run": i + 1, "summary": summary, "rows": list(rows)})
        print(
            f"run {i + 1}: accuracy {summary['accuracy']:.0%} "
            f"routing {summary['routing_accuracy']:.0%} p50 {summary['latency_p50_s']}s "
            f"${summary['mean_cost_usd']:.4f}/q  {summary['by']['tier']}"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    slug = settings.llm_model.rsplit("/", 1)[-1]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"{slug}_{stamp}"  # not with_suffix: model names contain dots
    Path(f"{out}.json").write_text(json.dumps({"meta": meta, "runs": runs}, indent=2, default=str))
    Path(f"{out}.md").write_text(to_markdown(meta, runs))
    print(f"wrote {out}.json and .md")


async def regrade(path: Path, judge_model: str | None) -> None:
    """Re-run the judge on answers whose judging failed (e.g. a rate limit), keep the rest.
    With a different judge model, re-judge every LLM-judged answer so one judge grades all."""
    data = json.loads(path.read_text())
    settings = get_settings()
    model: str = judge_model or str(data["meta"]["judge_model"])
    switch = model != data["meta"]["judge_model"]
    data["meta"]["judge_model"] = model
    judge = OpenAICompatLLM(
        make_client(settings.api_key, settings.llm_base_url, 120),
        default_model=model,
        temperature=0.0,
        max_tokens=2048,
        reasoning_effort=None,
        max_concurrency=1,
    )
    for run in data["runs"]:
        for r in run["rows"]:
            if (
                r.get("evaluation") == "llm_judge"
                and "answer" in r
                and (switch or r.get("judge_error"))
            ):
                s = await scorers.llm_judge(judge, model, r["question"], r["gold"], r["answer"])
                r.update(
                    passed=s.passed,
                    score=s.score,
                    score_detail=s.detail,
                    judge_error=s.judge_error,
                    judge_cost_usd=r.get("judge_cost_usd", 0) + s.judge_cost_usd,
                )
                print(f"run {run['run']} {r['id']}: {'pass' if s.passed else 'fail'}", flush=True)
        run["summary"] = summarize(run["rows"])
    path.write_text(json.dumps(data, indent=2, default=str))
    (path.parent / f"{path.stem}.md").write_text(to_markdown(data["meta"], data["runs"]))
    print("accuracy:", [run["summary"]["accuracy"] for run in data["runs"]])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--model", help="override LLM_MODEL")
    p.add_argument("--judge-model", help="override JUDGE_MODEL")
    p.add_argument("--no-judge", action="store_true", help="skip llm_judge questions' grading")
    p.add_argument("--no-escalation", action="store_true")
    p.add_argument("--ids", help="comma-separated question ids to run")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--regrade", type=Path, help="re-judge failed judgments in this results .json")
    args = p.parse_args()
    if args.regrade:
        asyncio.run(regrade(args.regrade, args.judge_model))
    else:
        asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
