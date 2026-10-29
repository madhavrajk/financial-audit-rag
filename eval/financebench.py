"""FinanceBench (Islam et al., 2023): 150 public questions over 84 filings of 32 companies.

  uv run python -m eval.financebench download  # questions + the 84 PDFs (166 MB)
  uv run python -m eval.financebench index     # page index of those PDFs
  uv run python -m eval.financebench run       # answer every question (resumable)
  uv run python -m eval.financebench grade     # LLM-judge every answer (resumable)
  uv run python -m eval.financebench report    # accuracy, evidence recall, latency

The agent sees only the documents (no database) and must find the right one among all 84
itself, which matches FinanceBench's "shared vector store" setting. Each answer is graded by
the main eval's LLM judge against FinanceBench's reference answer. Evidence recall is the
share of questions where the agent retrieved a page FinanceBench marks as evidence. Results
are appended to eval/results/financebench_<model>.jsonl after every question, so a run that
hits a daily quota resumes where it stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from app.adapters.openai_compat import OpenAICompatLLM, make_client
from app.container import build_container
from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.errors import AppError
from app.ingest.build_index import Document, build_index
from app.ingest.financials import SCHEMA
from eval import scorers

REPO = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
DATA = PROJECT_ROOT / "data" / "financebench"
RESULTS = PROJECT_ROOT / "eval" / "results"
QUESTIONS = "financebench_open_source.jsonl"
DOC_INFO = "financebench_document_information.jsonl"
FORMS = {"10k": "10-K", "10q": "10-Q", "8k": "8-K", "earnings": "Earnings release"}
TICKERS = {  # tickers as of the filings (Block was SQ until 2025)
    "3M": "MMM",
    "AES Corporation": "AES",
    "AMD": "AMD",
    "Activision Blizzard": "ATVI",
    "Adobe": "ADBE",
    "Amazon": "AMZN",
    "Amcor": "AMCR",
    "American Express": "AXP",
    "American Water Works": "AWK",
    "Best Buy": "BBY",
    "Block": "SQ",
    "Boeing": "BA",
    "CVS Health": "CVS",
    "Coca-Cola": "KO",
    "Corning": "GLW",
    "Costco": "COST",
    "Foot Locker": "FL",
    "General Mills": "GIS",
    "JPMorgan": "JPM",
    "Johnson & Johnson": "JNJ",
    "Kraft Heinz": "KHC",
    "Lockheed Martin": "LMT",
    "MGM Resorts": "MGM",
    "Microsoft": "MSFT",
    "Netflix": "NFLX",
    "Nike": "NKE",
    "Paypal": "PYPL",
    "PepsiCo": "PEP",
    "Pfizer": "PFE",
    "Ulta Beauty": "ULTA",
    "Verizon": "VZ",
    "Walmart": "WMT",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def questions() -> list[dict[str, Any]]:
    return read_jsonl(DATA / QUESTIONS)


def documents() -> dict[str, Document]:
    """FinanceBench doc_name -> Document, for the documents the questions use."""
    info = {d["doc_name"]: d for d in read_jsonl(DATA / DOC_INFO)}
    docs = {}
    for name in sorted({q["doc_name"] for q in questions()}):
        d = info[name]
        form, year = FORMS[d["doc_type"].lower()], int(d["doc_period"])
        if form == "8-K":
            period = re.search(r"\d{4}-\d{2}-\d{2}", name).group()  # type: ignore[union-attr]
        elif m := re.search(r"_(\d{4})(Q[1-4])_", name):
            period = f"FY{m[1]} {m[2]}"
        else:
            period = ""
        docs[name] = Document(
            DATA / "pdfs" / f"{name}.pdf",
            ticker=TICKERS[d["company"]],
            fiscal_year=year,
            form=form,
            period=period,
            company=d["company"],
        )
    return docs


def doc_label(doc: Document) -> str:
    return f"{doc.ticker} {doc.period or f'FY{doc.fiscal_year}'} {doc.form}"


def download() -> None:
    (DATA / "pdfs").mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for name in (QUESTIONS, DOC_INFO):
            resp = client.get(f"{REPO}/data/{name}")
            resp.raise_for_status()
            (DATA / name).write_bytes(resp.content)
        names = sorted({q["doc_name"] for q in questions()})
        for i, name in enumerate(names, 1):
            path = DATA / "pdfs" / f"{name}.pdf"
            if path.exists() and path.read_bytes()[:4] == b"%PDF":
                continue
            resp = client.get(f"{REPO}/pdfs/{name}.pdf")
            resp.raise_for_status()
            if resp.content[:4] != b"%PDF":
                raise SystemExit(f"{name}: not a PDF")
            path.write_bytes(resp.content)
            print(f"[{i}/{len(names)}] {name}.pdf {len(resp.content) / 1e6:.1f} MB", flush=True)
    print(f"{len(questions())} questions, {len(names)} documents in {DATA}")


def index() -> None:
    settings = get_settings()
    embedder = None
    if settings.embed_model:
        from app.ingest.build_index import make_embedder

        embedder = make_embedder(required=False)
    docs = list(documents().values())
    stats = asyncio.run(build_index(docs, DATA / "index", embedder, settings.embed_dims))
    # No database: an empty one with the schema, so query_database is not offered.
    with sqlite3.connect(DATA / "no-financials.db") as empty:
        empty.executescript(SCHEMA)
    print(f"index {stats['index_version']}: {stats['pages']} pages from {len(docs)} documents")


def results_path(model: str) -> Path:
    """Answers, one line per question; grades go next to it in .grades.jsonl so answering
    and grading can run at the same time without sharing a file."""
    return RESULTS / f"financebench_{model.rsplit('/', 1)[-1]}.jsonl"


def grades_path(model: str) -> Path:
    return results_path(model).with_suffix(".grades.jsonl")


def answer_files() -> list[Path]:
    return [p for p in sorted(RESULTS.glob("financebench_*.jsonl")) if ".grades" not in p.name]


def latest(path: Path) -> dict[str, dict[str, Any]]:
    """The last record per question id (records are appended as they change)."""
    return {r["id"]: r for r in read_jsonl(path)}


def answer_key(answer: str) -> str:
    return hashlib.sha256(answer.encode()).hexdigest()[:16]


def merged(model: str) -> dict[str, dict[str, Any]]:
    """Answers with their grade, if the grade is for that exact answer."""
    rows = latest(results_path(model))
    for qid, g in latest(grades_path(model)).items():
        row = rows.get(qid)
        if row and "answer" in row and g.get("answer_key") == answer_key(row["answer"]):
            row.update({k: v for k, v in g.items() if k not in ("id", "answer_key")})
    return rows


def append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def daily_quota(exc: Exception) -> bool:
    """The provider's quota is used up for hours (see with_retries), not just this minute."""
    return "quota for this api key is used up" in str(exc).lower()


async def run(limit: int | None, retry_errors: bool) -> None:
    base = get_settings()
    settings = Settings(
        **{
            **base.model_dump(),
            "index_dir": DATA / "index",
            "db_path": DATA / "no-financials.db",
            "answer_cache": "none",
            "request_timeout_s": 900.0,  # free tiers make requests wait out rate limits
        }
    )
    c = build_container(settings)
    if c.service is None or not c.ready:
        raise SystemExit("Not ready: " + "; ".join(c.problems) + " (run download and index)")
    docs = documents()
    out = results_path(settings.llm_model)
    done = latest(out)
    todo = [
        q
        for q in questions()
        if q["financebench_id"] not in done
        or (retry_errors and "error" in done[q["financebench_id"]])
    ][:limit]
    print(f"{len(done)} already answered; answering {len(todo)} with {settings.llm_model}")
    for i, q in enumerate(todo, 1):
        evidence = {
            f"{doc_label(docs[e['doc_name']])} p.{e['evidence_page_num'] + 1}"
            for e in q["evidence"]
            if e["doc_name"] in docs
        }
        record: dict[str, Any] = {
            "id": q["financebench_id"],
            "company": q["company"],
            "document": doc_label(docs[q["doc_name"]]),
            "question_type": q["question_type"],
            "question": q["question"],
            "gold": q["answer"],
            "evidence_pages": sorted(evidence),
        }
        t0 = time.monotonic()
        try:
            r = await c.service.ask(q["question"], use_cache=False)
        except AppError as exc:
            record.update(error=str(exc), latency_ms=int((time.monotonic() - t0) * 1000))
            append(out, record)
            print(f"[{i}/{len(todo)}] {record['id']} ERROR {str(exc)[:120]}", flush=True)
            if daily_quota(exc):
                print("Daily quota reached; run again tomorrow to continue.")
                return
            continue
        retrieved = sorted({s["label"] for s in r.sources if s.get("kind") == "filing"})
        record.update(
            answer=r.answer,
            retrieved_pages=retrieved,
            evidence_hit=bool(evidence & set(retrieved)),
            right_document=any(p.startswith(record["document"]) for p in retrieved),
            tools_used=r.tools_used,
            unsupported_numbers=(r.grounding or {}).get("unsupported", []),
            latency_ms=r.latency_ms,
            usage=r.usage,
            models=r.models,
            versions=r.versions.model_dump(),
        )
        append(out, record)
        hit = "evidence" if record["evidence_hit"] else "-"
        print(f"[{i}/{len(todo)}] {record['id']} {r.latency_ms / 1000:.1f}s {hit}", flush=True)


async def grade(judge_model: str | None, model: str | None) -> None:
    settings = get_settings()
    judge_model = judge_model or settings.judge_model or settings.llm_model
    judge = OpenAICompatLLM(
        make_client(settings.api_key, settings.llm_base_url, 120),
        default_model=judge_model,
        temperature=0.0,
        max_tokens=2048,
        reasoning_effort=None,
        max_concurrency=2,
    )
    model_name = model or settings.llm_model
    # Ungraded, failed to grade, or graded by another judge: one judge grades every answer.
    rows = [
        r
        for r in merged(model_name).values()
        if "answer" in r
        and ("passed" not in r or r.get("judge_error") or r.get("judge_model") != judge_model)
    ]
    print(f"grading {len(rows)} answers with {judge_model}")
    for i, row in enumerate(rows, 1):
        s = await scorers.llm_judge(judge, judge_model, row["question"], row["gold"], row["answer"])
        grade_record = {
            "id": row["id"],
            "answer_key": answer_key(row["answer"]),
            "passed": s.passed,
            "score": s.score,
            "judge_detail": s.detail,
            "judge_error": s.judge_error,
            "judge_model": judge_model,
        }
        append(grades_path(model_name), grade_record)
        print(f"[{i}/{len(rows)}] {row['id']} {'pass' if s.passed else 'fail'}", flush=True)
        if s.judge_error and daily_quota(Exception(s.detail)):
            print("Daily quota reached; run again tomorrow to continue.")
            return


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, round(p / 100 * (len(xs) - 1))))] if xs else 0.0


def report(model: str | None) -> str:
    settings = get_settings()
    out = results_path(model or settings.llm_model)
    rows = list(merged(model or settings.llm_model).values())
    answered = [r for r in rows if "answer" in r]
    graded = [r for r in answered if "passed" in r and not r.get("judge_error")]
    lat = [r["latency_ms"] / 1000 for r in answered]
    lines = [
        f"# FinanceBench (open-source 150): {out.stem.removeprefix('financebench_')}",
        "",
        f"Answered {len(answered)}/{len(rows)}; graded {len(graded)}. Documents only (no "
        "database), all 84 filings in one index, keyword search"
        + (", judge " + graded[0]["judge_model"] if graded else "")
        + ".",
        "",
        "| | value |",
        "|---|---|",
    ]
    if graded:
        lines.append(
            f"| accuracy | {sum(r['passed'] for r in graded)}/{len(graded)} "
            f"({sum(r['passed'] for r in graded) / len(graded):.0%}) |"
        )
        for qt in ("metrics-generated", "domain-relevant", "novel-generated"):
            g = [r for r in graded if r["question_type"] == qt]
            if g:
                lines.append(f"| accuracy, {qt} | {sum(r['passed'] for r in g)}/{len(g)} |")
    if answered:
        hit = sum(r["evidence_hit"] for r in answered)
        right = sum(r["right_document"] for r in answered)
        lines += [
            f"| retrieved an evidence page | {hit}/{len(answered)} ({hit / len(answered):.0%}) |",
            f"| retrieved from the right document | {right}/{len(answered)} |",
            f"| answers with unsupported numbers | "
            f"{sum(bool(r['unsupported_numbers']) for r in answered)} |",
            f"| latency p50 / p95, including rate-limit waits | {pct(lat, 50):.1f} s / "
            f"{pct(lat, 95):.1f} s |",
            f"| mean prompt tokens per question | "
            f"{int(statistics.mean(r['usage'].get('prompt_tokens', 0) for r in answered))} |",
        ]
    errors = [r for r in rows if "error" in r and "answer" not in r]
    if errors:
        lines.append(f"| errors (not answered) | {len(errors)} |")
    text = "\n".join(lines) + "\n"
    out.with_suffix(".md").write_text(text)
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    sub.add_parser("index")
    p_run = sub.add_parser("run")
    p_run.add_argument("--limit", type=int, help="answer at most this many questions")
    p_run.add_argument("--retry-errors", action="store_true")
    p_grade = sub.add_parser("grade")
    p_grade.add_argument("--judge-model", help="defaults to JUDGE_MODEL, then LLM_MODEL")
    p_grade.add_argument("--model", help="whose answers to grade; defaults to LLM_MODEL")
    p_report = sub.add_parser("report")
    p_report.add_argument("--model", help="defaults to LLM_MODEL")
    args = p.parse_args()
    if args.command == "download":
        download()
    elif args.command == "index":
        index()
    elif args.command == "run":
        asyncio.run(run(args.limit, args.retry_errors))
    elif args.command == "grade":
        asyncio.run(grade(args.judge_model, args.model))
    else:
        print(report(args.model))


if __name__ == "__main__":
    main()
