"""How often does the LLM judge agree with a person? Label a sample of judged answers
yourself, without seeing the judge's verdict, then compare.

  uv run python -m eval.judge_agreement sample    # draw 30 judged answers
  uv run python -m eval.judge_agreement label     # label them in the terminal (resumable)
  uv run python -m eval.judge_agreement report    # agreement, Cohen's kappa, disagreements

The sample comes from every result file with LLM-judged answers (the main eval and
FinanceBench), balanced so judge failures are well represented. Labels are saved after each
answer to eval/judge_labels.json; the report goes to eval/results/judge_agreement.md.
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from typing import Any

from app.core.config import PROJECT_ROOT
from eval.financebench import answer_files, merged

RESULTS = PROJECT_ROOT / "eval" / "results"
LABELS = PROJECT_ROOT / "eval" / "judge_labels.json"


def judged_answers() -> list[dict[str, Any]]:
    """Every LLM-judged answer in eval/results, one per distinct (question, answer)."""
    items: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(RESULTS.glob("*.json")):
        data = json.loads(path.read_text())
        meta = data.get("meta", {})
        for run in data.get("runs", []):
            for r in run["rows"]:
                if r.get("evaluation") != "llm_judge" or r.get("judge_error") or "answer" not in r:
                    continue
                items.setdefault(
                    (r["question"], r["answer"]),
                    {
                        "id": f"{path.stem}:{r['id']}",
                        "source": "main eval",
                        "model": meta.get("model", ""),
                        "judge_model": meta.get("judge_model", ""),
                        "question": r["question"],
                        "reference": r["gold"],
                        "answer": r["answer"],
                        "judge_passed": r["passed"],
                        "judge_detail": r.get("score_detail", ""),
                    },
                )
    for path in answer_files():
        for r in merged(path.stem.removeprefix("financebench_")).values():
            if "passed" not in r or r.get("judge_error"):
                continue
            items.setdefault(
                (r["question"], r["answer"]),
                {
                    "id": f"{path.stem}:{r['id']}",
                    "source": "FinanceBench",
                    "model": path.stem.removeprefix("financebench_"),
                    "judge_model": r.get("judge_model", ""),
                    "question": r["question"],
                    "reference": r["gold"],
                    "answer": r["answer"],
                    "judge_passed": r["passed"],
                    "judge_detail": r.get("judge_detail", ""),
                },
            )
    return list(items.values())


def draw(pool: list[dict[str, Any]], n: int, seed: int = 7) -> list[dict[str, Any]]:
    """n answers to different questions, alternating between sources. At least a third are
    answers the judge failed, when there are that many, so both verdicts get checked."""
    rng = random.Random(seed)
    shuffled = rng.sample(pool, len(pool))
    seen: set[tuple[str, str]] = set()

    def pick(passed: bool, k: int) -> list[dict[str, Any]]:
        by_source: dict[str, list[dict[str, Any]]] = {}
        for x in shuffled:
            if x["judge_passed"] == passed:
                by_source.setdefault(x["source"], []).append(x)
        picked: list[dict[str, Any]] = []
        while len(picked) < k and any(by_source.values()):
            for group in by_source.values():
                while group and len(picked) < k:
                    x = group.pop()
                    if (x["source"], x["question"]) not in seen:
                        seen.add((x["source"], x["question"]))
                        picked.append(x)
                        break
        return picked

    sample = pick(False, n // 3)
    sample += pick(True, n - len(sample))
    sample += pick(False, n - len(sample))  # too few passed answers: more failed ones
    rng.shuffle(sample)
    return [{**x, "human": None} for x in sample]


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa for two raters' yes/no labels; None if chance agreement is total."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    p_a = sum(a for a, _ in pairs) / n
    p_b = sum(b for _, b in pairs) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def save(items: list[dict[str, Any]]) -> None:
    LABELS.write_text(json.dumps(items, indent=2) + "\n")


def sample(n: int) -> None:
    pool = judged_answers()
    if LABELS.exists() and any(x["human"] is not None for x in json.loads(LABELS.read_text())):
        raise SystemExit(f"{LABELS} already has labels; delete it to draw a new sample.")
    items = draw(pool, n)
    save(items)
    fails = sum(not x["judge_passed"] for x in items)
    print(f"{len(items)} answers drawn from {len(pool)} ({fails} judged wrong) -> {LABELS}")


def wrap(label: str, text: str) -> str:
    body = "\n".join(textwrap.fill(p, 96) for p in str(text).splitlines() if p.strip())
    return f"{label}\n{textwrap.indent(body, '  ')}"


def label() -> None:
    items = json.loads(LABELS.read_text())
    todo = [x for x in items if x["human"] is None]
    print(
        f"{len(todo)} of {len(items)} answers left. For each, decide whether the system answer "
        "is correct, by the judge's rule: it has every key fact and number of the reference "
        "(rounding is fine) and nothing contradicts it. Extra correct detail is fine. Grade "
        "against the reference as written, even where you think it is wrong (the judge is told "
        "to do the same); skip an answer you can't decide.\n"
    )
    for x in todo:
        done = sum(y["human"] is not None for y in items)
        print("=" * 100)
        print(f"[{done + 1}/{len(items)}] {x['source']}")
        print(wrap("QUESTION", x["question"]))
        print(wrap("REFERENCE ANSWER", x["reference"]))
        print(wrap("SYSTEM ANSWER", x["answer"]))
        choice = ""
        while choice not in {"y", "n", "s", "q"}:
            choice = input("Correct? [y]es / [n]o / [s]kip / [q]uit: ").strip().lower()
        if choice == "q":
            break
        x["human"] = {"y": True, "n": False, "s": "skip"}[choice]
        save(items)
    left = sum(x["human"] is None for x in items)
    print(
        f"\nSaved. {left} left; run label again to continue."
        if left
        else "\nAll labeled. Next: uv run python -m eval.judge_agreement report"
    )


def report() -> str:
    items = json.loads(LABELS.read_text())
    labeled = [x for x in items if isinstance(x["human"], bool)]
    pairs = [(x["judge_passed"], x["human"]) for x in labeled]
    if not pairs:
        raise SystemExit("No labels yet: run `uv run python -m eval.judge_agreement label`.")
    agree = sum(a == b for a, b in pairs)
    k = kappa(pairs)
    lines = [
        "# LLM judge vs. human labels",
        "",
        f"{len(labeled)} answers to different questions, labeled blind to the judge's verdict "
        f"({sum(x['human'] == 'skip' for x in items)} skipped). Answers the judge failed are "
        f"oversampled ({sum(not a for a, _ in pairs)} of {len(pairs)}) so both verdicts get "
        "checked.",
        "",
        "| | value |",
        "|---|---|",
        f"| agreement | {agree}/{len(pairs)} ({agree / len(pairs):.0%}) |",
        f"| Cohen's kappa | {k:.2f} |" if k is not None else "| Cohen's kappa | n/a |",
    ]
    for judge_says, human_says, name in (
        (True, False, "judge passed, human failed"),
        (False, True, "judge failed, human passed"),
    ):
        n = sum(a == judge_says and b == human_says for a, b in pairs)
        lines.append(f"| {name} | {n} |")
    for key, name in (("judge_model", "judge `{}`"), ("source", "{}")):
        for value in sorted({x[key] for x in labeled}):
            s = [(x["judge_passed"], x["human"]) for x in labeled if x[key] == value]
            lines.append(
                f"| agreement, {name.format(value)} | {sum(a == b for a, b in s)}/{len(s)} |"
            )
    disagreements = [x for x in labeled if x["judge_passed"] != x["human"]]
    if disagreements:
        lines += ["", "## Disagreements", ""]
        for x in disagreements:
            verdict = "passed" if x["judge_passed"] else "failed"
            lines.append(
                f"- `{x['id']}` (judge {x['judge_model']} {verdict} it): {x['question'][:140]} "
                f"Judge's reason: {x['judge_detail'][:300]}"
            )
    text = "\n".join(lines) + "\n"
    (RESULTS / "judge_agreement.md").write_text(text)
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    p_sample = sub.add_parser("sample")
    p_sample.add_argument("-n", type=int, default=30)
    sub.add_parser("label")
    sub.add_parser("report")
    args = p.parse_args()
    if args.command == "sample":
        sample(args.n)
    elif args.command == "label":
        label()
    else:
        print(report())


if __name__ == "__main__":
    main()
