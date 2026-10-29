"""Deterministic numeric grounding check.

Every number in the final answer must be traceable to the evidence: a tool result, a
calculate result, or the question itself. This is a heuristic with known limits. It checks
that a value exists in the evidence, not that it is attached to the right company or year.
Still, it is cheap (no LLM call), it catches the most damaging failure in financial QA
(an invented or mis-computed number), and it triggers one repair round.

Matching rules:
  * Units are normalised: "$36.1 billion" (answer) matches "36,147" (a 10-K table in
    millions) and 36147000000 (SQL, raw dollars). Evidence is tried at x1, x1e3, x1e6, x1e9.
  * Tolerance comes from the precision the answer states: "$1.5 billion" must be within one
    unit of the last digit (±$0.1B) of some evidence value. Rounding passes; invented
    numbers fail.
  * Ignored: years, citations in [brackets], the Sources line, page/item/note numbers, and
    small bare integers (counts and list markers).
"""

from __future__ import annotations

import re

from pydantic import BaseModel

_UNIT = {
    "trillion": 1e12,
    "tn": 1e12,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "thousand": 1e3,
    "k": 1e3,
}
_ANSWER_NUM = re.compile(
    r"(?P<pre>(?:item|note|page|p\.|fy|q|part)\s*)?"
    r"(?P<dollar>\$)?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<unit>trillion|billion|million|thousand|tn|bn|mn|[bmk])\b)?"
    r"(?P<pct>\s?%|\s?percent| percentage points?| pp\b)?"
    r"(?P<x>x\b)?",
    re.I,
)
_EVIDENCE_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_BRACKETS = re.compile(r"\[[^\]]*\]")
_SOURCES_TAIL = re.compile(r"\n\s*\**sources?\**\s*:.*\Z", re.I | re.S)
_SCALES = (1.0, 1e3, 1e6, 1e9)


class GroundingReport(BaseModel):
    checked: int
    unsupported: list[str]

    @property
    def ok(self) -> bool:
        return not self.unsupported


def evidence_numbers(texts: list[str]) -> list[float]:
    values: set[float] = set()
    for t in texts:
        for m in _EVIDENCE_NUM.finditer(t):
            values.add(float(m.group().replace(",", "")))
    return sorted(values)


def extract_claims(answer: str) -> list[tuple[str, float, float]]:
    """(literal, value_in_base_units, tolerance) for each checkable number in the answer."""
    body = _SOURCES_TAIL.sub("", answer)
    body = _BRACKETS.sub(" ", body)
    out = []
    for m in _ANSWER_NUM.finditer(body):
        if m["pre"]:
            continue
        raw = m["num"]
        value = float(raw.replace(",", ""))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        unit = _UNIT.get((m["unit"] or "").lower(), 1.0)
        is_money_or_pct = bool(m["dollar"] or m["unit"] or m["pct"] or m["x"])
        if not is_money_or_pct and decimals == 0 and (1900 <= value <= 2100 or value < 100):
            continue  # a year, count, ordinal or list marker
        tol = (10**-decimals) * unit  # one unit in the last stated digit
        out.append((m.group().strip(), value * unit, tol))
    return out


def _supported(value: float, tol: float, ev: list[float]) -> bool:
    v = abs(value)
    return any(abs(v - abs(e) * s) <= tol + 1e-9 * v for e in ev for s in _SCALES)


def check_grounding(answer: str, evidence: list[str]) -> GroundingReport:
    ev = evidence_numbers(evidence)
    claims = extract_claims(answer)
    unsupported = [lit for lit, value, tol in claims if not _supported(value, tol, ev)]
    return GroundingReport(checked=len(claims), unsupported=list(dict.fromkeys(unsupported)))


_OPERAND = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")
# Constants that legitimately appear in formulas: percent scaling, units, small integers.
_FORMULA_CONSTANTS = {100.0, 1e3, 1e6, 1e9, 1e12}


def unsupported_operands(expressions: list[str], evidence: list[str]) -> list[str]:
    """Numbers typed into `calculate` that don't appear in any earlier tool result.

    Catches transcription errors at the source (a filing says 40,367, the model types 40167).
    The answer-level grounding check can't see this, because the wrong result then *is* in
    the evidence: it is the calculator's own output.
    """
    ev = evidence_numbers(evidence)
    bad = []
    for expr in expressions:
        for m in _OPERAND.finditer(expr):
            raw = m.group()
            value = float(raw.replace(",", ""))
            if value in _FORMULA_CONSTANTS or (value < 10 and "." not in raw):
                continue
            decimals = len(raw.split(".")[1]) if "." in raw else 0
            tol = 10**-decimals
            # Operands carry no unit: 40.4 may mean billions while the filing is in millions.
            if not any(_supported(value * u, tol * u, ev) for u in _SCALES):
                bad.append(raw)
    return list(dict.fromkeys(bad))


# A page as the tools show it: "=== [MSFT FY2025 10-K p.38] Item 7 ..." and then its text.
_PAGE_BLOCK = re.compile(
    r"^=== \[(?P<label>[^\]\n]+? p\.\d+)\][^\n]*\n(?P<text>.*?)(?=^=== \[|\Z)", re.M | re.S
)
_CITATION = re.compile(r"\[([^\[\]\n]+)\]")
_PAGE_REF = re.compile(r"\bp\.(\d+)")
_LONE_NUMBER = re.compile(r"^\s*(\d{1,4})\s*$", re.M)  # a line that is only a number


def fix_citations(answer: str, evidence: list[str]) -> tuple[str, list[str]]:
    """Map citations of a filing's own printed page numbers to the pages actually read.

    Filings print page numbers that differ from the tools' page labels: the PDF page "MSFT
    FY2025 10-K p.38" prints "39", and a model reading it sometimes cites p.39. A cited page
    that was not read becomes the one read page of the same document that prints that number,
    if exactly one does. Returns the answer and a description of each change.
    """
    printed: dict[str, dict[int, set[str]]] = {}  # document -> page read -> lone numbers on it
    for text in evidence:
        for m in _PAGE_BLOCK.finditer(text):
            doc, _, page = m["label"].rpartition(" p.")
            printed.setdefault(doc, {})[int(page)] = set(_LONE_NUMBER.findall(m["text"]))
    changes: list[str] = []

    def fix(m: re.Match[str]) -> str:
        doc, sep, _ = m[1].partition(" p.")
        pages = printed.get(doc) if sep else None
        if not pages:
            return m[0]

        def page(pm: re.Match[str]) -> str:
            if int(pm[1]) in pages:
                return pm[0]
            shows = [p for p, numbers in pages.items() if pm[1] in numbers]
            if len(shows) != 1:
                return pm[0]
            changes.append(f"{doc} p.{pm[1]} -> p.{shows[0]}")
            return f"p.{shows[0]}"

        return "[" + _PAGE_REF.sub(page, m[1]) + "]"

    return _CITATION.sub(fix, answer), list(dict.fromkeys(changes))
