"""A filing as filed on EDGAR (HTML, usually with inline XBRL) → page texts and tagged facts.

One pass with the standard library's HTMLParser:
* Pages split where the filing breaks pages (`page-break-before/after: always`), so a page
  here is the page a reader sees, as with the PDFs.
* A table row becomes one line with its cells joined by " | " ("YouTube ads | 36,147 |
  40,367"), the same shape the PDF extractor produces.
* Inline XBRL: every tagged number (ix:nonFraction) with its context (period, plus dimensions
  such as segment or region) and the label of the table row it sits in. The row label is how
  the filing itself names a segment ("Google Cloud").
* Content hidden with display:none (the XBRL header) never reaches the page text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

_BLOCK = frozenset(
    {"p", "div", "br", "li", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6"}
    | {"section", "center", "blockquote", "pre", "dl", "dt", "dd"}
)
_VOID = frozenset(
    {"br", "hr", "img", "meta", "input", "link", "area", "base", "col", "embed", "wbr"}
    | {"source", "track"}
)
_WS = re.compile(r"\s+")


@dataclass
class Context:
    start: date | None = None
    end: date | None = None  # the instant, for point-in-time (balance sheet) facts
    instant: bool = False
    dims: dict[str, str] = field(default_factory=dict)  # axis -> member


@dataclass
class Fact:
    concept: str  # "us-gaap:Revenues"
    context: str  # context id
    value: float
    row_label: str  # first cell of the table row it appears in; "" outside tables


@dataclass
class ParsedFiling:
    pages: list[str]
    facts: list[Fact]
    contexts: dict[str, Context]
    dei: dict[str, list[str]]  # cover-page facts: "DocumentFiscalYearFocus" -> ["2024"]
    dei_context: str  # context of the cover-page facts: the filing's own fiscal period

    @property
    def company(self) -> str:
        """The registrant's name, in title case if filed in capitals ("MICROSOFT CORPORATION")."""
        name = (self.dei.get("EntityRegistrantName") or [""])[0].strip()
        if not name.isupper():
            return name
        words = []
        for w in name.split():
            core = w.strip(".,")  # "INC." -> "Inc.", "AMAZON.COM," -> "Amazon.com,"
            words.append(
                w.replace(core, core.capitalize()) if core.replace(".", "").isalpha() else w
            )
        return " ".join(words)


def _join_cells(cells: list[str]) -> str:
    """Row cells → "Label | $1,234 | (56)%", gluing "$", "(", ")" and "%" to their number."""
    out: list[str] = []
    for c in cells:
        if not c:
            continue
        if out and (c in {")", "%", ")%"} or out[-1] in {"$", "(", "($", "$("}):
            out[-1] += c
        else:
            out.append(c)
    return " | ".join(out)


def _number(text: str, fmt: str, scale: str, sign: str) -> float | None:
    fmt, t = fmt.lower(), text.strip()
    if "zero" in fmt or t in {"-", "—", "–"}:  # ixt:fixed-zero, ixt:zerodash
        value = 0.0
    elif "numwordsen" in fmt:  # "no", "none"
        if t.lower() not in {"no", "none", "zero"}:
            return None
        value = 0.0
    else:
        t = t.replace(" ", "")
        t = t.replace(".", "").replace(",", ".") if "comma" in fmt else t.replace(",", "")
        try:
            value = float(t)
        except ValueError:
            return None
    if scale.lstrip("-").isdigit():
        value *= 10 ** int(scale)
    return -value if sign == "-" else value


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pages: list[str] = []
        self.facts: list[Fact] = []
        self.contexts: dict[str, Context] = {}
        self.dei: dict[str, list[str]] = {}
        self.dei_context = ""
        self._lines: list[str] = []
        self._line: list[str] = []
        self._stack: list[tuple[str, bool, bool]] = []  # (tag, hidden, page break after)
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._fact: dict[str, str] | None = None
        self._dei: str | None = None
        self._ctx_id = ""
        self._ctx: Context | None = None
        self._member_axis = ""
        self._date_field = ""
        self._buf: list[str] = []

    def _hidden(self) -> bool:
        return bool(self._stack) and self._stack[-1][1]

    def _sink(self) -> list[str]:
        return self._cell if self._cell is not None else self._line

    def _end_line(self) -> None:
        text = _WS.sub(" ", "".join(self._line)).strip()
        if text:
            self._lines.append(text)
        self._line = []

    def end_page(self) -> None:
        self._end_line()
        self.pages.append("\n".join(self._lines))
        self._lines = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        style = a.get("style", "").replace(" ", "").lower()
        if "page-break-before:always" in style or "break-before:page" in style:
            self.end_page()
        break_after = "page-break-after:always" in style or "break-after:page" in style
        if tag in _BLOCK:
            if self._cell is not None:
                self._cell.append(" ")
            else:
                self._end_line()
        if tag == "tr":
            self._end_line()
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "span" and ("padding-left" in style or "margin-left" in style):
            self._sink().append(" ")  # spans laid out apart: "Item 7." "Management's ..."
        if tag == "ix:nonfraction":
            self._fact, self._buf = a, []
        elif tag == "ix:nonnumeric" and a.get("name", "").startswith("dei:"):
            self._dei, self._buf = a["name"][4:], []
            self.dei_context = self.dei_context or a.get("contextref", "")
        elif tag == "xbrli:context":
            self._ctx_id, self._ctx = a.get("id", ""), Context()
        elif tag == "xbrldi:explicitmember" and self._ctx is not None:
            self._member_axis, self._buf = a.get("dimension", ""), []
        elif tag in ("xbrli:startdate", "xbrli:enddate", "xbrli:instant") and self._ctx:
            self._date_field, self._buf = tag.split(":")[1], []
        if tag == "hr" and break_after:
            self.end_page()
        if tag not in _VOID:
            self._stack.append((tag, self._hidden() or "display:none" in style, break_after))

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                break_after = self._stack[i][2]
                del self._stack[i:]
                break
        else:
            return  # stray end tag
        text = _WS.sub(" ", "".join(self._buf)).strip()
        if tag == "ix:nonfraction" and self._fact is not None:
            self._add_fact(self._fact, text)
            self._fact = None
        elif tag == "ix:nonnumeric" and self._dei is not None:
            self.dei.setdefault(self._dei, []).append(text)
            self._dei = None
        elif tag == "xbrldi:explicitmember" and self._ctx is not None and self._member_axis:
            self._ctx.dims[self._member_axis] = text
            self._member_axis = ""
        elif self._ctx is not None and self._date_field and tag == f"xbrli:{self._date_field}":
            d = date.fromisoformat(text[:10])
            if self._date_field == "startdate":
                self._ctx.start = d
            else:
                self._ctx.end, self._ctx.instant = d, self._date_field == "instant"
            self._date_field = ""
        elif tag == "xbrli:context" and self._ctx is not None:
            self.contexts[self._ctx_id] = self._ctx
            self._ctx = None
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(_WS.sub(" ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            line = _join_cells(self._row)
            if line:
                self._lines.append(line)
            self._row = None
        elif tag in _BLOCK and self._cell is None:
            self._end_line()
        if break_after:
            self.end_page()

    def handle_data(self, data: str) -> None:
        if self._fact is not None or self._dei is not None or self._member_axis or self._date_field:
            self._buf.append(data)
        if not self._hidden():
            self._sink().append(data)

    def _add_fact(self, a: dict[str, str], text: str) -> None:
        value = _number(text, a.get("format", ""), a.get("scale", "0"), a.get("sign", ""))
        if value is None or not a.get("name") or not a.get("contextref"):
            return
        label = next((c for c in self._row if c), "") if self._row is not None else ""
        self.facts.append(Fact(a["name"], a["contextref"], value, label))


def parse_filing(path: Path) -> ParsedFiling:
    parser = _Parser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.close()
    parser.end_page()
    pages = parser.pages
    while pages and not pages[-1].strip():
        pages.pop()
    return ParsedFiling(pages, parser.facts, parser.contexts, parser.dei, parser.dei_context)
