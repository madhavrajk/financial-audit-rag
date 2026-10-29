from __future__ import annotations

import pytest

from app.core.grounding import check_grounding

EVIDENCE = [
    "company_ticker | revenue\nAAPL | 416161000000\nAAPL | 391035000000",
    "YouTube ads | 36,147 | 40,367\nOther Bets | 1,648 | 1,537\nHedging | 211 | (127)",
    '{"g": {"value": 6.425512}, "us_share": {"value": 51.307958}, "cr": {"value": 2.005354}}',
]


@pytest.mark.parametrize(
    "answer",
    [
        "Apple's net sales in fiscal year 2025 were $416.2 billion, up 6.4% from "
        "$391.0 billion.\nSources: [AAPL FY2025 10-K p.24], [SQL: income_statements]",
        "YouTube ads grew from $36.1 billion to $40.4 billion.",  # table in millions → billions
        "Other Bets declined to $1.5 billion; hedging was a $127 million loss.",
        "The US share was 51.3%; the current ratio was 2.01x.",
        "Apple has 5 product categories and discusses this in Item 7 of its FY2025 10-K.",
    ],
)
def test_supported_numbers_pass(answer):
    report = check_grounding(answer, EVIDENCE)
    assert report.ok, report.unsupported


def test_invented_numbers_are_flagged():
    report = check_grounding("Revenue was $420.5 billion, up 7.9%.", EVIDENCE)
    assert report.unsupported == ["$420.5 billion", "7.9%"]


def test_precision_matters():
    # $1.5B is a fine rounding of 1,537 (millions), but $1.6B is not.
    assert check_grounding("Other Bets: $1.5 billion", EVIDENCE).ok
    assert not check_grounding("Other Bets: $1.4 billion", EVIDENCE).ok


def test_calculator_operands_must_come_from_evidence():
    from app.core.grounding import unsupported_operands

    evidence = ["YouTube ads | 36,147 | 40,367"]
    assert unsupported_operands(["(40367 - 36147) / 36147 * 100"], evidence) == []
    assert unsupported_operands(["(40167 - 36147) / 36147 * 100"], evidence) == ["40167"]
    assert unsupported_operands(["40.4 - 36.1"], evidence) == []  # rounded billions are fine


PAGES = (
    "=== [MSFT FY2025 10-K p.38] Item 7. MD&A\nAzure and other cloud services revenue grew 34%.\n"
    "39\nPART II\nItem 7\n\n"
    "=== [MSFT FY2025 10-K p.40] Item 7. MD&A\nMore Personal Computing\n41\n"
)


def test_citing_a_printed_page_number_points_to_the_page_read():
    from app.core.grounding import fix_citations

    answer = "Azure grew 34% [MSFT FY2025 10-K p.39].\nSources: [MSFT FY2025 10-K p.39, p.40]"
    fixed, changes = fix_citations(answer, ["question", PAGES])
    assert (
        fixed == "Azure grew 34% [MSFT FY2025 10-K p.38].\nSources: [MSFT FY2025 10-K p.38, p.40]"
    )
    assert changes == ["MSFT FY2025 10-K p.39 -> p.38"]


@pytest.mark.parametrize(
    "answer",
    [
        "Azure grew 34% [MSFT FY2025 10-K p.38]. Sources: [SQL: segment_revenue]",  # read
        "Azure grew 34% [MSFT FY2025 10-K p.12]",  # no page read prints 12: left alone
        "Azure grew 34% [AAPL FY2025 10-K p.39]",  # another document
    ],
)
def test_other_citations_are_left_alone(answer):
    from app.core.grounding import fix_citations

    assert fix_citations(answer, [PAGES]) == (answer, [])
