"""Safe arithmetic for the agent.

LLMs are unreliable at multi-digit arithmetic (FinQA/TAT-QA show this is the dominant error
class in financial QA), so every growth rate, ratio, share, and difference goes through this
tool. Expressions are parsed with `ast` and evaluated against a whitelist. There is no `eval`,
no names, and no attribute access, so model-written input cannot execute code.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from typing import Any

from app.core.errors import ToolError

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_FUNCS: dict[str, Callable[..., Any]] = {"abs": abs, "round": round, "min": min, "max": max}
_MAX_EXPR_CHARS = 500
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")  # "416,161" -> "416161"; keeps arg commas
_MAX_EXPONENT = 100


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ToolError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ToolError(
        f"unsupported syntax: {ast.dump(node)[:80]}. Use numbers, + - * / ** %, parentheses, "
        "and abs/round/min/max only."
    )


def evaluate(expression: str) -> float:
    expr = _THOUSANDS.sub("", expression).strip()
    if not expr:
        raise ToolError("empty expression")
    if len(expr) > _MAX_EXPR_CHARS:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression syntax: {exc.msg}") from exc
    try:
        value = float(_eval(tree))
    except ZeroDivisionError as exc:
        raise ToolError("division by zero, so the ratio/growth is undefined") from exc
    except OverflowError as exc:
        raise ToolError("numeric overflow") from exc
    if math.isnan(value) or math.isinf(value):
        raise ToolError("result is not a finite number")
    return value


def calculate_many(expressions: dict[str, str]) -> dict[str, Any]:
    """Evaluate several labelled expressions in one call, which saves agent round-trips.

    With two or more results it also returns them ranked highest-first. Models compute
    correctly but sometimes compare wrongly ("Apple 19.5% grew fastest" while Alphabet was
    32.0%), so ranking is done deterministically here instead.
    """
    results: dict[str, Any] = {}
    values: dict[str, float] = {}
    for label, expr in expressions.items():
        try:
            value = evaluate(expr)
            results[label] = {"expression": expr, "value": round(value, 6)}
            values[label] = value
        except ToolError as exc:
            results[label] = {"expression": expr, "error": str(exc)}
    if len(values) >= 2:
        results["ranking_highest_first"] = sorted(values, key=lambda k: -values[k])
    return results
