"""Analysis-only tool: a deterministic ``compute`` statistics tool.

``compute`` offloads precise arithmetic and aggregation from the LLM (which is unreliable at math)
to safe, regex-guarded evaluation — never arbitrary code execution. It reads the working dataset and
its call budget from the injected runtime context, so it stays decoupled from the agent wiring via
the :class:`ComputeRuntimeContext` protocol. It lives in the analysis package (not
``general_utils``) because the ``dataset`` / ``max_compute_calls`` state it depends on exists only
on the Analysis Agent.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from statistics import mean
from typing import Annotated, Any, Protocol, runtime_checkable

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from app.src.general_utils.streaming import emit_status

logger = logging.getLogger(__name__)

_ARITHMETIC = re.compile(r"^[\d\s\.\+\-\*/\(\)]+$")
_NUMERIC_OPS = {"sum", "average", "min", "max"}


@runtime_checkable
class ComputeRuntimeContext(Protocol):
    """Structural contract the ``compute`` tool needs from the analysis runtime context."""

    dataset: list[dict[str, Any]]
    compute_count: int
    max_compute_calls: int


_EXPRESSION_DESC = (
    "Direct arithmetic expression to evaluate. No dataset needed. ONLY arithmetic operators: "
    "+ - * / () and numbers. Example: '(2.25 / (2 * 0.25)) * (1 / 7)'. Use EITHER expression "
    "(raw math) OR metrics (dataset aggregation)."
)
_METRICS_DESC = (
    "List of aggregations over the working dataset. Each is a dict with 'op' "
    "('count','count_where','sum','average','min','max','distinct','group_by'), 'field' (dot "
    "notation, e.g. 'price.amount'), and 'value' (for count_where). Results are m[0], m[1], ... "
    'Example: [{"op":"average","field":"score"}]'
)
_FORMULA_DESC = (
    "Arithmetic formula combining metric results via m[0], m[1], ... ONLY + - * / (). "
    "Example: m[0] * 100 / m[1]. If omitted, metric values are returned as-is."
)
_FILTER_FIELD_DESC = "Optional: keep only items where this field equals filter_value."
_FILTER_VALUE_DESC = "Value to match for the filter_field pre-filter."


@tool
def compute(
    runtime: ToolRuntime[ComputeRuntimeContext],
    expression: Annotated[str | None, _EXPRESSION_DESC] = None,
    metrics: Annotated[list[dict[str, Any]] | None, _METRICS_DESC] = None,
    formula: Annotated[str | None, _FORMULA_DESC] = None,
    filter_field: Annotated[str | None, _FILTER_FIELD_DESC] = None,
    filter_value: Annotated[str | None, _FILTER_VALUE_DESC] = None,
) -> dict[str, Any]:
    """Compute exact statistics over the working dataset OR evaluate raw arithmetic.

    Use this for any precise calculation — counts, sums, averages, ratios, percentages,
    comparisons — instead of computing in your head. TWO MODES (use one):
    1. Expression: pass `expression` for direct arithmetic, e.g. "3 * 120 + 1.929 * 80".
    2. Metrics: pass `metrics` (and optional `formula`) to aggregate the dataset provided to you.

    Only arithmetic operators are allowed in expressions and formulas; no other code runs.

    Returns:
        A dict with the result(s) and the remaining compute-call budget, or an ``error`` key.
    """
    ctx = runtime.context
    ctx.compute_count += 1
    remaining = ctx.max_compute_calls - ctx.compute_count
    if ctx.compute_count > ctx.max_compute_calls:
        logger.warning("compute budget reached (%d/%d)", ctx.compute_count, ctx.max_compute_calls)
        return {
            "error": "Compute call limit reached. Use results you already have.",
            "compute_calls_remaining": 0,
        }
    emit_status("Calculating...")
    if expression is not None:
        return _with_budget(_eval_expression(expression), remaining)
    metrics_result = _run_metrics(ctx.dataset, metrics, formula, filter_field, filter_value)
    return _with_budget(metrics_result, remaining)


def _with_budget(result: dict[str, Any], remaining: int) -> dict[str, Any]:
    """Attach the remaining compute-call budget to a result dict."""
    result["compute_calls_remaining"] = remaining
    return result


def _run_metrics(
    dataset: list[dict[str, Any]],
    metrics: list[dict[str, Any]] | None,
    formula: str | None,
    filter_field: str | None,
    filter_value: str | None,
) -> dict[str, Any]:
    """Validate inputs, pre-filter the dataset, and compute the requested metrics."""
    if not metrics:
        return {"error": "Provide either 'expression' or 'metrics'."}
    if not dataset:
        return {"error": "No dataset available to aggregate."}
    items = _apply_filter(dataset, filter_field, filter_value)
    return _compute_all(items, metrics, formula)


def _apply_filter(
    items: list[dict[str, Any]], filter_field: str | None, filter_value: str | None
) -> list[dict[str, Any]]:
    """Keep only items whose ``filter_field`` equals ``filter_value`` (no-op if unset)."""
    if not filter_field or filter_value is None:
        return items
    return [i for i in items if str(_resolve_field(i, filter_field)) == str(filter_value)]


def _compute_all(
    items: list[dict[str, Any]], metrics: list[dict[str, Any]], formula: str | None
) -> dict[str, Any]:
    """Compute every metric and optionally evaluate the combining formula."""
    values = [_compute_metric(items, m) for m in metrics]
    result: dict[str, Any] = {
        "metrics": [
            {"op": m.get("op"), "field": m.get("field", ""), "result": v}
            for m, v in zip(metrics, values, strict=False)
        ]
    }
    if formula:
        result["formula"] = formula
        result["result"] = _safe_eval_formula(formula, values)
    return result


def _compute_metric(items: list[dict[str, Any]], metric: dict[str, Any]) -> Any:
    """Compute a single metric value from a metric spec dict."""
    op = metric.get("op", "")
    field = metric.get("field", "")
    if op == "count":
        return len(items)
    if op == "count_where":
        target = str(metric.get("value"))
        return sum(1 for i in items if str(_resolve_field(i, field)) == target)
    if op == "distinct":
        return len({str(v) for i in items if (v := _resolve_field(i, field)) is not None})
    if op == "group_by":
        return _group_by(items, field)
    return _numeric_metric(items, field, op)


def _numeric_metric(items: list[dict[str, Any]], field: str, op: str) -> float:
    """Compute a numeric aggregation (sum/average/min/max) over a field."""
    numeric = [v for i in items if isinstance(v := _resolve_field(i, field), (int, float))]
    if not numeric or op not in _NUMERIC_OPS:
        return 0
    if op == "sum":
        return sum(numeric)
    if op == "average":
        return round(mean(numeric), 4)
    return min(numeric) if op == "min" else max(numeric)


def _group_by(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    """Count items by the distinct values of a field, dropping missing values."""
    from collections import Counter

    counter = Counter(
        str(v) for i in items if (v := _resolve_field(i, field)) is not None
    )
    return dict(counter.most_common())


def _resolve_field(item: dict[str, Any], field: str) -> Any:
    """Resolve a possibly dotted field path against a nested dict."""
    current: Any = item
    for part in field.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _eval_safe_arithmetic(expr: str) -> float | str:
    """Evaluate an arithmetic-only expression, rejecting anything with non-arithmetic characters."""
    if not _ARITHMETIC.match(expr):
        return f"Expression error: invalid characters in '{expr}'"
    try:
        return round(eval(expr), 6)  # noqa: S307 - guarded to digits/operators only
    except (ZeroDivisionError, SyntaxError) as exc:
        return f"Expression error: {exc}"


def _eval_expression(expr: str) -> dict[str, Any]:
    """Evaluate a raw arithmetic expression into a result dict."""
    result = _eval_safe_arithmetic(expr)
    if isinstance(result, str):
        return {"error": result}
    return {"expression": expr, "result": result}


def _safe_eval_formula(formula: str, values: Sequence[float]) -> float | str:
    """Substitute m[N] references with metric values, then evaluate arithmetic only."""
    try:
        expr = re.sub(r"m\[(\d+)\]", lambda m: str(values[int(m.group(1))]), formula)
    except IndexError as exc:
        return f"Formula error: {exc}"
    return _eval_safe_arithmetic(expr)
