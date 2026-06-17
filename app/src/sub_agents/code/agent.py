"""Code Agent: structured ``generate`` / ``explain`` / ``debug`` plus a two-tier validation gate.

There is no tool loop: the model returns a structured ``CodeOutput`` in one call, then a bounded
correction loop improves it. **Tier 1** (deterministic, ground truth) applies to languages with a
registered parser (Python via ``ast``, JavaScript via ``tree-sitter``): a syntax error feeds back
until the code parses or the ``max_syntax_retries`` budget is spent, after which the best-effort
code is returned with its ``parses`` state surfaced. **Tier 2** (LLM critic, fallback only) applies
to languages with no parser: an independent, cheaper reviewer model decides ``revise`` vs ``return``
and drives a bounded regeneration loop. No code is ever executed. The result maps into the
platform's uniform ``AgentResult`` with token totals and pre/post cost figures.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel

from app.src.general_utils.agent_base import AgentResult, Messages, extract_tokens
from app.src.general_utils.cost import estimate_cost, token_cost
from app.src.general_utils.llm import build_chat_model
from app.src.schemas.config import CodeAgentConfig, ModelPrice, get_config
from app.src.sub_agents.code import prompts
from app.src.sub_agents.code.schemas import CodeInput, CodeOutput, CodeVerdict, coerce_action
from app.src.sub_agents.code.validation import has_validator, validate_syntax

AGENT_NAME = "code"
_AVG_INPUT_TOKENS = 700
_AVG_OUTPUT_TOKENS = 450

SignalFn = Callable[[CodeOutput], Awaitable[tuple[str | None, int, float]]]
RefineFn = Callable[[CodeInput, str, str], Messages]


async def _invoke(
    model: BaseChatModel, schema: type[Any], messages: Messages, price: ModelPrice
) -> tuple[Any, int, float]:
    """Run one structured call, returning the parsed output plus its tokens and cost."""
    runnable = model.with_structured_output(schema, include_raw=True)
    result = cast(dict[str, Any], await runnable.ainvoke(messages))
    raw = result.get("raw")
    meta = getattr(raw, "usage_metadata", None) or {}
    in_tok = int(meta.get("input_tokens") or 0)
    out_tok = int(meta.get("output_tokens") or 0)
    return result["parsed"], extract_tokens(raw), token_cost(price, in_tok, out_tok)


async def _correct(
    model: BaseChatModel,
    out: CodeOutput,
    inp: CodeInput,
    price: ModelPrice,
    max_rounds: int,
    signal_fn: SignalFn,
    refine_fn: RefineFn,
) -> tuple[CodeOutput, int, float, str | None]:
    """Refine the output while ``signal_fn`` reports a problem, bounded by ``max_rounds``."""
    problem, tokens, cost = await signal_fn(out)
    rounds = 0
    while problem is not None and rounds < max_rounds:
        messages = refine_fn(inp, out.code, problem)
        out, r_tok, r_cost = await _invoke(model, CodeOutput, messages, price)
        rounds, tokens, cost = rounds + 1, tokens + r_tok, cost + r_cost
        problem, s_tok, s_cost = await signal_fn(out)
        tokens, cost = tokens + s_tok, cost + s_cost
    return out, tokens, cost, problem


def _parser_signal(language: str) -> SignalFn:
    """Build a free deterministic-parser signal for ``language`` (no token cost)."""

    async def signal(out: CodeOutput) -> tuple[str | None, int, float]:
        return validate_syntax(out.code, language), 0, 0.0

    return signal


def _critic_signal(model: BaseChatModel, inp: CodeInput, price: ModelPrice) -> SignalFn:
    """Build an LLM-critic signal: returns joined issues when the reviewer asks to revise."""

    async def signal(out: CodeOutput) -> tuple[str | None, int, float]:
        messages = prompts.critic_messages(inp, out.code, out.content)
        parsed, tokens, cost = await _invoke(model, CodeVerdict, messages, price)
        verdict = cast(CodeVerdict, parsed)
        if verdict.verdict != "revise":
            return None, tokens, cost
        return ("\n".join(verdict.issues) or "revision requested"), tokens, cost

    return signal


def _critic_refine(inp: CodeInput, code: str, problem: str) -> Messages:
    """Adapt the joined critic problem back to the issue-list refine prompt."""
    return prompts.review_refine_messages(inp, code, problem.splitlines())


async def _validate(
    corrector: BaseChatModel,
    out: CodeOutput,
    inp: CodeInput,
    cfg: CodeAgentConfig,
    price: ModelPrice,
) -> tuple[CodeOutput, int, float, bool | None, str]:
    """Run the right tier, returning output, tokens, cost, parse state, and any error message."""
    if has_validator(inp.language):
        out, tokens, cost, problem = await _correct(
            corrector,
            out,
            inp,
            price,
            cfg.max_syntax_retries,
            _parser_signal(inp.language),
            prompts.refine_messages,
        )
        return out, tokens, cost, problem is None, problem or ""
    signal = _critic_signal(corrector, inp, price)
    out, tokens, cost, _ = await _correct(
        corrector, out, inp, price, cfg.max_review_retries, signal, _critic_refine
    )
    return out, tokens, cost, None, ""


def _build_output(out: CodeOutput, parses: bool | None, error: str) -> dict[str, Any]:
    """Map the structured output into the platform output dict, surfacing the parse state."""
    payload: dict[str, Any] = {
        "content": out.content,
        "code": out.code,
        "language": out.language,
        "parses": parses,
    }
    if error:
        payload["validation_error"] = error
    return payload


def _assemble_result(
    output: dict[str, Any], step_id: str, tokens: int, cost: float, est_cost: float, elapsed_ms: int
) -> AgentResult:
    """Wrap the output dict and metrics into the platform's uniform AgentResult."""
    return AgentResult(
        step_id=step_id,
        agent=AGENT_NAME,
        status="completed",
        output=output,
        tokens_used=tokens,
        execution_time_ms=elapsed_ms,
        est_cost_usd=round(est_cost, 6),
        actual_cost_usd=round(cost, 6),
    )


def _estimate(
    gen_price: ModelPrice, rev_price: ModelPrice, cfg: CodeAgentConfig, parser_path: bool
) -> float:
    """Estimate USD cost: one generator call plus the bounded corrective calls on the reviewer."""
    rev_calls = cfg.max_syntax_retries if parser_path else 2 * cfg.max_review_retries + 1
    generator = estimate_cost(gen_price, 1, _AVG_INPUT_TOKENS, _AVG_OUTPUT_TOKENS)
    corrective = estimate_cost(rev_price, rev_calls, _AVG_INPUT_TOKENS, _AVG_OUTPUT_TOKENS)
    return generator + corrective


def _resolve_models(
    app_cfg: Any, cfg: CodeAgentConfig, model: BaseChatModel | None
) -> tuple[BaseChatModel, BaseChatModel]:
    """Return (generator, corrector) models; an injected model is reused for both (tests)."""
    if model is not None:
        return model, model
    api_key = app_cfg.google_api_key.get_secret_value()
    generator = build_chat_model(cfg.model_id, cfg.temperature, api_key)
    corrector = build_chat_model(cfg.review_model_id, cfg.review_temperature, api_key)
    return generator, corrector


def _build_input(action: str, task_input: str, language: str, context: str) -> CodeInput:
    """Construct the validated CodeInput, coercing an off-vocabulary action to ``generate``."""
    return CodeInput(
        action=coerce_action(action), input=task_input, language=language, context=context
    )


async def run_code_agent(
    task_input: str,
    action: str = "generate",
    step_id: str = "code",
    language: str | None = None,
    upstream_context: str = "",
    model: BaseChatModel | None = None,
) -> AgentResult:
    """Run the Code Agent end-to-end, returning a structured result on any failure.

    Args:
        task_input: The untrusted spec / code / error text for this step.
        action: One of ``generate`` / ``explain`` / ``debug`` (off-vocabulary degrades to generate).
        step_id: Orchestrator-assigned step identifier (echoed into the result).
        language: Target language; defaults to the configured ``default_language`` when omitted.
        upstream_context: Optional dependency output to ground the request (fenced as data).
        model: Optional injected model (defaults to the configured models); enables tests.

    Returns:
        An AgentResult; status ``failed`` (with an ``error`` field) on unrecoverable errors.
    """
    started = time.perf_counter()
    app_cfg = get_config()
    cfg = app_cfg.code_agent
    try:
        generator, corrector = _resolve_models(app_cfg, cfg, model)
        gen_price, rev_price = app_cfg.pricing[cfg.model_id], app_cfg.pricing[cfg.review_model_id]
        inp = _build_input(action, task_input, language or cfg.default_language, upstream_context)
        messages = prompts.build_messages(inp)
        parsed, tokens, cost = await _invoke(generator, CodeOutput, messages, gen_price)
        out, v_tok, v_cost, parses, error = await _validate(
            corrector, cast(CodeOutput, parsed), inp, cfg, rev_price
        )
        est = _estimate(gen_price, rev_price, cfg, has_validator(inp.language))
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        output = _build_output(out, parses, error)
        return _assemble_result(output, step_id, tokens + v_tok, cost + v_cost, est, elapsed_ms)
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return AgentResult(
            step_id=step_id,
            agent=AGENT_NAME,
            status="failed",
            output={"error": str(exc)},
            tokens_used=0,
            execution_time_ms=elapsed_ms,
        )
