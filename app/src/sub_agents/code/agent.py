"""Code Agent: structured ``generate`` / ``explain`` / ``debug`` as a LangGraph reflection loop.

A single ``generate`` node returns a structured ``CodeOutput`` in one call; a ``judge`` node then
gates it and routes back to a ``refine`` node until the code is accepted or the retry budget is
spent. **Tier 1** (deterministic, ground truth) runs for languages with a registered parser (Python
via ``ast``, JavaScript via ``tree-sitter``): a syntax error routes back until the code parses or
``max_syntax_retries`` is hit, after which the best-effort code is returned with its ``parses``
state surfaced. **Tier 2** (LLM critic, fallback only) runs for parser-less languages: an
independent, cheaper reviewer decides ``revise`` vs ``return``. No code is ever executed. The
terminal state maps into the platform's uniform ``AgentResult`` with token totals and cost figures.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.src.general_utils.agent_base import AgentResult
from app.src.general_utils.cost import estimate_cost
from app.src.general_utils.llm import build_chat_model
from app.src.general_utils.tokens import count_prompt_tokens
from app.src.schemas.config import CodeAgentConfig, ModelPrice, get_config
from app.src.sub_agents.code import nodes, prompts
from app.src.sub_agents.code.routing import route_after_judge
from app.src.sub_agents.code.schemas import CodeInput, CodeState, coerce_action
from app.src.sub_agents.code.validation import has_validator

AGENT_NAME = "code"


def build_code_graph(
    generator: BaseChatModel,
    corrector: BaseChatModel,
    gen_price: ModelPrice,
    rev_price: ModelPrice,
) -> CompiledStateGraph[Any]:
    """Wire the generate -> judge -> (refine) reflection StateGraph.

    Args:
        generator: Model for the initial generate node.
        corrector: Model for the judge (critic) and refine nodes.
        gen_price: Price card for the generator model.
        rev_price: Price card for the corrector model.

    Returns:
        A compiled LangGraph runnable over ``CodeState``.
    """
    graph = StateGraph(CodeState)
    graph.add_node("generate", partial(nodes.generate_node, model=generator, price=gen_price))
    graph.add_node("judge", partial(nodes.judge_node, model=corrector, price=rev_price))
    graph.add_node("refine", partial(nodes.refine_node, model=corrector, price=rev_price))
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "judge")
    graph.add_conditional_edges("judge", route_after_judge, {"refine": "refine", END: END})
    graph.add_edge("refine", "judge")
    return graph.compile()


def initial_state(inp: CodeInput, max_rounds: int) -> CodeState:
    """Build the initial graph state from the validated input.

    Args:
        inp: The validated code request.
        max_rounds: Retry cap for the active validation tier.

    Returns:
        A fully-initialized CodeState.
    """
    return {
        "action": inp.action,
        "input": inp.input,
        "language": inp.language,
        "context": inp.context,
        "has_parser": has_validator(inp.language),
        "max_rounds": max_rounds,
        "content": "",
        "code": "",
        "out_language": inp.language,
        "tokens_used": 0,
        "cost": 0.0,
        "rounds": 0,
        "problem": None,
        "issues": [],
        "parses": None,
        "validation_error": "",
    }


def _output(final: CodeState) -> dict[str, Any]:
    """Map the terminal state into the platform output dict, surfacing the parse state."""
    payload: dict[str, Any] = {
        "content": final["content"],
        "code": final["code"],
        "language": final["out_language"],
        "parses": final["parses"],
    }
    if final["validation_error"]:
        payload["validation_error"] = final["validation_error"]
    return payload


def assemble_result(
    final: CodeState, step_id: str, est_cost: float, elapsed_ms: int
) -> AgentResult:
    """Wrap the terminal state and metrics into the platform's uniform AgentResult.

    Args:
        final: Terminal graph state.
        step_id: Orchestrator-assigned step identifier.
        est_cost: Pre-run USD cost estimate.
        elapsed_ms: Wall-clock duration of the full invoke.

    Returns:
        An AgentResult carrying the output dict plus token and pre/post cost figures.
    """
    return AgentResult(
        step_id=step_id,
        agent=AGENT_NAME,
        status="completed",
        output=_output(final),
        tokens_used=final["tokens_used"],
        execution_time_ms=elapsed_ms,
        est_cost_usd=round(est_cost, 6),
        actual_cost_usd=round(final["cost"], 6),
    )


def _estimate(
    gen_price: ModelPrice,
    rev_price: ModelPrice,
    cfg: CodeAgentConfig,
    parser_path: bool,
    input_tokens: int,
) -> float:
    """Estimate USD cost: one generator call plus the bounded corrective calls on the reviewer."""
    rev_calls = cfg.max_syntax_retries if parser_path else 2 * cfg.max_review_retries + 1
    generator = estimate_cost(gen_price, 1, input_tokens, cfg.avg_output_tokens)
    corrective = estimate_cost(rev_price, rev_calls, input_tokens, cfg.avg_output_tokens)
    return generator + corrective


def _max_rounds(cfg: CodeAgentConfig, parser_path: bool) -> int:
    """Pick the retry cap for the active validation tier."""
    return cfg.max_syntax_retries if parser_path else cfg.max_review_retries


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
        parser_path = has_validator(inp.language)
        graph = build_code_graph(generator, corrector, gen_price, rev_price)
        state = initial_state(inp, _max_rounds(cfg, parser_path))
        final = cast(CodeState, await graph.ainvoke(state))
        input_tokens = count_prompt_tokens(
            prompts.build_messages(inp), app_cfg.estimation.chars_per_token
        )
        est = _estimate(gen_price, rev_price, cfg, parser_path, input_tokens)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return assemble_result(final, step_id, est, elapsed_ms)
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
