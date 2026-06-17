"""LangGraph node functions for the Code Agent reflection loop.

Three nodes drive a generate -> judge -> (refine -> judge) loop. The judge node runs the
deterministic Tier-1 parser when ``has_parser`` is set, otherwise the Tier-2 LLM critic; the refine
node mirrors that split when correcting. Counters (``tokens_used``, ``cost``, ``rounds``) use
additive reducers so each node returns only its delta. No code is ever executed.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel

from app.src.general_utils.agent_base import Messages, extract_tokens
from app.src.general_utils.cost import token_cost
from app.src.schemas.config import ModelPrice
from app.src.sub_agents.code import prompts
from app.src.sub_agents.code.schemas import CodeInput, CodeOutput, CodeState, CodeVerdict
from app.src.sub_agents.code.validation import validate_syntax


async def _invoke(
    model: BaseChatModel, schema: type[Any], messages: Messages, price: ModelPrice
) -> tuple[Any, int, float]:
    """Run one structured call, returning the parsed output plus its tokens and cost."""
    runnable = model.with_structured_output(schema, include_raw=True)
    result = cast(dict[str, Any], await runnable.ainvoke(messages))
    raw = result.get("raw")
    meta = getattr(raw, "usage_metadata", None) or {}
    in_tok, out_tok = int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0)
    return result["parsed"], extract_tokens(raw), token_cost(price, in_tok, out_tok)


def _input_from_state(state: CodeState) -> CodeInput:
    """Rebuild the validated CodeInput from graph state for the prompt builders."""
    return CodeInput(
        action=state["action"],
        input=state["input"],
        language=state["language"],
        context=state["context"],
    )


def _code_delta(out: CodeOutput, tokens: int, cost: float) -> dict[str, Any]:
    """Shape the shared state delta a generate or refine call contributes."""
    return {
        "content": out.content,
        "code": out.code,
        "out_language": out.language,
        "tokens_used": tokens,
        "cost": cost,
    }


async def generate_node(
    state: CodeState, *, model: BaseChatModel, price: ModelPrice
) -> dict[str, Any]:
    """Produce the first structured code result from the fenced request."""
    messages = prompts.build_messages(_input_from_state(state))
    parsed, tokens, cost = await _invoke(model, CodeOutput, messages, price)
    return _code_delta(cast(CodeOutput, parsed), tokens, cost)


async def judge_node(
    state: CodeState, *, model: BaseChatModel, price: ModelPrice
) -> dict[str, Any]:
    """Gate the current code: the Tier-1 parser when available, else the Tier-2 LLM critic."""
    if state["has_parser"]:
        problem = validate_syntax(state["code"], state["language"])
        return {"problem": problem, "parses": problem is None, "validation_error": problem or ""}
    return await _critic_judge(state, model, price)


async def _critic_judge(
    state: CodeState, model: BaseChatModel, price: ModelPrice
) -> dict[str, Any]:
    """Run the Tier-2 critic and translate its verdict into a routing delta."""
    messages = prompts.critic_messages(_input_from_state(state), state["code"], state["content"])
    parsed, tokens, cost = await _invoke(model, CodeVerdict, messages, price)
    verdict = cast(CodeVerdict, parsed)
    problem = (
        None
        if verdict.verdict != "revise"
        else ("\n".join(verdict.issues) or "revision requested")
    )
    return {"problem": problem, "issues": verdict.issues, "tokens_used": tokens, "cost": cost}


async def refine_node(
    state: CodeState, *, model: BaseChatModel, price: ModelPrice
) -> dict[str, Any]:
    """Regenerate the code to fix the open problem, mirroring the active validation tier."""
    parsed, tokens, cost = await _invoke(model, CodeOutput, _refine_messages(state), price)
    return {**_code_delta(cast(CodeOutput, parsed), tokens, cost), "rounds": 1}


def _refine_messages(state: CodeState) -> Messages:
    """Pick the syntax-fix or critic-fix prompt for the active tier."""
    inp = _input_from_state(state)
    problem = state["problem"] or ""
    if state["has_parser"]:
        return prompts.refine_messages(inp, state["code"], problem)
    return prompts.review_refine_messages(inp, state["code"], problem.splitlines())
